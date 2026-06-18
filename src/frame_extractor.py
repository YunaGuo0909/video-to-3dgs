"""
Frame extraction from video with adaptive quality filtering.

Extracts sharp, temporally-spaced frames suitable for Structure-from-Motion.
Blur detection uses Laplacian variance; the threshold adapts to video content
so the sharpest ~75% of frames are kept regardless of camera or scene.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

logger = logging.getLogger(__name__)


@dataclass
class ExtractionStats:
    total_decoded: int = 0
    accepted: int = 0
    rejected_blur: int = 0
    rejected_stride: int = 0

    def summary(self) -> str:
        return (
            f"Decoded {self.total_decoded} → accepted {self.accepted} "
            f"(blur={self.rejected_blur}, stride={self.rejected_stride})"
        )


class FrameExtractor:
    """Extract high-quality frames from a video file.

    Parameters
    ----------
    blur_threshold : float
        Laplacian variance below this → frame discarded as blurry.
    min_frame_gap_ms : float
        Minimum milliseconds between consecutive accepted frames.
    """

    def __init__(
        self,
        blur_threshold: float = 80.0,
        min_frame_gap_ms: float = 250.0,
    ) -> None:
        self.blur_threshold = blur_threshold
        self.min_frame_gap_ms = min_frame_gap_ms

    @classmethod
    def from_video(cls, video_path: Path | str) -> "FrameExtractor":
        """Create an extractor with thresholds adapted to the input video."""
        scan = cls._scan_video(Path(video_path))

        blur_threshold = float(np.clip(scan["sharpness_p25"], 30.0, 200.0))
        motion = scan["motion_mean"]
        if motion > 15.0:
            gap = 150.0
        elif motion > 8.0:
            gap = 200.0
        else:
            gap = 300.0

        logger.info(
            "Adaptive: blur_thresh=%.1f  gap_ms=%.0f  (motion=%.1f)",
            blur_threshold, gap, motion,
        )
        return cls(blur_threshold=blur_threshold, min_frame_gap_ms=gap)

    def extract(self, video_path: Path | str, output_dir: Path | str) -> ExtractionStats:
        """Extract frames from video into output_dir/frame_XXXXXX.jpg."""
        video_path, output_dir = Path(video_path), Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        stats = ExtractionStats()
        last_ms = -self.min_frame_gap_ms

        with tqdm(total=total, desc="Extracting frames", unit="fr") as pbar:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                stats.total_decoded += 1
                ts = cap.get(cv2.CAP_PROP_POS_MSEC)

                if ts - last_ms < self.min_frame_gap_ms:
                    stats.rejected_stride += 1
                    pbar.update(1)
                    continue

                sharpness = cv2.Laplacian(
                    cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), cv2.CV_64F
                ).var()
                if sharpness < self.blur_threshold:
                    stats.rejected_blur += 1
                    pbar.update(1)
                    continue

                out = output_dir / f"frame_{stats.accepted:06d}.jpg"
                cv2.imwrite(str(out), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                last_ms = ts
                stats.accepted += 1
                pbar.update(1)

        cap.release()
        logger.info(stats.summary())
        return stats

    @staticmethod
    def _scan_video(video_path: Path, n_samples: int = 60) -> dict:
        """Pre-scan video for sharpness and motion statistics."""
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open: {video_path}")

        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        indices = np.linspace(0, total - 1, min(n_samples, total), dtype=int)

        sharpness: list[float] = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ret, frame = cap.read()
            if ret:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                sharpness.append(float(cv2.Laplacian(gray, cv2.CV_64F).var()))

        # Motion: consecutive-frame pairs in 3 local windows
        motion: list[float] = []
        for start in [total // 8, total // 2, total * 6 // 8]:
            prev = None
            for off in range(10):
                cap.set(cv2.CAP_PROP_POS_FRAMES, min(start + off, total - 1))
                ret, frame = cap.read()
                if not ret:
                    break
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                if prev is not None:
                    motion.append(float(np.mean(np.abs(
                        gray.astype(np.float32) - prev.astype(np.float32)
                    ))))
                prev = gray

        cap.release()
        sh = np.array(sharpness) if sharpness else np.array([100.0])
        mo = np.array(motion) if motion else np.array([10.0])
        return {
            "sharpness_p25": float(np.percentile(sh, 25)),
            "motion_mean": float(np.mean(mo)),
        }
