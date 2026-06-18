"""
Video to 3D Gaussian Splatting — end-to-end reconstruction pipeline.

Takes a short video of an indoor scene (e.g. captured on a phone) and
produces a 3D Gaussian Splatting reconstruction viewable in any
standard 3DGS viewer.

Usage
-----
    python run.py --video room.mp4 --output results/

The pipeline runs four stages:
  1. Frame extraction — adaptive blur/stride filtering
  2. SfM (COLMAP) — camera pose estimation via pycolmap
  3. Training — 3D Gaussian Splatting with MCMC density control (gsplat)
  4. Export — checkpoint → standard PLY + rendered video
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)


def create_downscaled_images(images_dir: Path, output_dir: Path, factor: int) -> Path:
    """Downscale images by `factor` for memory-efficient training."""
    dst = output_dir / f"images_{factor}"
    if dst.exists() and any(dst.iterdir()):
        logger.info("images_%d already exists, skipping downscale.", factor)
        return dst

    dst.mkdir(parents=True, exist_ok=True)
    for f in tqdm(sorted(images_dir.glob("*.jpg")), desc=f"Downscaling {factor}x"):
        img = Image.open(f)
        img.resize((img.width // factor, img.height // factor), Image.LANCZOS).save(
            dst / f.name, quality=95,
        )
    logger.info("Created %s", dst)
    return dst


def render_video(
    ckpt_path: Path, data_dir: Path, output_path: Path, data_factor: int,
) -> None:
    """Render a fly-through video from the trained model."""
    import torch
    import torch.nn.functional as F
    import imageio

    from src.gaussian_trainer import ColmapDataset

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    splats = {k: v.to(device) for k, v in ckpt["splats"].items()}

    dataset = ColmapDataset.load(data_dir / "sparse" / "0")
    Ks = dataset.Ks.clone()
    if data_factor > 1:
        Ks[:, :2, :] /= data_factor
    W = dataset.widths[0] // data_factor
    H = dataset.heights[0] // data_factor

    from gsplat.rendering import rasterization

    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(str(output_path), fps=30)

    # Render from all training cameras
    for i in tqdm(range(len(dataset.image_names)), desc="Rendering video"):
        viewmat = dataset.w2cs[i:i+1].to(device)
        K = Ks[i:i+1].to(device)
        colors = torch.cat([splats["sh0"], splats["shN"]], dim=1)

        with torch.no_grad():
            renders, _, _ = rasterization(
                means=splats["means"],
                quats=F.normalize(splats["quats"], dim=-1),
                scales=torch.exp(splats["scales"]),
                opacities=torch.sigmoid(splats["opacities"]),
                colors=colors,
                viewmats=viewmat,
                Ks=K,
                width=W,
                height=H,
                sh_degree=3,
                packed=False,
                near_plane=0.01,
                far_plane=1e10,
                render_mode="RGB",
            )

        frame = (renders[0].clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
        writer.append_data(frame)

    writer.close()
    logger.info("Video saved: %s", output_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Video → 3D Gaussian Splatting reconstruction.",
    )
    parser.add_argument("--video", type=Path, required=True,
                        help="Input video file (MP4, MOV, ...).")
    parser.add_argument("--output", type=Path, required=True,
                        help="Output directory for all results.")
    parser.add_argument("--max-steps", type=int, default=30_000,
                        help="Training iterations (default: 30000).")
    parser.add_argument("--downscale", type=int, default=1,
                        help="Image downscale factor for training (default: 1).")
    args = parser.parse_args()

    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    # ── Stage 1: Extract frames ──────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STAGE 1: Frame extraction")
    logger.info("=" * 60)

    images_dir = output / "images"
    if images_dir.exists() and any(images_dir.glob("*.jpg")):
        n = len(list(images_dir.glob("*.jpg")))
        logger.info("Found %d existing frames, skipping extraction.", n)
    else:
        from src.frame_extractor import FrameExtractor
        extractor = FrameExtractor.from_video(args.video)
        stats = extractor.extract(args.video, images_dir)
        logger.info("Extracted %d frames.", stats.accepted)

    # ── Stage 2: Camera poses (COLMAP) ───────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STAGE 2: Camera pose estimation (COLMAP)")
    logger.info("=" * 60)

    sparse_dir = output / "sparse" / "0"
    if sparse_dir.exists() and (sparse_dir / "cameras.bin").exists():
        logger.info("COLMAP output exists, skipping SfM.")
    else:
        from src.pose_estimator import PoseEstimator
        estimator = PoseEstimator()
        n_registered = estimator.estimate(images_dir, output)
        logger.info("Registered %d cameras.", n_registered)

    # ── Stage 2.5: Downscale images if needed ────────────────────────────────
    if args.downscale > 1:
        create_downscaled_images(images_dir, output, args.downscale)

    # ── Stage 3: Train 3D Gaussians ──────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STAGE 3: Gaussian Splatting training (MCMC)")
    logger.info("=" * 60)

    from src.gaussian_trainer import GaussianTrainer, TrainConfig
    config = TrainConfig(max_steps=args.max_steps)
    trainer = GaussianTrainer(cfg=config)
    ckpt_path = trainer.train(output, output / "results", data_factor=args.downscale)

    # ── Stage 4: Export ──────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STAGE 4: Export PLY + video")
    logger.info("=" * 60)

    ply_path = output / "point_cloud.ply"
    from src.exporter import export_ply
    export_ply(ckpt_path, ply_path)

    video_path = output / "flythrough.mp4"
    render_video(ckpt_path, output, video_path, args.downscale)

    elapsed = time.time() - t0
    logger.info("=" * 60)
    logger.info("DONE in %.0f seconds", elapsed)
    logger.info("  PLY:   %s", ply_path)
    logger.info("  Video: %s", video_path)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
