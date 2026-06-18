"""
3D Gaussian Splatting training with gsplat MCMC strategy.

Trains a set of 3D Gaussians from COLMAP-format data using gsplat's
rasteriser and MCMC-based adaptive density control. MCMC eliminates
manual densification tuning — Gaussians are added, relocated, and
removed via a stochastic process that converges to the optimal count.

Reference
---------
- Kerbl et al., "3D Gaussian Splatting for Real-Time Radiance Field
  Rendering", SIGGRAPH 2023.
- Kheradmand et al., "3D Gaussian Splatting as Markov Chain Monte Carlo",
  NeurIPS 2024.
"""

from __future__ import annotations

import logging
import math
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from tqdm import tqdm

logger = logging.getLogger(__name__)


# ── COLMAP binary reader ─────────────────────────────────────────────────────

def _read_cameras_binary(path: Path) -> dict:
    """Read cameras.bin → {cam_id: (model_id, width, height, params)}."""
    cameras = {}
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        for _ in range(n):
            cid = struct.unpack("<I", f.read(4))[0]
            model = struct.unpack("<i", f.read(4))[0]
            w = struct.unpack("<Q", f.read(8))[0]
            h = struct.unpack("<Q", f.read(8))[0]
            # Number of params per model: {0:3, 1:4, 2:4, 3:5, 4:8, ...}
            n_params = {0: 3, 1: 4, 2: 4, 3: 5, 4: 8, 5: 12, 6: 5}.get(model, 4)
            params = struct.unpack(f"<{n_params}d", f.read(8 * n_params))
            cameras[cid] = (model, w, h, params)
    return cameras


def _read_images_binary(path: Path) -> list:
    """Read images.bin → [(name, qvec, tvec, camera_id), ...]."""
    images = []
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        for _ in range(n):
            _img_id = struct.unpack("<I", f.read(4))[0]
            qvec = struct.unpack("<4d", f.read(32))  # qw, qx, qy, qz
            tvec = struct.unpack("<3d", f.read(24))
            cam_id = struct.unpack("<I", f.read(4))[0]
            name = b""
            while True:
                c = f.read(1)
                if c == b"\x00":
                    break
                name += c
            n_pts2d = struct.unpack("<Q", f.read(8))[0]
            f.read(n_pts2d * 24)  # skip 2D observations
            images.append((name.decode(), qvec, tvec, cam_id))
    return images


def _read_points3d_binary(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Read points3D.bin → (xyz [N,3], rgb [N,3])."""
    xyz_list, rgb_list = [], []
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        for _ in range(n):
            _pid = struct.unpack("<Q", f.read(8))[0]
            x, y, z = struct.unpack("<3d", f.read(24))
            r, g, b = struct.unpack("<3B", f.read(3))
            _err = struct.unpack("<d", f.read(8))[0]
            track_len = struct.unpack("<Q", f.read(8))[0]
            f.read(track_len * 8)  # skip track
            xyz_list.append([x, y, z])
            rgb_list.append([r, g, b])
    if not xyz_list:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.uint8)
    return np.array(xyz_list, dtype=np.float32), np.array(rgb_list, dtype=np.uint8)


def _qvec_to_rotmat(qvec: tuple) -> np.ndarray:
    """Quaternion (w,x,y,z) → 3x3 rotation matrix."""
    w, x, y, z = qvec
    return np.array([
        [1 - 2*y*y - 2*z*z, 2*x*y - 2*w*z,     2*x*z + 2*w*y],
        [2*x*y + 2*w*z,     1 - 2*x*x - 2*z*z, 2*y*z - 2*w*x],
        [2*x*z - 2*w*y,     2*y*z + 2*w*x,     1 - 2*x*x - 2*y*y],
    ], dtype=np.float64)


def _camera_params_to_K(model_id: int, params: tuple, w: int, h: int) -> np.ndarray:
    """COLMAP camera model → 3x3 intrinsic matrix K."""
    if model_id in (0, 2):  # SIMPLE_PINHOLE, SIMPLE_RADIAL
        f, cx, cy = params[0], params[1], params[2]
        return np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]], dtype=np.float64)
    elif model_id in (1, 4):  # PINHOLE, OPENCV
        fx, fy, cx, cy = params[0], params[1], params[2], params[3]
        return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
    elif model_id == 3:  # RADIAL
        f, cx, cy = params[0], params[1], params[2]
        return np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]], dtype=np.float64)
    else:
        f = params[0]
        return np.array([[f, 0, w/2], [0, f, h/2], [0, 0, 1]], dtype=np.float64)


# ── Dataset ──────────────────────────────────────────────────────────────────

@dataclass
class ColmapDataset:
    """In-memory dataset loaded from COLMAP binary format."""
    image_names: List[str]         # filenames
    Ks: Tensor                     # (N, 3, 3) intrinsics
    w2cs: Tensor                   # (N, 4, 4) world-to-camera
    widths: List[int]
    heights: List[int]
    points_xyz: Tensor             # (M, 3) initial 3D points
    points_rgb: Tensor             # (M, 3) colours [0,1]

    @staticmethod
    def load(sparse_dir: Path) -> "ColmapDataset":
        cameras = _read_cameras_binary(sparse_dir / "cameras.bin")
        images = _read_images_binary(sparse_dir / "images.bin")
        pts_xyz, pts_rgb = _read_points3d_binary(sparse_dir / "points3D.bin")

        # Sort images by name for deterministic order
        images.sort(key=lambda x: x[0])

        names, Ks, w2cs, ws, hs = [], [], [], [], []
        for name, qvec, tvec, cam_id in images:
            model_id, w, h, params = cameras[cam_id]
            K = _camera_params_to_K(model_id, params, w, h)

            R = _qvec_to_rotmat(qvec)
            t = np.array(tvec, dtype=np.float64)
            w2c = np.eye(4, dtype=np.float64)
            w2c[:3, :3] = R
            w2c[:3, 3] = t

            names.append(name)
            Ks.append(K)
            w2cs.append(w2c)
            ws.append(w)
            hs.append(h)

        return ColmapDataset(
            image_names=names,
            Ks=torch.tensor(np.array(Ks), dtype=torch.float32),
            w2cs=torch.tensor(np.array(w2cs), dtype=torch.float32),
            widths=ws,
            heights=hs,
            points_xyz=torch.tensor(pts_xyz, dtype=torch.float32),
            points_rgb=torch.tensor(pts_rgb / 255.0, dtype=torch.float32),
        )


# ── Utilities ────────────────────────────────────────────────────────────────

def _rgb_to_sh0(rgb: Tensor) -> Tensor:
    """Convert linear RGB [0,1] to zeroth-order SH coefficient."""
    C0 = 0.28209479177387814
    return (rgb - 0.5) / C0


def _knn_scale(points: Tensor, k: int = 4) -> Tensor:
    """Estimate initial Gaussian scale from k-nearest-neighbour distances."""
    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=k + 1, algorithm="auto")
    nn.fit(points.numpy())
    dists, _ = nn.kneighbors(points.numpy())
    mean_dist = torch.tensor(dists[:, 1:].mean(axis=1), dtype=torch.float32)
    return torch.log(mean_dist.clamp(min=1e-7)).unsqueeze(-1).expand(-1, 3)


# ── Trainer ──────────────────────────────────────────────────────────────────

@dataclass
class TrainConfig:
    max_steps: int = 30_000
    sh_degree: int = 3
    sh_degree_interval: int = 1000
    init_opa: float = 0.5
    lr_means: float = 1.6e-4
    lr_scales: float = 5e-3
    lr_quats: float = 1e-3
    lr_opacities: float = 5e-2
    lr_sh0: float = 2.5e-3
    lr_shN: float = 1.25e-4
    ssim_weight: float = 0.2
    cap_max: int = 1_000_000       # MCMC max Gaussian count
    save_every: int = 7000


class GaussianTrainer:
    """Train 3D Gaussians from COLMAP data using gsplat MCMC."""

    def __init__(self, cfg: TrainConfig | None = None) -> None:
        self.cfg = cfg or TrainConfig()

    def train(
        self,
        data_dir: Path,
        result_dir: Path,
        data_factor: int = 1,
    ) -> Path:
        """Run training. Returns path to final checkpoint."""
        from gsplat.rendering import rasterization
        from gsplat.strategy import MCMCStrategy

        cfg = self.cfg
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load dataset
        sparse_dir = data_dir / "sparse" / "0"
        dataset = ColmapDataset.load(sparse_dir)
        logger.info(
            "Loaded %d cameras, %d initial points",
            len(dataset.image_names), len(dataset.points_xyz),
        )

        # Determine image directory (images_2, images_4, etc.)
        if data_factor > 1:
            img_dir = data_dir / f"images_{data_factor}"
            if not img_dir.exists():
                img_dir = data_dir / "images"
                logger.warning("images_%d not found, using images/", data_factor)
        else:
            img_dir = data_dir / "images"

        # Preload and downscale images
        images = self._load_images(dataset, img_dir, data_factor, device)
        H, W = images[0].shape[:2]
        logger.info("Training resolution: %dx%d (%d images)", W, H, len(images))

        # Scale intrinsics for downscale factor
        Ks = dataset.Ks.clone()
        if data_factor > 1:
            Ks[:, :2, :] /= data_factor

        # Scene scale (for MCMC noise injection)
        cam_centers = -torch.bmm(
            dataset.w2cs[:, :3, :3].transpose(1, 2),
            dataset.w2cs[:, :3, 3:],
        ).squeeze(-1)
        scene_scale = (cam_centers.max(dim=0).values - cam_centers.min(dim=0).values).norm().item() * 1.1

        # Initialise Gaussians
        N = len(dataset.points_xyz)
        means = dataset.points_xyz.to(device).requires_grad_(True)
        scales = _knn_scale(dataset.points_xyz).to(device).requires_grad_(True)
        quats = F.normalize(torch.randn(N, 4, device=device), dim=-1).requires_grad_(True)
        opacities = torch.logit(torch.full((N,), cfg.init_opa, device=device)).requires_grad_(True)

        sh0 = _rgb_to_sh0(dataset.points_rgb).unsqueeze(1).to(device).requires_grad_(True)
        sh_rest_dim = (cfg.sh_degree + 1) ** 2 - 1
        shN = torch.zeros(N, sh_rest_dim, 3, device=device).requires_grad_(True)

        splats = torch.nn.ParameterDict({
            "means": torch.nn.Parameter(means),
            "scales": torch.nn.Parameter(scales),
            "quats": torch.nn.Parameter(quats),
            "opacities": torch.nn.Parameter(opacities),
            "sh0": torch.nn.Parameter(sh0),
            "shN": torch.nn.Parameter(shN),
        })

        # Optimisers
        try:
            from gsplat.optimizers import SelectiveAdam
            OptimizerClass = SelectiveAdam
        except ImportError:
            OptimizerClass = torch.optim.Adam

        optimizers = {
            "means": OptimizerClass([splats["means"]], lr=cfg.lr_means, eps=1e-15),
            "scales": OptimizerClass([splats["scales"]], lr=cfg.lr_scales, eps=1e-15),
            "quats": OptimizerClass([splats["quats"]], lr=cfg.lr_quats, eps=1e-15),
            "opacities": OptimizerClass([splats["opacities"]], lr=cfg.lr_opacities, eps=1e-15),
            "sh0": OptimizerClass([splats["sh0"]], lr=cfg.lr_sh0, eps=1e-15),
            "shN": OptimizerClass([splats["shN"]], lr=cfg.lr_shN, eps=1e-15),
        }

        # MCMC strategy
        strategy = MCMCStrategy(cap_max=cfg.cap_max)
        strategy_state = strategy.initialize_state(scene_scale=scene_scale)

        # Checkpoint dir
        ckpt_dir = result_dir / "ckpts"
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        # Training loop
        n_images = len(images)
        pbar = tqdm(range(cfg.max_steps), desc="Training")

        for step in pbar:
            # Random camera
            idx = step % n_images
            gt = images[idx].to(device)
            viewmat = dataset.w2cs[idx:idx+1].to(device)
            K = Ks[idx:idx+1].to(device)

            # Current SH degree (progressive activation)
            cur_sh = min(step // cfg.sh_degree_interval, cfg.sh_degree)

            # Rasterise
            colors = torch.cat([splats["sh0"], splats["shN"]], dim=1)
            renders, alphas, info = rasterization(
                means=splats["means"],
                quats=F.normalize(splats["quats"], dim=-1),
                scales=torch.exp(splats["scales"]),
                opacities=torch.sigmoid(splats["opacities"]),
                colors=colors,
                viewmats=viewmat,
                Ks=K,
                width=W,
                height=H,
                sh_degree=cur_sh,
                packed=False,
                near_plane=0.01,
                far_plane=1e10,
                render_mode="RGB",
                absgrad=True,
            )

            rendered = renders[0, ..., :3]  # (H, W, 3)

            # Loss: L1 + SSIM
            l1 = F.l1_loss(rendered, gt)
            ssim_val = self._ssim(rendered, gt)
            loss = (1 - cfg.ssim_weight) * l1 + cfg.ssim_weight * (1 - ssim_val)

            # MCMC pre-backward
            strategy.step_pre_backward(
                params=splats, optimizers=optimizers,
                state=strategy_state, step=step, info=info,
            )

            loss.backward()

            # MCMC post-backward
            strategy.step_post_backward(
                params=splats, optimizers=optimizers,
                state=strategy_state, step=step, info=info,
            )

            # Optimiser step
            for opt in optimizers.values():
                opt.step()
                opt.zero_grad(set_to_none=True)

            pbar.set_postfix(loss=f"{loss.item():.4f}", n=len(splats["means"]))

            # Save checkpoint
            if (step + 1) % cfg.save_every == 0 or step == cfg.max_steps - 1:
                path = ckpt_dir / f"ckpt_{step}.pt"
                torch.save({
                    "step": step,
                    "splats": {k: v.data for k, v in splats.items()},
                }, str(path))
                logger.info("Checkpoint saved: %s", path)

        final = ckpt_dir / f"ckpt_{cfg.max_steps - 1}.pt"
        logger.info("Training complete. Final checkpoint: %s", final)
        return final

    def _load_images(
        self, dataset: ColmapDataset, img_dir: Path, factor: int, device: torch.device,
    ) -> List[Tensor]:
        """Load and optionally downscale training images."""
        from PIL import Image

        images = []
        for name in tqdm(dataset.image_names, desc="Loading images"):
            path = img_dir / name
            if not path.exists():
                raise FileNotFoundError(f"Image not found: {path}")
            img = np.array(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
            # If images aren't pre-downscaled, do it now
            orig_h, orig_w = img.shape[:2]
            target_h = dataset.heights[0] // factor
            target_w = dataset.widths[0] // factor
            if orig_h != target_h or orig_w != target_w:
                from PIL import Image as PILImage
                pil = PILImage.fromarray((img * 255).astype(np.uint8))
                pil = pil.resize((target_w, target_h), PILImage.LANCZOS)
                img = np.array(pil, dtype=np.float32) / 255.0
            images.append(torch.tensor(img, dtype=torch.float32))
        return images

    @staticmethod
    def _ssim(img1: Tensor, img2: Tensor) -> Tensor:
        """Compute SSIM between two HWC images."""
        # Transpose to NCHW for F.conv2d
        x = img1.permute(2, 0, 1).unsqueeze(0)
        y = img2.permute(2, 0, 1).unsqueeze(0)
        C1, C2 = 0.01 ** 2, 0.03 ** 2
        kernel_size = 11
        sigma = 1.5
        # Gaussian kernel
        coords = torch.arange(kernel_size, dtype=x.dtype, device=x.device) - kernel_size // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g = g / g.sum()
        kernel = g.unsqueeze(0) * g.unsqueeze(1)
        kernel = kernel.expand(3, 1, -1, -1)
        pad = kernel_size // 2
        mu1 = F.conv2d(x, kernel, padding=pad, groups=3)
        mu2 = F.conv2d(y, kernel, padding=pad, groups=3)
        mu1_sq, mu2_sq, mu12 = mu1 * mu1, mu2 * mu2, mu1 * mu2
        s1_sq = F.conv2d(x * x, kernel, padding=pad, groups=3) - mu1_sq
        s2_sq = F.conv2d(y * y, kernel, padding=pad, groups=3) - mu2_sq
        s12 = F.conv2d(x * y, kernel, padding=pad, groups=3) - mu12
        ssim_map = ((2 * mu12 + C1) * (2 * s12 + C2)) / ((mu1_sq + mu2_sq + C1) * (s1_sq + s2_sq + C2))
        return ssim_map.mean()
