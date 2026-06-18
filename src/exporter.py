"""
Export a trained gsplat checkpoint to standard 3DGS PLY format.

The PLY contains all Gaussian parameters (position, scale, rotation,
opacity, spherical harmonics) and can be viewed in SuperSplat,
antimatter15 viewer, or any 3DGS-compatible tool.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch
from plyfile import PlyData, PlyElement

logger = logging.getLogger(__name__)


def export_ply(ckpt_path: Path, output_path: Path) -> None:
    """Convert a gsplat .pt checkpoint to a standard 3DGS .ply file."""
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    splats = ckpt["splats"]

    means = splats["means"].numpy()
    scales = splats["scales"].numpy()
    quats = splats["quats"].numpy()
    opacities = splats["opacities"].numpy().reshape(-1)
    sh0 = splats["sh0"].numpy()
    shN = splats.get("shN")
    if shN is not None:
        shN = shN.numpy()

    N = len(means)

    # Build structured array with standard 3DGS PLY fields
    attrs: list[tuple[str, str]] = [
        ("x", "f4"), ("y", "f4"), ("z", "f4"),
        ("nx", "f4"), ("ny", "f4"), ("nz", "f4"),
        ("f_dc_0", "f4"), ("f_dc_1", "f4"), ("f_dc_2", "f4"),
    ]

    n_rest = shN.shape[1] * 3 if shN is not None else 0
    for i in range(n_rest):
        attrs.append((f"f_rest_{i}", "f4"))

    attrs.append(("opacity", "f4"))
    attrs += [("scale_0", "f4"), ("scale_1", "f4"), ("scale_2", "f4")]
    attrs += [("rot_0", "f4"), ("rot_1", "f4"), ("rot_2", "f4"), ("rot_3", "f4")]

    v = np.empty(N, dtype=attrs)

    v["x"], v["y"], v["z"] = means[:, 0], means[:, 1], means[:, 2]
    v["nx"] = v["ny"] = v["nz"] = 0.0

    dc = sh0.reshape(N, 3)
    v["f_dc_0"], v["f_dc_1"], v["f_dc_2"] = dc[:, 0], dc[:, 1], dc[:, 2]

    if shN is not None:
        rest = shN.reshape(N, -1, 3)
        for i in range(rest.shape[1]):
            v[f"f_rest_{i * 3}"] = rest[:, i, 0]
            v[f"f_rest_{i * 3 + 1}"] = rest[:, i, 1]
            v[f"f_rest_{i * 3 + 2}"] = rest[:, i, 2]

    v["opacity"] = opacities
    v["scale_0"], v["scale_1"], v["scale_2"] = scales[:, 0], scales[:, 1], scales[:, 2]
    v["rot_0"], v["rot_1"], v["rot_2"], v["rot_3"] = (
        quats[:, 0], quats[:, 1], quats[:, 2], quats[:, 3],
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    PlyData([PlyElement.describe(v, "vertex")]).write(str(output_path))
    size_mb = output_path.stat().st_size / 1e6
    logger.info("Exported %d Gaussians → %s (%.1f MB)", N, output_path, size_mb)
