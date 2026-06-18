"""
Export a trained gsplat checkpoint to standard 3DGS PLY format.

Writes binary little-endian PLY directly (no plyfile dependency)
to guarantee exact compatibility with SuperSplat, antimatter15, etc.
"""

from __future__ import annotations

import logging
import struct
from pathlib import Path

import numpy as np
import torch

logger = logging.getLogger(__name__)


def export_ply(ckpt_path: Path, output_path: Path) -> None:
    """Convert a gsplat .pt checkpoint to a standard 3DGS .ply file."""
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    splats = ckpt["splats"]

    means = splats["means"].numpy()                        # (N, 3)
    scales = splats["scales"].numpy()                      # (N, 3) log-space
    quats = splats["quats"].numpy()                        # (N, 4) unnormalized
    opacities = splats["opacities"].numpy().reshape(-1)    # (N,) logit-space
    sh0 = splats["sh0"].numpy()                            # (N, 1, 3)
    shN = splats.get("shN")
    if shN is not None:
        shN = shN.numpy()                                  # (N, K, 3)

    # Normalize quaternions
    quat_norms = np.linalg.norm(quats, axis=1, keepdims=True)
    quats = quats / np.clip(quat_norms, 1e-8, None)

    N = len(means)
    n_sh_rest = shN.shape[1] * 3 if shN is not None else 0

    # Build PLY header
    props = []
    props += ["property float x", "property float y", "property float z"]
    props += ["property float nx", "property float ny", "property float nz"]
    props += ["property float f_dc_0", "property float f_dc_1", "property float f_dc_2"]
    for i in range(n_sh_rest):
        props.append(f"property float f_rest_{i}")
    props.append("property float opacity")
    props += ["property float scale_0", "property float scale_1", "property float scale_2"]
    props += ["property float rot_0", "property float rot_1", "property float rot_2", "property float rot_3"]

    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {N}\n"
        + "\n".join(props) + "\n"
        "end_header\n"
    )

    # Prepare SH data
    dc = sh0.reshape(N, 3)  # (N, 3)
    if shN is not None:
        K = shN.shape[1]
        # Channel-first order: [R0..RK, G0..GK, B0..BK]
        sh_rest = shN.reshape(N, K, 3).transpose(0, 2, 1).reshape(N, -1)  # (N, 3K)
    else:
        sh_rest = np.zeros((N, 0), dtype=np.float32)

    # Number of floats per vertex
    n_floats = 3 + 3 + 3 + n_sh_rest + 1 + 3 + 4  # pos + normal + dc + rest + opa + scale + rot

    # Pack all data into a contiguous float32 array
    data = np.zeros((N, n_floats), dtype=np.float32)
    col = 0
    data[:, col:col+3] = means;                    col += 3
    data[:, col:col+3] = 0.0;                      col += 3  # normals
    data[:, col:col+3] = dc;                        col += 3
    if n_sh_rest > 0:
        data[:, col:col+n_sh_rest] = sh_rest;       col += n_sh_rest
    data[:, col] = opacities;                        col += 1
    data[:, col:col+3] = scales;                     col += 3
    data[:, col:col+4] = quats;                      col += 4

    # Write binary little-endian PLY
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(str(output_path), "wb") as f:
        f.write(header.encode("ascii"))
        # Ensure little-endian
        if data.dtype.byteorder not in ("<", "=") or (data.dtype.byteorder == "=" and struct.pack("H", 1)[0] != 1):
            data = data.astype(data.dtype.newbyteorder("<"))
        f.write(data.tobytes())

    size_mb = output_path.stat().st_size / 1e6
    logger.info("Exported %d Gaussians → %s (%.1f MB)", N, output_path, size_mb)
