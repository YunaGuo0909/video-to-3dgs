"""
Camera pose estimation via COLMAP (pycolmap).

Runs SIFT feature extraction, sequential matching, and incremental SfM.
Outputs COLMAP binary format (sparse/0/) and a sparse point cloud PLY
for Gaussian initialisation.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


class PoseEstimator:
    """Run COLMAP SfM and write results in COLMAP binary format."""

    def estimate(self, images_dir: Path, output_dir: Path) -> int:
        """Run SfM on images_dir, write binary model to output_dir/sparse/0/.

        Returns the number of registered images.
        """
        try:
            import pycolmap
        except ImportError:
            raise ImportError("pycolmap required: pip install pycolmap")

        colmap_dir = output_dir / "colmap"
        colmap_dir.mkdir(parents=True, exist_ok=True)
        db = colmap_dir / "database.db"

        logger.info("Extracting SIFT features...")
        pycolmap.extract_features(database_path=str(db), image_path=str(images_dir))

        logger.info("Sequential matching (overlap=20)...")
        try:
            opts = pycolmap.SequentialMatchingOptions()
            opts.overlap = 20
            opts.loop_detection = False
            pycolmap.match_sequential(database_path=str(db), options=opts)
        except (AttributeError, TypeError):
            pycolmap.match_sequential(database_path=str(db))

        logger.info("Incremental mapping...")
        sparse_tmp = colmap_dir / "sparse"
        sparse_tmp.mkdir(exist_ok=True)
        maps = pycolmap.incremental_mapping(
            database_path=str(db),
            image_path=str(images_dir),
            output_path=str(sparse_tmp),
        )
        if not maps:
            raise RuntimeError("COLMAP produced no reconstructions.")

        best_id = max(maps, key=lambda k: len(maps[k].images))
        best = maps[best_id]
        n_images = len(best.images)
        n_points = len(best.points3D)
        logger.info("Best model: %d images, %d points", n_images, n_points)

        # Write binary format for gsplat
        out_sparse = output_dir / "sparse" / "0"
        out_sparse.mkdir(parents=True, exist_ok=True)
        best.write_binary(str(out_sparse))
        logger.info("COLMAP binary written to %s", out_sparse)

        # Export sparse PLY for visualisation / dense init
        self._export_ply(best, output_dir / "sparse_pt_cloud.ply")

        return n_images

    @staticmethod
    def _export_ply(reconstruction, ply_path: Path) -> None:
        """Write sparse 3D points to a PLY file."""
        try:
            from plyfile import PlyData, PlyElement
        except ImportError:
            logger.warning("plyfile not installed, skipping PLY export.")
            return

        pts = reconstruction.points3D
        if not pts:
            return

        xyz = np.array([p.xyz for p in pts.values()], dtype=np.float32)
        rgb = np.array([p.color[:3] for p in pts.values()], dtype=np.uint8)

        verts = np.empty(
            len(xyz),
            dtype=[("x", "f4"), ("y", "f4"), ("z", "f4"),
                   ("red", "u1"), ("green", "u1"), ("blue", "u1")],
        )
        verts["x"], verts["y"], verts["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
        verts["red"], verts["green"], verts["blue"] = rgb[:, 0], rgb[:, 1], rgb[:, 2]

        PlyData([PlyElement.describe(verts, "vertex")]).write(str(ply_path))
        logger.info("Sparse PLY: %d points → %s", len(xyz), ply_path)
