"""Helpers for attaching auxiliary URDF articulations to an RMBench scene."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np


class MountedArticulation:
    """A fixed-root articulation whose base pose can be calibrated at runtime."""

    def __init__(
        self,
        scene: Any,
        urdf_path: str | Path,
        *,
        position: Sequence[float],
        quaternion: Sequence[float] = (1.0, 0.0, 0.0, 0.0),
        qpos: Sequence[float] | None = None,
    ) -> None:
        path = Path(urdf_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"auxiliary articulation URDF not found: {path}")

        loader = scene.create_urdf_loader()
        loader.fix_root_link = True
        entity = loader.load(str(path))
        if entity is None:
            raise RuntimeError(f"failed to load auxiliary articulation: {path}")

        self.entity = entity
        self.urdf_path = path
        self.set_base_pose(position, quaternion)
        if qpos is not None:
            self.set_qpos(qpos)

    def set_base_pose(
        self, position: Sequence[float], quaternion: Sequence[float]
    ) -> None:
        import sapien.core as sapien

        position_array = np.asarray(position, dtype=np.float64)
        quaternion_array = np.asarray(quaternion, dtype=np.float64)
        if position_array.shape != (3,):
            raise ValueError("position must contain three values")
        if quaternion_array.shape != (4,):
            raise ValueError("quaternion must contain four values in wxyz order")
        self.entity.set_root_pose(sapien.Pose(position_array, quaternion_array))

    def set_qpos(self, qpos: Sequence[float]) -> None:
        values = np.asarray(qpos, dtype=np.float64)
        expected = len(self.entity.get_active_joints())
        if values.shape != (expected,):
            raise ValueError(f"expected {expected} active joint values, got {values.size}")
        self.entity.set_qpos(values)

    def find_link(self, name: str) -> Any:
        link = self.entity.find_link_by_name(name)
        if link is None:
            raise ValueError(f"articulation has no link named {name!r}")
        return link
