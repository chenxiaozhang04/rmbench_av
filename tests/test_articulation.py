import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
from rmbench_sim.articulation import MountedArticulation


class _Entity:
    def __init__(self) -> None:
        self.pose = None
        self.qpos = None

    def set_root_pose(self, pose) -> None:
        self.pose = pose

    def get_active_joints(self):
        return [object()] * 6

    def set_qpos(self, qpos) -> None:
        self.qpos = np.asarray(qpos)

    def find_link_by_name(self, name):
        return "camera-link" if name == "link_cam" else None


class _Scene:
    def __init__(self) -> None:
        self.entity = _Entity()
        self.loader = SimpleNamespace(fix_root_link=False, load=lambda _path: self.entity)

    def create_urdf_loader(self):
        return self.loader


def test_mounted_articulation_loads_fixed_root(tmp_path: Path, monkeypatch) -> None:
    urdf = tmp_path / "head.urdf"
    urdf.write_text("<robot name='head'/>", encoding="utf-8")
    fake_sapien = ModuleType("sapien.core")
    fake_sapien.Pose = lambda p, q: (np.asarray(p), np.asarray(q))
    fake_sapien_package = ModuleType("sapien")
    fake_sapien_package.core = fake_sapien
    monkeypatch.setitem(sys.modules, "sapien", fake_sapien_package)
    monkeypatch.setitem(sys.modules, "sapien.core", fake_sapien)

    scene = _Scene()
    head = MountedArticulation(
        scene,
        urdf,
        position=(1, 2, 3),
        qpos=(0, 1, 2, 3, 4, 5),
    )

    assert scene.loader.fix_root_link is True
    assert np.allclose(head.entity.pose[0], [1, 2, 3])
    assert np.allclose(head.entity.qpos, [0, 1, 2, 3, 4, 5])
    assert head.find_link("link_cam") == "camera-link"


def test_mounted_articulation_validates_joint_count(tmp_path: Path, monkeypatch) -> None:
    urdf = tmp_path / "head.urdf"
    urdf.write_text("<robot name='head'/>", encoding="utf-8")
    fake_sapien = ModuleType("sapien.core")
    fake_sapien.Pose = lambda p, q: (p, q)
    fake_sapien_package = ModuleType("sapien")
    fake_sapien_package.core = fake_sapien
    monkeypatch.setitem(sys.modules, "sapien", fake_sapien_package)
    monkeypatch.setitem(sys.modules, "sapien.core", fake_sapien)
    head = MountedArticulation(_Scene(), urdf, position=(0, 0, 0))

    with pytest.raises(ValueError, match="expected 6"):
        head.set_qpos((0, 1))
