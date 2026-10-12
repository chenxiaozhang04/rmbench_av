"""Policy-independent lifecycle wrapper around RMBench tasks."""

from __future__ import annotations

import importlib
import os
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any, ClassVar, Self

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = ROOT / "env_cfg" / "task_config"


@contextmanager
def _repository_cwd() -> Iterator[None]:
    previous = Path.cwd()
    os.chdir(ROOT)
    try:
        yield
    finally:
        os.chdir(previous)


def list_tasks() -> tuple[str, ...]:
    ignored = {"__init__", "_base_task", "_GLOBAL_CONFIGS"}
    return tuple(
        sorted(path.stem for path in (ROOT / "envs").glob("*.py") if path.stem not in ignored)
    )


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as file:
        value = yaml.safe_load(file)
    if not isinstance(value, dict):
        raise TypeError(f"expected a mapping in {path}")
    return value


class RMBenchEnv:
    """One RMBench task with a Gymnasium-compatible lifecycle.

    Observations remain RMBench-native. Camera and robot name normalization
    belongs to the consuming system rather than this simulator package.
    """

    metadata: ClassVar[dict[str, list[str]]] = {"render_modes": []}

    def __init__(
        self,
        task: str,
        *,
        task_config: str = "demo_clean",
        action_type: str = "qpos",
    ) -> None:
        if task not in list_tasks():
            raise ValueError(f"unknown RMBench task {task!r}; available={list_tasks()}")
        if action_type not in {"qpos", "ee"}:
            raise ValueError("action_type must be 'qpos' or 'ee'")
        if not (CONFIG_ROOT / f"{task_config}.yml").is_file():
            raise ValueError(f"unknown task config {task_config!r}")

        self.task_name = task
        self.task_config_name = task_config
        self.action_type = action_type
        self._task: Any | None = None
        self._config: dict[str, Any] | None = None

    def _load_config(self) -> dict[str, Any]:
        config = _read_yaml(CONFIG_ROOT / f"{self.task_config_name}.yml")
        registry = _read_yaml(CONFIG_ROOT / "_embodiment_config.yml")
        embodiment = config.get("embodiment") or []
        if len(embodiment) not in {1, 3}:
            raise ValueError("embodiment must be [robot] or [left, right, distance]")

        def robot_path(name: str) -> Path:
            try:
                configured = Path(registry[name]["file_path"])
            except (KeyError, TypeError) as exc:
                raise ValueError(f"unknown embodiment {name!r}") from exc
            path = configured if configured.is_absolute() else ROOT / configured
            path = path.resolve()
            if not (path / "config.yml").is_file():
                raise FileNotFoundError(
                    f"missing embodiment asset {path}; run scripts/download_assets.sh"
                )
            return path

        left_path = robot_path(str(embodiment[0]))
        right_name = embodiment[0] if len(embodiment) == 1 else embodiment[1]
        right_path = robot_path(str(right_name))

        config.update(
            task_name=self.task_name,
            task_config=self.task_config_name,
            eval_mode=True,
            save_data=False,
            collect_data=False,
            eval_video_log=False,
            left_robot_file=str(left_path),
            right_robot_file=str(right_path),
            left_embodiment_config=_read_yaml(left_path / "config.yml"),
            right_embodiment_config=_read_yaml(right_path / "config.yml"),
            dual_arm_embodied=len(embodiment) == 1,
        )
        if len(embodiment) == 3:
            config["embodiment_dis"] = embodiment[2]
        return config

    def _new_task(self) -> Any:
        module = importlib.import_module(f"envs.{self.task_name}")
        return getattr(module, self.task_name)()

    def reset(
        self, *, seed: int | None = None, options: Mapping[str, Any] | None = None
    ) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
        del options
        self.close()
        with _repository_cwd():
            config = self._load_config()
            task = self._new_task()
            task.setup_demo(now_ep_num=0, seed=0 if seed is None else seed, is_test=True, **config)
            self._config = config
            self._task = task
            observation = task.get_obs()
        return observation, {"backend": "rmbench", "task": self.task_name, "seed": seed}

    def observe(self) -> Mapping[str, Any]:
        task = self._require_task()
        with _repository_cwd():
            return task.get_obs()

    def step(
        self, action: Sequence[float] | np.ndarray
    ) -> tuple[Mapping[str, Any], float, bool, bool, Mapping[str, Any]]:
        task = self._require_task()
        flat_action = np.asarray(action, dtype=np.float64).reshape(-1)
        expected = self.describe()["action_size"]
        if flat_action.size != expected:
            raise ValueError(f"expected action with {expected} values, got {flat_action.size}")

        with _repository_cwd():
            task.take_action(flat_action, action_type=self.action_type)
            observation = task.get_obs()
            terminated = bool(task.eval_success or task.check_success())
            truncated = bool(task.step_lim is not None and task.take_action_cnt >= task.step_lim)
            reward = float(getattr(task, "max_reward", 0.0))
        return observation, reward, terminated, truncated and not terminated, {
            "backend": "rmbench",
            "task": self.task_name,
            "step_count": task.take_action_cnt,
        }

    def describe(self) -> Mapping[str, Any]:
        config = self._config or self._load_config()
        dual_arm = bool(config.get("dual_arm", True))
        if self.action_type == "qpos":
            size = len(config["left_embodiment_config"]["arm_joints_name"][0]) + 1
            if dual_arm:
                size += len(config["right_embodiment_config"]["arm_joints_name"][1]) + 1
        else:
            size = 8 * (2 if dual_arm else 1)
        camera_config = config.get("camera", {})
        camera_names: list[str] = []
        if camera_config.get("collect_head_camera", True):
            camera_names.append("head_camera")
        if camera_config.get("collect_wrist_camera", True):
            camera_names.append("left_camera")
            if dual_arm:
                camera_names.append("right_camera")
        return {
            "backend": "rmbench",
            "task": self.task_name,
            "embodiment": tuple(config["embodiment"]),
            "action_type": self.action_type,
            "action_size": size,
            "camera_names": tuple(camera_names),
        }

    def close(self) -> None:
        if self._task is None:
            return
        task, self._task = self._task, None
        with _repository_cwd():
            task.close_env()
            viewer = getattr(task, "viewer", None)
            if viewer is not None:
                viewer.close()

    def _require_task(self) -> Any:
        if self._task is None:
            raise RuntimeError("call reset() before observe() or step()")
        return self._task

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def make(task: str, **kwargs: Any) -> RMBenchEnv:
    return RMBenchEnv(task, **kwargs)
