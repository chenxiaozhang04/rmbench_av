# RMBench AV Simulator

Simulator-only fork of [RMBench](https://github.com/RoboTwin-Platform/RMBench),
pinned from upstream commit `7d1cdfc`. This repository contains SAPIEN tasks,
robot/object assets and a small environment API. It intentionally excludes
policy implementations, XPolicyLab, training code and evaluation servers.

## Install

Use a dedicated Python 3.10 environment:

```bash
pip install -e .
bash scripts/download_assets.sh
```

## API

```python
import rmbench_sim

env = rmbench_sim.make("cover_blocks", action_type="qpos")
observation, info = env.reset(seed=0)
description = env.describe()

observation, reward, terminated, truncated, info = env.step(action)
env.close()
```

The observation is RMBench-native. Camera/robot name normalization and policy
action codecs belong to the consuming system, not this simulator repository.

Available task names are returned by `rmbench_sim.list_tasks()`.

## Scope

Included:

- SAPIEN task and scene implementation
- task and embodiment configuration
- robot/object asset downloader
- `reset`, `observe`, `step`, `describe`, and `close`

Excluded:

- policies and checkpoints
- XPolicyLab
- model servers and evaluation schedulers
- training and benchmark data downloads

The original MIT license and copyright notice are retained in `LICENSE`.
