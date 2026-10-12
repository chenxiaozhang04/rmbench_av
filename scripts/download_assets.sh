#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT/assets"
python _download.py
cd "$ROOT"
python scripts/update_embodiment_config_path.py
