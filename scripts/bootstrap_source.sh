#!/usr/bin/env bash
# Fetch upstream assets/dependencies; never overwrite the experiment's Python source.
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
DEST="$ROOT/vendor/RL-100"
if [[ -e "$DEST" ]]; then echo "vendor/RL-100 already exists; inspect it before retrying" >&2; exit 1; fi
mkdir -p "$ROOT/vendor"
git clone --no-checkout https://github.com/Lei-Kun/RL-100.git "$DEST"
git -C "$DEST" checkout --detach 64264d952c1fda9d5096c090ddaa7177757a77ad
"${PYTHON:-python}" - "$ROOT" <<'PY'
from pathlib import Path
import shutil,sys
r=Path(sys.argv[1]);src=r/'vendor/RL-100/RL-100'
for p in src.rglob('*'):
 if p.is_file() and p.suffix not in ('.py','.pyc') and '.git' not in p.parts:
  dst=r/'rl100'/p.relative_to(src)
  if not dst.exists():dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dst)
print('Upstream assets copied. Follow docs/INSTALL.md for dependencies.')
PY
