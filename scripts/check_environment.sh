#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "${PROJECT_ROOT}"

status=0

required_tool() {
  local name="$1"
  if command -v "${name}" >/dev/null 2>&1; then
    printf '%-12s %s\n' "${name}" "$(command -v "${name}")"
  else
    printf '%-12s MISSING\n' "${name}"
    status=1
  fi
}

printf 'Workspace\n'
printf '%s\n\n' "${PROJECT_ROOT}"

printf 'Operating system\n'
grep '^PRETTY_NAME=' /etc/os-release | cut -d= -f2- | tr -d '"'
uname -srmo
printf '\n'

printf 'Required tool paths\n'
required_tool python3
required_tool gcc
required_tool g++
required_tool make
required_tool git
printf '\n'

printf 'Native toolchain\n'
python3 --version
gcc --version | sed -n '1p'
g++ --version | sed -n '1p'
make --version | sed -n '1p'
git --version
if command -v cmake >/dev/null 2>&1; then
  cmake --version | sed -n '1p'
else
  printf 'cmake: not installed (optional; current ABC build uses Make)\n'
fi
printf '\n'

printf 'Canonical Python environment\n'
if [[ -x .venv/bin/python ]]; then
  .venv/bin/python - <<'PY'
import sys
import joblib
import numpy
import scipy
import sklearn

print(f"executable:     {sys.executable}")
print(f"python:         {sys.version.split()[0]}")
print(f"numpy:          {numpy.__version__}")
print(f"scipy:          {scipy.__version__}")
print(f"scikit-learn:   {sklearn.__version__}")
print(f"joblib:         {joblib.__version__}")
PY
  .venv/bin/python -m pip check
else
  printf '.venv is missing; run make bootstrap\n'
  status=1
fi
printf '\n'

printf 'Berkeley ABC\n'
ABC_BINARY="${PROJECT_ROOT}/.tools/abc"
if [[ -x "${ABC_BINARY}" ]]; then
  temporary_dir="$(mktemp -d)"
  (
    cd "${temporary_dir}"
    "${ABC_BINARY}" -c version 2>&1 | tail -n 1
  )
  rm -rf -- "${temporary_dir}"
  if [[ -d .tools/abc-src/.git ]]; then
    printf 'commit: %s\n' "$(git -C .tools/abc-src rev-parse HEAD)"
  fi
else
  printf 'ABC is missing; run make abc\n'
  status=1
fi
printf '\n'

printf 'EPFL data\n'
if [[ -d data/epfl ]]; then
  epfl_count="$(find data/epfl/arithmetic data/epfl/random_control -maxdepth 1 -type f -name '*.aig' 2>/dev/null | wc -l)"
  printf 'small AIGER circuits: %s\n' "${epfl_count}"
  if [[ -d data/epfl/.git ]]; then
    printf 'commit: %s\n' "$(git -C data/epfl rev-parse HEAD)"
  fi
else
  printf 'EPFL data is missing; run make epfl\n'
  status=1
fi
printf '\n'

printf 'Disk usage\n'
du -sh .venv .tools data artifacts 2>/dev/null || true
df -h . | tail -n 1

exit "${status}"
