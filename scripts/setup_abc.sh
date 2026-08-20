#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TOOLS_DIR="${PROJECT_ROOT}/.tools"
ABC_DIR="${TOOLS_DIR}/abc-src"
ABC_REVISION="${ABC_REVISION:-5ea34643247f9f45baa3bba40738832d7db19c60}"
mkdir -p "${TOOLS_DIR}"

if [[ ! -d "${ABC_DIR}/.git" ]]; then
  git clone --filter=blob:none --no-checkout https://github.com/berkeley-abc/abc.git "${ABC_DIR}"
fi
git -C "${ABC_DIR}" fetch --depth 1 origin "${ABC_REVISION}"
git -C "${ABC_DIR}" checkout --detach "${ABC_REVISION}"

make -C "${ABC_DIR}" -j"$(nproc)" ABC_USE_NO_READLINE=1
ln -sfn "${ABC_DIR}/abc" "${TOOLS_DIR}/abc"
printf 'ABC ready: %s\n' "${TOOLS_DIR}/abc"
printf 'ABC revision: %s\n' "$(git -C "${ABC_DIR}" rev-parse HEAD)"
