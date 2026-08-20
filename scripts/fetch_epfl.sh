#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DESTINATION="${1:-${PROJECT_ROOT}/data/epfl}"
EPFL_REVISION="${EPFL_REVISION:-82d8cc6910419298e713a46644ed59fd3df53038}"

if [[ ! -d "${DESTINATION}/.git" ]]; then
  git clone --filter=blob:none --no-checkout https://github.com/lsils/benchmarks.git "${DESTINATION}"
fi

git -C "${DESTINATION}" sparse-checkout init --no-cone
git -C "${DESTINATION}" sparse-checkout set '/arithmetic/*.aig' '/random_control/*.aig'
git -C "${DESTINATION}" fetch --depth 1 origin "${EPFL_REVISION}"
git -C "${DESTINATION}" checkout --detach "${EPFL_REVISION}"

COUNT="$(find "${DESTINATION}/arithmetic" "${DESTINATION}/random_control" -maxdepth 1 -type f -name '*.aig' | wc -l)"
printf 'EPFL AIGER subset ready: %s files in %s\n' "${COUNT}" "${DESTINATION}"
printf 'EPFL revision: %s\n' "$(git -C "${DESTINATION}" rev-parse HEAD)"
