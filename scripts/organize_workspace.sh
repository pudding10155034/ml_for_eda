#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "${PROJECT_ROOT}"

assert_workspace_target() {
  local target="$1"
  case "${target}" in
    "${PROJECT_ROOT}"/*) ;;
    *)
      printf 'Refusing target outside workspace: %s\n' "${target}" >&2
      exit 1
      ;;
  esac
}

archive_obsolete_python_envs() {
  local candidates=(.venv-linux .uv-python .uv-cache)
  local existing=()
  local name
  for name in "${candidates[@]}"; do
    if [[ -e "${PROJECT_ROOT}/${name}" ]]; then
      existing+=("${name}")
    fi
  done
  if [[ ${#existing[@]} -eq 0 ]]; then
    return
  fi

  mkdir -p artifacts/archive
  tar -czf artifacts/archive/obsolete-python-envs-20260821.tar.gz "${existing[@]}"
  for name in "${existing[@]}"; do
    local target="${PROJECT_ROOT}/${name}"
    assert_workspace_target "${target}"
    rm -rf -- "${target}"
  done
}

move_if_present() {
  local source="$1"
  local destination="$2"
  if [[ -e "${source}" ]]; then
    mkdir -p "$(dirname "${destination}")"
    mv -- "${source}" "${destination}"
  fi
}

archive_obsolete_python_envs

move_if_present artifacts/abc_smoke.csv artifacts/smoke/abc_runner/trajectories.csv
move_if_present artifacts/abc_smoke_recipes.json artifacts/smoke/abc_runner/recipes.json
move_if_present artifacts/real_small.csv artifacts/smoke/epfl_lodo/trajectories.csv
move_if_present artifacts/real_small_live_search.json artifacts/smoke/epfl_lodo/live_search.json
move_if_present artifacts/real_small_model.joblib artifacts/smoke/epfl_lodo/model.joblib
move_if_present artifacts/real_small_recipes.json artifacts/smoke/epfl_lodo/recipes.json
move_if_present artifacts/real_small_simulation.json artifacts/smoke/epfl_lodo/simulation.json
move_if_present artifacts/real_small_training.json artifacts/smoke/epfl_lodo/training_report.json
move_if_present artifacts/demo_cli_output.json artifacts/demo/cli_output.json

rm -f -- "${PROJECT_ROOT}/.coverage" "${PROJECT_ROOT}/abc.history"
for target in "${PROJECT_ROOT}/.pytest_cache" "${PROJECT_ROOT}/src/riskaware_eda.egg-info"; do
  assert_workspace_target "${target}"
  rm -rf -- "${target}"
done
find src tests -depth -type d -name __pycache__ -exec rm -rf -- {} +

printf 'Workspace organized: %s\n' "${PROJECT_ROOT}"
