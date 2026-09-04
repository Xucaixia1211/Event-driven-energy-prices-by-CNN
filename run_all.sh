#!/usr/bin/env bash
set -euo pipefail

repository_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$repository_dir"

run_notebook() {
  local source_notebook="$1"
  jupyter nbconvert \
    --to notebook \
    --execute "$source_notebook" \
    --stdout \
    --ExecutePreprocessor.timeout=-1 \
    > /dev/null
}

mode="${1:-}"
case "$mode" in
  empirical)
    run_notebook code/application/empirical_analysis.ipynb
    ;;
  simulation)
    THEOREM9_N_REPEATS=500 run_notebook \
      code/simulation/theory_validation_simulation.ipynb
    ;;
  simulation-smoke)
    THEOREM9_N_REPEATS=5 run_notebook \
      code/simulation/theory_validation_simulation.ipynb
    ;;
  all)
    "$0" empirical
    "$0" simulation
    ;;
  *)
    echo "Usage: $0 {empirical|simulation|simulation-smoke|all}" >&2
    exit 2
    ;;
esac
