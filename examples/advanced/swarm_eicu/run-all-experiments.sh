#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

PYTHON_BIN="../swarm_learning_tf/swarm_env/bin/python"
MODE="${1:-all}"   # all | swarm | central
CONFIG_DIR="${CONFIG_DIR:-configs_all_datasets_epochs}"

if [[ "${MODE}" != "all" && "${MODE}" != "swarm" && "${MODE}" != "central" ]]; then
  echo "Usage: $0 [all|swarm|central]"
  exit 1
fi

echo "Generating configs in ${CONFIG_DIR} ..."
"${PYTHON_BIN}" generate_configs_eicu.py --base-dir "${CONFIG_DIR}" --iterations 5

mapfile -t CONFIG_FILES < <(find "${CONFIG_DIR}" -maxdepth 1 -type f -name "*.json" | sort)

if [[ ${#CONFIG_FILES[@]} -eq 0 ]]; then
  echo "No config files found in ${CONFIG_DIR}"
  exit 1
fi

echo "Found ${#CONFIG_FILES[@]} configs"

i=0
for config in "${CONFIG_FILES[@]}"; do
  i=$((i + 1))
  echo
  echo "[${i}/${#CONFIG_FILES[@]}] Running config: ${config}"

  if [[ "${MODE}" == "all" || "${MODE}" == "swarm" ]]; then
    echo "  -> swarm"
    "${PYTHON_BIN}" ../swarm_learning_tf/run-experiment_eicu.py "${config}"
  fi

  if [[ "${MODE}" == "all" || "${MODE}" == "central" ]]; then
    echo "  -> central"
    "${PYTHON_BIN}" run-central.py -c "${config}"
  fi
done

echo

echo "All experiments completed. Results are saved under new_results/."
