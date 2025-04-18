#!/bin/bash

num_clients=$1
shift
weights=("$@")

source swarm_env/bin/activate

./prepare_data.sh "$num_clients" "${weights[@]}"

cp -r ../mimic ./code/

nvflare job create -j ./jobs/mimic_swarm_"$num_clients" -w swarm_cse_tf_model_learner -sd ./code -force

nvflare simulator ./jobs/mimic_swarm_"$num_clients" -w /tmp/nvflare/mimic_swarm_"$num_clients" -n "$num_clients" -t "$num_clients"

python3 -m json.tool /tmp/nvflare/mimic_swarm_"$num_clients"/server/simulate_job/cross_site_val/cross_val_results.json

script_dir="$( dirname -- "$0"; )";
python3 "${script_dir}"/../mimic/utils/val_glob_model.py "$num_clients" --weights "${weights[@]}"

rm -r /tmp/nvflare/mimic_swarm_"$num_clients"