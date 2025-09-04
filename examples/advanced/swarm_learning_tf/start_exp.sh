#!/bin/bash

iteration=$1
num_clients=$2
shift
shift
weights=("$@")

source swarm_env/bin/activate

sed -i "s/\(min_responses_required = \)[0-9]\+/\1$num_clients/" ../../../job_templates/swarm_cse_tf_model_learner/config_fed_client.conf

# ./prepare_data.sh "$num_clients" "${weights[@]}"
python3 prepare_data_from_csv.py "$iteration" "$num_clients" "${weights[@]}"

rsync -av --exclude='dataset' ../mimic ./code/

nvflare job create -j ./jobs/mimic_swarm_"$num_clients" -w swarm_cse_tf_model_learner -sd ./code -force

nvflare simulator ./jobs/mimic_swarm_"$num_clients" -w /tmp/nvflare/mimic_swarm_"$num_clients" -n "$num_clients" -t "$num_clients"

python3 -m json.tool /tmp/nvflare/mimic_swarm_"$num_clients"/server/simulate_job/cross_site_val/cross_val_results.json

script_dir="$( dirname -- "$0"; )";
python3 "${script_dir}"/../mimic/utils/val_glob_model.py "$iteration" "$num_clients" --weights "${weights[@]}"
python3 "${script_dir}"/../mimic/utils/val_local_model.py "$iteration" "$num_clients" --weights "${weights[@]}"

rm -r /tmp/nvflare/mimic_swarm_"$num_clients"