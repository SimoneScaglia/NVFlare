source swarm_env/bin/activate

./prepare_data.sh $1

cp -r ../mimic ./code/

nvflare job create -j ./jobs/mimic_swarm_$1 -w swarm_cse_tf_model_learner -sd ./code -force

nvflare simulator ./jobs/mimic_swarm_$1 -w /tmp/nvflare/mimic_swarm_$1 -n $1 -t $1

python3 -m json.tool /tmp/nvflare/mimic_swarm_$1/server/simulate_job/cross_site_val/cross_val_results.json