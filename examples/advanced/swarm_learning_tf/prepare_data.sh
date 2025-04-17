#!/bin/bash

num_clients=$1
shift
weights=("$@")

script_dir="$( dirname -- "$0"; )";
python3 "${script_dir}"/../mimic/utils/mimic_dataset.py "$num_clients" --stratify --weights "${weights[@]}"
