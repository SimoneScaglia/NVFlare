#!/usr/bin/env python3
import json
import os
import sys
import subprocess
from datetime import datetime
import re
import pandas as pd

def run_command(command, shell=True):
    """Execute a shell command and return the result."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    activate_env = "source swarm_env/bin/activate && "
    full_command = activate_env + command
    print(f"\nExecuting: {full_command}")
    result = subprocess.run(full_command, shell=shell, capture_output=True, text=True, cwd=script_dir, executable='/bin/bash')
    if result.returncode != 0:
        print(f"Error executing command: {command}")
        print(f"STDOUT: {result.stdout}")
        print(f"STDERR: {result.stderr}")
    return result

def edit_config_file(config_path, min_responses_required, learning_rate, batch_size):
    """Edit the config file with values from JSON."""
    print(f"\nEditing config file: {config_path}")
    
    try:
        with open(config_path, 'r') as f:
            content = f.read()
        
        # Update min_responses_required
        pattern_min_responses = r'min_responses_required\s*=\s*\d+'
        replacement_min_responses = f'min_responses_required = {min_responses_required}'
        content = re.sub(pattern_min_responses, replacement_min_responses, content)
        
        # Find the mimic-learner component and update lr and batch_size within its args
        # First, let's find the mimic-learner section and update both lr and batch_size
        learner_pattern = r'(id = "mimic-learner"[^{]*args \{)([^}]+)(\})'
        
        def update_learner_args(match):
            args_section = match.group(2)
            # Update lr
            args_section = re.sub(r'lr\s*=\s*\d+\.?\d*', f'lr = {learning_rate}', args_section)
            # Update batch_size
            args_section = re.sub(r'batch_size\s*=\s*\d+', f'batch_size = {batch_size}', args_section)
            return match.group(1) + args_section + match.group(3)
        
        content = re.sub(learner_pattern, update_learner_args, content, flags=re.DOTALL)
        
        # Write back the modified content
        with open(config_path, 'w') as f:
            f.write(content)
        
        print("Config file updated successfully")
        
    except Exception as e:
        print(f"Error editing config file: {e}")
        sys.exit(1)

def copy_data_files(data_dir, iteration):
    """Copy node CSV files to /tmp/mimic_data."""
    print(f"\nCopying data files to /tmp/mimic_data/")
    ids = [1, 2, 3, 4, 5]
    
    # Create destination directory if it doesn't exist
    dest_dir = "/tmp/mimic_data"
    os.makedirs(dest_dir, exist_ok=True)
    
    for i in range(len(ids)):
        src_file = os.path.join(data_dir, f"{ids[i]}_train.csv")

        # Before copying, shuffle dataset based on iteration
        df = pd.read_csv(src_file)
        df = df.sample(frac=1, random_state=iteration).reset_index(drop=True)

        dest_file = os.path.join(dest_dir, f"site-{i + 1}.csv")
        df.to_csv(dest_file, index=False)

def main():
    # Check if JSON file path is provided
    if len(sys.argv) < 2:
        print("Usage: python script.py <path_to_json_file>")
        sys.exit(1)
    
    json_path = sys.argv[1]
    
    # Print current datetime
    print(f"\n{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - Starting script")
    
    # Load JSON file
    try:
        with open(json_path, 'r') as f:
            config = json.load(f)
        
        print(f"Loaded configuration from: {json_path}")
        print(f"Experiment: {config.get('experiment_name')}")
        
    except FileNotFoundError:
        print(f"Error: JSON file not found: {json_path}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON format: {e}")
        sys.exit(1)
    
    # Activate virtual environment
    print("\nActivating virtual environment...")
    run_command("source swarm_env/bin/activate", shell=True)
    
    # Edit config file
    config_file_path = "../../../job_templates/swarm_cse_tf_model_learner/config_fed_client.conf"
    
    # Get values from JSON
    min_responses_for_aggregation = config.get("min_responses_for_aggregation", 0)
    learning_rate = config.get("hyperparameters", {}).get("learning_rate", 0)
    batch_size = config.get("hyperparameters", {}).get("batch_size", 0)
    
    # Edit the config file
    edit_config_file(config_file_path, min_responses_for_aggregation, learning_rate, batch_size)
    
    # Copy data files
    data_directory = config.get("data_directory", "")
    iteration = config.get("iteration", 0)
    num_nodes = config.get("num_nodes", 0)
    
    copy_data_files(data_directory, iteration)
    
    # Rsync command
    print("\nSyncing code...")
    rsync_result = run_command("rsync -av --exclude='dataset' ../mimic ./code/")
    if rsync_result.returncode != 0:
        print("Warning: Rsync command failed")
    
    # Create nvflare job
    num_clients = num_nodes
    job_name = f"mimic_swarm_{num_clients}"
    print(f"\nCreating nvflare job: {job_name}")
    
    create_job_cmd = f"nvflare job create -j ./jobs/{job_name} -w swarm_cse_tf_model_learner -sd ./code -force"
    run_command(create_job_cmd)
    
    # Run nvflare simulator
    print(f"\nRunning nvflare simulator with {num_clients} clients")
    simulator_cmd = f"nvflare simulator ./jobs/{job_name} -w /tmp/nvflare/{job_name} -n {num_clients} -t {num_clients}"
    run_command(simulator_cmd)
    
    # Run validation script
    print(f"\nRunning validation script...")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    val_script = os.path.join(script_dir, "../mimic/utils/val_glob_model_eicu.py")
    
    json_path = f"../swarm_eicu/{json_path}"
    if os.path.exists(val_script):
        run_command(f"python3 \"{val_script}\" \"{json_path}\"")
    else:
        # Try alternative path
        val_script = os.path.join(os.path.dirname(script_dir), "mimic/utils/val_glob_model_eicu.py")
        if os.path.exists(val_script):
            run_command(f"python3 \"{val_script}\" \"{json_path}\"")
        else:
            print(f"Warning: Validation script not found at {val_script}")
    
    # Clean up temporary directory
    print(f"\nCleaning up temporary directory...")
    temp_dir = f"/tmp/nvflare/mimic_swarm_{num_clients}"
    if os.path.exists(temp_dir):
        run_command(f"rm -rf \"{temp_dir}\"")
        print(f"Removed: {temp_dir}")
    else:
        print(f"Directory does not exist: {temp_dir}")
    
    # Print final timestamp
    print(f"\n{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - Script completed")

if __name__ == "__main__":
    main()
