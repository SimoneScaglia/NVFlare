
# Swarm Learning on MIMIC Tabular Data with TensorFlow (NVFLARE)

This experiment demonstrates **swarm learning** using **NVFLARE** on **tabular MIMIC data**, with **TensorFlow**. The goal is to simulate swarm training across multiple sites using custom data splits.

## 📁 Directory Structure

```
swarm_learning_tf/
├── create_run.py
├── prepare_data.sh
├── start_exp.sh
├── code/
│   └── (populated automatically)
└── jobs/
    └── mimic_swarm_{num_clients}/
```

## ⚙️ Setup

Run the following commands from the **NVFLARE root directory**:

```
cd examples/advanced/swarm_learning_tf

# Create and activate Python virtual environment
python3 -m venv swarm_env
source swarm_env/bin/activate

# Go back to NVFLARE root
cd ../../..

# Install NVFLARE and dependencies
pip install -e .
pip install pandas scikit-learn tensorflow
```

## 🧪 Running the Experiment

### 1. Prepare Data
```
./prepare_data.sh <num_clients> [--stratify] --weights <w1> <w2> ...` 

-   `num_clients`: Number of clients/sites (e.g. 3)
    
-   `--stratify`: Optional flag for stratified split
    
-   `--weights`: Space-separated weights for each site's data
```
**Example:**

`./prepare_data.sh 3 --stratify --weights 0.5 0.25 0.25` 

This will generate split datasets as:

-   `/tmp/mimic_data/site-1.csv`, `site-2.csv`, ...
    
-   `/tmp/mimic_data/site-1.npy`, `site-2.npy`, ...

### 2. Run the Experiment

`./start_exp.sh <num_clients> <w1> <w2> ...` 

This script:

-   Activates the environment
    
-   Prepares data
    
-   Copies necessary code
    
-   Creates and runs the SL job using the `swarm_cse_tf_model_learner` job template
    
-   Runs validation on the global model
    
-   Cleans up temporary files
    

**Example:**

`./start_exp.sh 3 0.5 0.25 0.25` 

## 🧠 Code Overview

This experiment leverages custom code located in `examples/advanced/mimic`:

-   **learners/mimic_model_learner.py**  
    Custom learner logic executed on each client
    
-   **networks/mimic_nets.py**  
    CNN model used for tabular data
    
-   **utils/mimic_dataset.py**  
    Data loading and stratified/random splitting logic
    
-   **utils/val_glob_model.py**  
    Validates the aggregated global model after training
    

Job template used:  
📁 `job_templates/swarm_cse_tf_model_learner`  
You can configure the number of rounds, aggregation epochs, etc., inside the job template.

## 📊 Results

After execution, results are saved in:

`/tmp/nvflare/results/aucs_{num_clients}_nodes.csv` 

This file tracks the AUC of each client's CNN on the shared validation set:

```
user,split,auc,iteration
1,0.5,0.8301,0
2,0.25,0.8181,0
3,0.25,0.8181,0
...
``` 

-   `user`: Site ID
    
-   `split`: Training data weight
    
-   `auc`: Validation AUC
    
-   `iteration`: Iteration round

## 📂 Data Location

Place the full dataset files in:

```
mimic/dataset/
├── data_train.csv # Will be split each run
└── data_test.csv # Used for validation
``` 

## 🧼 Clean-up

Temporary simulation results are removed after each run by default.