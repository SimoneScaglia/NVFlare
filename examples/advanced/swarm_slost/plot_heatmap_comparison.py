import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import matplotlib.colors as mcolors

os.chdir(Path(__file__).resolve().parent)

def parse_lr_from_dir(dir_path):
    """Extract learning rate from directory name"""
    dir_name = os.path.basename(dir_path)
    lr_part = dir_name.split('lr')[-1].split('_')[0]
    # Convert string like '0-0010' to float 0.001
    lr_str = lr_part.replace('-', '.')
    return float(lr_str)

def parse_bs_from_dir(dir_path):
    """Extract batch size from directory name"""
    dir_name = os.path.basename(dir_path)
    bs_part = dir_name.split('bs')[-1]
    return int(bs_part)

def load_results(directories, csv_filename, learning_rates=None, batch_sizes=None):
    """Load results from directories with given CSV filename"""
    auc_results = {bs: {lr: 0 for lr in learning_rates} for bs in batch_sizes}
    loss_results = {bs: {lr: 0 for lr in learning_rates} for bs in batch_sizes}
    
    for directory in directories:
        csv_path = os.path.join(directory, csv_filename)
        
        if os.path.exists(csv_path):
            try:
                df = pd.read_csv(csv_path)

                # Extract parameters
                lr = parse_lr_from_dir(directory)
                bs = parse_bs_from_dir(directory)

                if bs in auc_results and lr in auc_results[bs]:
                    auc_results[bs][lr] = df["auc"].mean()
                    loss_results[bs][lr] = df["loss"].mean()

            except Exception as e:
                print(f"Error processing {directory}: {e}")
        else:
            print(f"File not found: {csv_path}")

    return auc_results, loss_results

def build_directories(base_dir, learning_rates, batch_sizes, nodes):
    directories = []

    for lr in learning_rates:
        lr_str_5 = f"{lr:.5f}"

        for bs in batch_sizes:
            dir_5 = f"{nodes}_0_lr{lr_str_5}_bs{bs}"
            full_5 = os.path.join(base_dir, dir_5)

            if os.path.exists(full_5):
                directories.append(dir_5)
                continue

            lr_str_4 = f"{lr:.4f}"
            dir_4 = f"{nodes}_0_lr{lr_str_4}_bs{bs}"
            full_4 = os.path.join(base_dir, dir_4)

            if os.path.exists(full_4):
                directories.append(dir_4)
                continue

    return directories

def create_comparison_heatmap(comparison, metric='auc', discrete=False):
    learning_rates = [0.0001, 0.00025, 0.0005, 0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.075, 0.1]
    batch_sizes = [8, 16, 32, 64, 128, 256, 512]
    
    conf1_dirs = [os.path.join(comparison['dir1'], d) for d in build_directories(comparison['dir1'], learning_rates, batch_sizes, comparison['node1'])]
    conf2_dirs = [os.path.join(comparison['dir2'], d) for d in build_directories(comparison['dir2'], learning_rates, batch_sizes, comparison['node2'])]
    
    conf1_auc, conf1_loss = load_results(conf1_dirs, comparison['file1'], learning_rates, batch_sizes)
    conf2_auc, conf2_loss = load_results(conf2_dirs, comparison['file2'], learning_rates, batch_sizes)
    
    if metric == 'auc':
        conf1_matrix = conf1_auc
        conf2_matrix = conf2_auc
        metric_title = 'AUC'
    else:
        conf1_matrix = conf1_loss
        conf2_matrix = conf2_loss
        metric_title = 'Loss'
    
    batch_sizes = sorted(list(conf1_matrix.keys()))
    learning_rates = sorted(list(next(iter(conf1_matrix.values())).keys()))
    
    conf1_values = np.zeros((len(batch_sizes), len(learning_rates)))
    conf2_values = np.zeros((len(batch_sizes), len(learning_rates)))
    
    for i, bs in enumerate(batch_sizes):
        for j, lr in enumerate(learning_rates):
            conf1_values[i, j] = conf1_matrix[bs][lr]
            conf2_values[i, j] = conf2_matrix[bs][lr]
    
    if metric == 'auc':
        difference = conf1_values - conf2_values
        cmap = 'RdYlGn'
    else:
        difference = conf1_values - conf2_values
        cmap = 'RdYlGn'
    
    # Find best values
    best_A_idx = np.unravel_index(np.argmax(conf1_values) if metric == 'auc' else np.argmin(conf1_values), conf1_values.shape)
    best_B_idx = np.unravel_index(np.argmax(conf2_values) if metric == 'auc' else np.argmin(conf2_values), conf2_values.shape)
    best_delta_idx = np.unravel_index(np.argmax(difference) if metric == 'auc' else np.argmin(difference), difference.shape)
    
    out_dir = 'plots_results/plot_heatmap_comparison'
    os.makedirs(out_dir, exist_ok=True)
    
    plt.figure(figsize=(28, 14))
    
    annotation_matrix = np.empty_like(difference, dtype=object)
    for i in range(len(batch_sizes)):
        for j in range(len(learning_rates)):
            conf1_val = conf1_values[i, j]
            conf2_val = conf2_values[i, j]
            diff_val = difference[i, j]
            
            # Create annotation with bold formatting for best values
            a_text = f"A:{conf1_val:.4f}"
            b_text = f"B:{conf2_val:.4f}"
            delta_text = f"Δ:{diff_val:+.4f}"
            
            # Apply bold formatting to best values
            if (i, j) == best_A_idx:
                a_text = f"**{a_text}**"
            if (i, j) == best_B_idx:
                b_text = f"**{b_text}**"
            if (i, j) == best_delta_idx:
                delta_text = f"**{delta_text}**"
            
            annotation_matrix[i, j] = f"{a_text}\n{b_text}\n{delta_text}"

    if discrete:
        bounds = [-1, -0.05, -0.01, 0.01, 0.05, 1]
        colors = ["#8b0000", "#ff4500", "#ffff66", "#66ff66", "#006400"]
        cmap = mcolors.ListedColormap(colors)
        norm = mcolors.BoundaryNorm(bounds, cmap.N)
    
    # Plot heatmap
    heatmap = sns.heatmap(difference,
                        xticklabels=[f"{lr:.5f}" for lr in learning_rates],
                        yticklabels=batch_sizes,
                        annot=annotation_matrix,
                        fmt="",
                        cmap=cmap,
                        norm=norm if discrete else None,
                        linewidths=0.5,
                        linecolor='white',
                        annot_kws={'size': 18, 'weight': 'normal', 'color': 'black'},
                        vmin=-0.05 if metric == 'auc' else -0.05,
                        vmax=0.05 if metric == 'auc' else 0.05)
    heatmap.set_xticklabels(heatmap.get_xticklabels(), fontsize=18)
    heatmap.set_yticklabels(heatmap.get_yticklabels(), fontsize=18)
    
    exp = comparison['exp']
    plt.title(f'{metric_title} Comparison: {exp}\n')
    plt.xlabel('Learning Rate')
    plt.ylabel('Batch Size')
    
    # Adjust layout to prevent text cutoff
    plt.tight_layout()
    
    # Save plot
    exp_replaced = exp.replace(' ', '_').replace('=', 'vs').replace('(', '').replace(')', '').replace(',', '').replace('/', '_')
    filename = f'{out_dir}/comparison_heatmap_{metric}_{exp_replaced}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Comparison heatmap saved: {filename}")

def try_plot_comparison(comparison, discrete):
    """Wrapper function with error handling"""
    try:
        create_comparison_heatmap(comparison, metric='auc', discrete=discrete)
        # create_comparison_heatmap(swarm_res_dir, rows_per_node, metric='loss', discrete=discrete)
    except Exception as e:
        print(f"An error occurred while plotting comparison for {comparison['exp']} rows/node: {e}")

if __name__ == "__main__":
    # Define the mapping between swarm and central directories
    comparisons = [
        # SLOST vs NVFlare
        {
            'exp': 'SLOST_NVFLARE_A = Swarm 2 nodes NVFlare (1000 rows/node) vs B = Swarm 2 nodes SLOST (1000 rows/node)',
            'dir1': 'results/heatmap_experiments_1000rows_2nodes',
            'file1': 'swarm_results.csv',
            'node1': 2,
            'dir2': '/home/swarm-learning/repos/SLOST-local/results/heatmap_experiments_1000rows_2nodes',
            'file2': 'swarm_results.csv',
            'node2': 2
        },
        {
            'exp': 'SLOST_NVFLARE_A = Swarm 2 nodes NVFlare (2000 rows/node) vs B = Swarm 2 nodes SLOST (2000 rows/node)',
            'dir1': 'results/heatmap_experiments_2000rows_2nodes',
            'file1': 'swarm_results.csv',
            'node1': 2,
            'dir2': '/home/swarm-learning/repos/SLOST-local/results/heatmap_experiments_2000rows_2nodes',
            'file2': 'swarm_results.csv',
            'node2': 2
        },
        {
            'exp': 'SLOST_NVFLARE_A = Swarm 2 nodes NVFlare (4000 rows/node) vs B = Swarm 2 nodes SLOST (4000 rows/node)',
            'dir1': 'results/heatmap_experiments_4000rows_2nodes',
            'file1': 'swarm_results.csv',
            'node1': 2,
            'dir2': '/home/swarm-learning/repos/SLOST-local/results/heatmap_experiments_4000rows_2nodes',
            'file2': 'swarm_results.csv',
            'node2': 2
        },
        {
            'exp': 'SLOST_NVFLARE_A = Swarm 5 nodes NVFlare (1000 rows/node) vs B = Swarm 5 nodes SLOST (1000 rows/node)',
            'dir1': 'results/heatmap_experiments_1000rows_5nodes',
            'file1': 'swarm_results.csv',
            'node1': 5,
            'dir2': '/home/swarm-learning/repos/SLOST-local/results/heatmap_experiments_1000rows_5nodes',
            'file2': 'swarm_results.csv',
            'node2': 5
        },
        {
            'exp': 'SLOST_NVFLARE_A = Swarm 5 nodes NVFlare (2000 rows/node) vs B = Swarm 5 nodes SLOST (2000 rows/node)',
            'dir1': 'results/heatmap_experiments_2000rows_5nodes',
            'file1': 'swarm_results.csv',
            'node1': 5,
            'dir2': '/home/swarm-learning/repos/SLOST-local/results/heatmap_experiments_2000rows_5nodes',
            'file2': 'swarm_results.csv',
            'node2': 5
        },
        {
            'exp': 'SLOST_NVFLARE_A = Swarm 5 nodes NVFlare (4000 rows/node) vs B = Swarm 5 nodes SLOST (4000 rows/node)',
            'dir1': 'results/heatmap_experiments_4000rows_5nodes',
            'file1': 'swarm_results.csv',
            'node1': 5,
            'dir2': '/home/swarm-learning/repos/SLOST-local/results/heatmap_experiments_4000rows_5nodes',
            'file2': 'swarm_results.csv',
            'node2': 5
        },
        {
            'exp': 'SLOST_NVFLARE_A = Swarm 4 nodes NVFlare (entire mimic iii) vs B = Swarm 4 nodes SLOST (entire mimic iii)',
            'dir1': 'results/heatmap_experiments_4nodes_iii',
            'file1': 'swarm_results.csv',
            'node1': 4,
            'dir2': '/home/swarm-learning/repos/SLOST-local/results/heatmap_experiments_4nodes_iii',
            'file2': 'swarm_results.csv',
            'node2': 4
        },
        {
            'exp': 'SLOST_NVFLARE_A = Swarm 4 nodes NVFlare (entire mimic iv) vs B = Swarm 4 nodes SLOST (entire mimic iv)',
            'dir1': 'results/heatmap_experiments_4nodes_iv',
            'file1': 'swarm_results.csv',
            'node1': 4,
            'dir2': '/home/swarm-learning/repos/SLOST-local/results/heatmap_experiments_4nodes_iv',
            'file2': 'swarm_results.csv',
            'node2': 4
        },
        # 2 nodes
        {
            'exp': 'A = Swarm 2 nodes (1000 rows/node) vs B = Centralized (2000 rows)',
            'dir1': 'results/heatmap_experiments_1000rows_2nodes',
            'file1': 'swarm_results.csv',
            'node1': 2,
            'dir2': 'results/heatmap_experiments_1000rows_2nodes',
            'file2': 'central_results.csv',
            'node2': 2
        },
        {
            'exp': 'A = Swarm 2 nodes (2000 rows/node) vs B = Centralized (4000 rows)',
            'dir1': 'results/heatmap_experiments_2000rows_2nodes',
            'file1': 'swarm_results.csv',
            'node1': 2,
            'dir2': 'results/heatmap_experiments_2000rows_2nodes',
            'file2': 'central_results.csv',
            'node2': 2
        },
        {
            'exp': 'A = Swarm 2 nodes (4000 rows/node) vs B = Centralized (8000 rows)',
            'dir1': 'results/heatmap_experiments_4000rows_2nodes',
            'file1': 'swarm_results.csv',
            'node1': 2,
            'dir2': 'results/heatmap_experiments_4000rows_2nodes',
            'file2': 'central_results.csv',
            'node2': 2
        },
        # 5 nodes
        {
            'exp': 'A = Swarm 5 nodes (1000 rows/node) vs B = Centralized (5000 rows)',
            'dir1': 'results/heatmap_experiments_1000rows_5nodes',
            'file1': 'swarm_results.csv',
            'node1': 5,
            'dir2': 'results/heatmap_experiments_1000rows_5nodes',
            'file2': 'central_results.csv',
            'node2': 5
        },
        {
            'exp': 'A = Swarm 5 nodes (2000 rows/node) vs B = Centralized (10000 rows)',
            'dir1': 'results/heatmap_experiments_2000rows_5nodes',
            'file1': 'swarm_results.csv',
            'node1': 5,
            'dir2': 'results/heatmap_experiments_2000rows_5nodes',
            'file2': 'central_results.csv',
            'node2': 5
        },
        {
            'exp': 'A = Swarm 5 nodes (4000 rows/node) vs B = Centralized (20000 rows)',
            'dir1': 'results/heatmap_experiments_4000rows_5nodes',
            'file1': 'swarm_results.csv',
            'node1': 5,
            'dir2': 'results/heatmap_experiments_4000rows_5nodes',
            'file2': 'central_results.csv',
            'node2': 5
        },
        # entire datasets
        {
            'exp': 'A = Swarm 4 nodes vs B = Centralized (entire mimic iii)',
            'dir1': 'results/heatmap_experiments_4nodes_iii',
            'file1': 'swarm_results.csv',
            'node1': 4,
            'dir2': 'results/heatmap_experiments_4nodes_iii',
            'file2': 'central_results.csv',
            'node2': 4
        },
        {
            'exp': 'A = Swarm 4 nodes vs B = Centralized (entire mimic iv)',
            'dir1': 'results/heatmap_experiments_4nodes_iv',
            'file1': 'swarm_results.csv',
            'node1': 4,
            'dir2': 'results/heatmap_experiments_4nodes_iv',
            'file2': 'central_results.csv',
            'node2': 4
        }
    ]
    
    for comp in comparisons:
        try_plot_comparison(comp, discrete=True)
