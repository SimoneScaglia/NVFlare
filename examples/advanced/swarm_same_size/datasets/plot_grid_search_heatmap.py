#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Plot heatmaps for swarm grid search results.

Reads grid_search_n{X}.csv files produced by run-swarm-adaptive.py and
generates heatmaps showing mean AUC for each (batch_size, learning_rate)
combination.

Usage:
    python plot_grid_search_heatmap.py                  # both datasets
    python plot_grid_search_heatmap.py mimic_iii         # single dataset
"""

import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

# Grid search hyperparameters (must match run-swarm-adaptive.py)
LEARNING_RATES = [0.0001, 0.001, 0.01, 0.1]
BATCH_SIZES = [16, 32, 64, 128, 256, 512]

CONFIGS = ["5nodes", "10nodes", "20nodes", "40nodes"]
GROUP_SIZE = 5


def parse_csv_and_create_heatmap(csv_path, ax, label):
    """
    Read a grid search CSV and plot a heatmap on the given axis.

    CSV columns: num_nodes, iteration, site_id, batch_size, learning_rate,
                 loss, auc, auprc, accuracy, precision, recall
    """
    try:
        df = pd.read_csv(csv_path)
        print(f"Loaded {csv_path} with {len(df)} rows.")
    except Exception as e:
        print(f"Error reading {csv_path}: {e}")
        return None

    required_cols = ['batch_size', 'learning_rate', 'auc']
    if not all(col in df.columns for col in required_cols):
        print(f"Error: {csv_path} missing required columns: {required_cols}")
        return None

    # Build AUC matrix (batch_size x learning_rate)
    auc_matrix = np.zeros((len(BATCH_SIZES), len(LEARNING_RATES)))

    for i, bs in enumerate(BATCH_SIZES):
        for j, lr in enumerate(LEARNING_RATES):
            mask = (df['batch_size'] == bs) & (np.isclose(df['learning_rate'], lr))
            subset = df[mask]
            if len(subset) > 0:
                auc_matrix[i, j] = subset['auc'].mean()
            else:
                auc_matrix[i, j] = np.nan

    # Colormap
    colors = ["#ffc6ba", "#ffd479", "#ffff66", "#a5db6f", "#7288cd"]
    cmap = LinearSegmentedColormap.from_list("auc_cmap", colors)

    valid_values = auc_matrix[~np.isnan(auc_matrix)]
    if len(valid_values) > 0:
        vmin = valid_values.min()
        vmax = valid_values.max()
    else:
        vmin, vmax = 0, 1

    # Find max
    try:
        max_idx = np.nanargmax(auc_matrix)
        max_i, max_j = np.unravel_index(max_idx, auc_matrix.shape)
        max_auc = auc_matrix[max_i, max_j]
        max_lr = LEARNING_RATES[max_j]
        max_bs = BATCH_SIZES[max_i]
    except ValueError:
        max_i, max_j = 0, 0
        max_auc, max_lr, max_bs = 0, 0, 0

    print(f"  AUC max: {max_auc:.4f} at LR={max_lr}, BS={max_bs}")

    # Plot
    im = ax.imshow(auc_matrix, cmap=cmap, aspect='auto', vmin=vmin, vmax=vmax)

    for i in range(len(BATCH_SIZES)):
        for j in range(len(LEARNING_RATES)):
            rect = plt.Rectangle(
                (j - 0.5, i - 0.5), 1, 1,
                facecolor='none', edgecolor='black', linewidth=0.5
            )
            ax.add_patch(rect)

            if i == max_i and j == max_j:
                rect_max = plt.Rectangle(
                    (j - 0.5, i - 0.5), 1, 1,
                    facecolor='none', edgecolor='red', linewidth=3
                )
                ax.add_patch(rect_max)

            auc_val = auc_matrix[i, j]
            if not np.isnan(auc_val):
                color_rgb = cmap((auc_val - vmin) / (vmax - vmin)) if vmax > vmin else (1, 1, 1, 1)
                brightness = 0.299 * color_rgb[0] + 0.587 * color_rgb[1] + 0.114 * color_rgb[2]
                text_color = 'white' if brightness < 0.5 else 'black'
                fontweight = 'bold' if (i == max_i and j == max_j) else 'normal'
                ax.text(j, i, f"{auc_val:.4f}", ha='center', va='center',
                        fontsize=12, color=text_color, fontweight=fontweight)
            else:
                ax.text(j, i, "N/A", ha='center', va='center',
                        fontsize=10, color='gray', style='italic')

    ax.set_xticks(np.arange(len(LEARNING_RATES)))
    ax.set_yticks(np.arange(len(BATCH_SIZES)))
    ax.set_xticklabels([f"{lr}" for lr in LEARNING_RATES], fontsize=11)
    ax.set_yticklabels(BATCH_SIZES, fontsize=11)
    ax.set_xlabel('Learning Rate', fontsize=12, fontweight='bold')
    ax.set_ylabel('Batch Size', fontsize=12, fontweight='bold')
    ax.set_title(
        f'{label}\nMax AUC: {max_auc:.4f} (LR={max_lr}, BS={max_bs})',
        fontsize=12, fontweight='bold', pad=10
    )

    return im


def plot_config_heatmaps(base_folder):
    """
    For each configuration, plot all grid search heatmaps (one per boundary).
    """
    for config_name in CONFIGS:
        results_dir = Path(base_folder) / "same_size_results" / config_name
        if not results_dir.is_dir():
            print(f"Skipping {config_name}: no results directory")
            continue

        # Find all grid_search_n*.csv files
        gs_files = sorted(results_dir.glob("grid_search_n*.csv"))
        if not gs_files:
            print(f"Skipping {config_name}: no grid search files")
            continue

        n_plots = len(gs_files)
        fig, axes = plt.subplots(n_plots, 1, figsize=(12, 5 * n_plots))
        if n_plots == 1:
            axes = [axes]

        ims = []
        for idx, gs_file in enumerate(gs_files):
            # Extract boundary number from filename
            boundary = gs_file.stem.replace("grid_search_n", "")
            label = f"{config_name} - Grid Search at n={boundary} (Swarm)"

            im = parse_csv_and_create_heatmap(str(gs_file), axes[idx], label)
            if im is not None:
                ims.append((im, axes[idx]))

        for im, ax in ims:
            cbar = fig.colorbar(im, ax=ax, orientation='vertical', shrink=0.8, pad=0.02)
            cbar.set_label('AUC (mean)', fontsize=12, fontweight='bold')

        plt.tight_layout()

        output_dir = Path("plots_results") / base_folder
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f"grid_search_heatmap_{config_name}.png"
        plt.savefig(str(output_file), dpi=300, bbox_inches='tight')
        print(f"Saved: {output_file}")
        plt.close()


def plot_all_combined(base_folder):
    """
    Create a single combined plot with all grid search heatmaps for a dataset.
    """
    all_files = []
    all_labels = []

    for config_name in CONFIGS:
        results_dir = Path(base_folder) / "same_size_results" / config_name
        if not results_dir.is_dir():
            continue
        gs_files = sorted(results_dir.glob("grid_search_n*.csv"))
        for gs_file in gs_files:
            boundary = gs_file.stem.replace("grid_search_n", "")
            all_files.append(gs_file)
            all_labels.append(f"{config_name} - n={boundary}")

    if not all_files:
        print(f"No grid search files found for {base_folder}")
        return

    n_plots = len(all_files)
    fig, axes = plt.subplots(n_plots, 1, figsize=(12, 4 * n_plots))
    if n_plots == 1:
        axes = [axes]

    ims = []
    for idx, (gs_file, label) in enumerate(zip(all_files, all_labels)):
        im = parse_csv_and_create_heatmap(str(gs_file), axes[idx], label)
        if im is not None:
            ims.append((im, axes[idx]))

    for im, ax in ims:
        cbar = fig.colorbar(im, ax=ax, orientation='vertical', shrink=0.8, pad=0.02)
        cbar.set_label('AUC (mean)', fontsize=12, fontweight='bold')

    plt.tight_layout()

    output_dir = Path("plots_results") / base_folder
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "grid_search_heatmap_all_combined.png"
    plt.savefig(str(output_file), dpi=300, bbox_inches='tight')
    print(f"Saved combined plot: {output_file}")
    plt.close()


def main():
    if len(sys.argv) > 1:
        datasets = [sys.argv[1]]
    else:
        datasets = ["mimic_iii", "mimic_iv"]

    # Change to datasets directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)

    for dataset in datasets:
        print(f"\n{'='*60}")
        print(f"Plotting grid search heatmaps for {dataset}")
        print(f"{'='*60}")
        plot_config_heatmaps(dataset)
        plot_all_combined(dataset)

    print("\nAll plots created!")


if __name__ == "__main__":
    main()
