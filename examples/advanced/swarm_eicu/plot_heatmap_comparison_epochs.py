import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.colors as mcolors

# ------------------------------------------------------------------------------
# Helper functions (adapted from the original script)
# ------------------------------------------------------------------------------

def get_brightness(color):
    """Return perceived brightness of an RGB color."""
    return 0.299 * color[0] + 0.587 * color[1] + 0.114 * color[2]

def parse_lr_from_dir(dir_name):
    """Extract learning rate from directory name (format: *_lrX_bsY)."""
    lr_part = dir_name.split('lr')[-1].split('_')[0]
    lr_str = lr_part.replace('-', '.')
    return float(lr_str)

def parse_bs_from_dir(dir_name):
    """Extract batch size from directory name (format: *_bsY)."""
    bs_part = dir_name.split('bs')[-1]
    return int(bs_part)

def parse_epochs_from_dir(dir_name):
    """Extract total epochs from directory name (first number before first underscore)."""
    return int(dir_name.split('_')[0])

def parse_iteration_from_dir(dir_name):
    """Extract iteration number from directory name (second number)."""
    return int(dir_name.split('_')[1])

def compute_iteration_auc(df):
    """
    Given a DataFrame for one iteration (possibly with multiple splits),
    compute the average AUC for that iteration.
    If 'splits' column exists and sum(splits)==100, use weighted average,
    otherwise simple mean.
    """
    if "splits" in df.columns:
        split_sum = df["splits"].sum()
        if split_sum == 100:
            return (df["auc"] * df["splits"]).sum() / split_sum
        else:
            return df["auc"].mean()
    else:
        return df["auc"].mean()

def load_results_for_config(base_dir, subdir, csv_filename):
    """
    Load all results for a given configuration type (swarm or central).
    Returns a nested dictionary: results[lr][bs][epochs] = list of iteration AUCs.
    """
    results = {}
    dataset_path = Path(base_dir) / subdir
    if not dataset_path.exists():
        print(f"Warning: path does not exist: {dataset_path}")
        return results

    for dir_path in dataset_path.iterdir():
        if not dir_path.is_dir():
            continue
        dir_name = dir_path.name
        try:
            total_epochs = parse_epochs_from_dir(dir_name)
            iteration = parse_iteration_from_dir(dir_name)
            lr = parse_lr_from_dir(dir_name)
            bs = parse_bs_from_dir(dir_name)
        except Exception as e:
            # Skip directories that don't match the expected pattern
            continue

        csv_path = dir_path / csv_filename
        if not csv_path.exists():
            continue

        try:
            df = pd.read_csv(csv_path)
            # For this run (iteration), we need the AUC at the final epoch.
            # The CSV may contain rows for multiple epochs; we take the last epoch.
            if 'epoch' in df.columns:
                max_epoch = df['epoch'].max()
                df = df[df['epoch'] == max_epoch]
            iter_auc = compute_iteration_auc(df)

            # Store in nested dict
            results.setdefault(lr, {}).setdefault(bs, {}).setdefault(total_epochs, []).append(iter_auc)

        except Exception as e:
            print(f"Error processing {csv_path}: {e}")

    return results

def compute_best_epoch_and_auc(results, lr_list, bs_list):
    """
    For each (lr, bs), compute average AUC for each total_epochs and pick the best.
    Returns:
        best_auc_matrix: 2D array (len(bs_list) x len(lr_list)) with best AUC values.
        best_epoch_matrix: 2D array with the epoch count that gave the best AUC.
    """
    best_auc = np.full((len(bs_list), len(lr_list)), np.nan)
    best_epoch = np.full((len(bs_list), len(lr_list)), np.nan)

    for i, bs in enumerate(bs_list):
        for j, lr in enumerate(lr_list):
            if lr not in results or bs not in results[lr]:
                continue
            epoch_auc = {}
            for epochs, auc_list in results[lr][bs].items():
                if len(auc_list) > 0:
                    epoch_auc[epochs] = np.mean(auc_list)
            if not epoch_auc:
                continue
            # Find epoch with highest mean AUC
            best = max(epoch_auc.items(), key=lambda x: x[1])
            best_epoch[i, j] = best[0]
            best_auc[i, j] = best[1]
    return best_auc, best_epoch

def prepare_3d_data(results, lr_list, bs_list):
    """
    Prepare data for 3D scatter: arrays of lr, bs, epochs, and average AUC.
    Returns flat arrays for plotting.
    """
    x_vals, y_vals, z_vals, c_vals = [], [], [], []
    for lr in lr_list:
        if lr not in results:
            continue
        for bs in bs_list:
            if bs not in results[lr]:
                continue
            for epochs, auc_list in results[lr][bs].items():
                if len(auc_list) == 0:
                    continue
                mean_auc = np.mean(auc_list)
                x_vals.append(lr)
                y_vals.append(bs)
                z_vals.append(epochs)
                c_vals.append(mean_auc)
    return np.array(x_vals), np.array(y_vals), np.array(z_vals), np.array(c_vals)

def get_common_lr_bs(results_a, results_b):
    """
    Return sorted lists of learning rates and batch sizes that appear in both result dictionaries.
    """
    lr_set = set(results_a.keys()).intersection(set(results_b.keys()))
    lr_list = sorted(lr_set)
    bs_set = set()
    for lr in lr_list:
        bs_set.update(results_a[lr].keys())
        bs_set.update(results_b[lr].keys())
    bs_list = sorted(bs_set)
    return lr_list, bs_list

# ------------------------------------------------------------------------------
# Plotting functions
# ------------------------------------------------------------------------------

def plot_heatmap(ax, data, epochs, lr_list, bs_list, title, cmap, vmin=None, vmax=None, norm=None):
    """
    Draw a single heatmap on the given axes.
    data: 2D array (bs x lr) of values.
    epochs: 2D array (bs x lr) of best epoch numbers (or None for difference).
    """
    if vmin is None or vmax is None:
        im = ax.imshow(data, aspect='auto', cmap=cmap, norm=norm)
    else:
        im = ax.imshow(data, aspect='auto', cmap=cmap, norm=norm, vmin=vmin, vmax=vmax)

    ax.set_xticks(np.arange(len(lr_list)))
    ax.set_yticks(np.arange(len(bs_list)))
    ax.set_xticklabels([f"{lr:.5f}" for lr in lr_list], rotation=45, ha='right', fontsize=12)
    ax.set_yticklabels(bs_list, fontsize=12)
    ax.set_xlabel('Learning Rate', fontsize=14)
    ax.set_ylabel('Batch Size', fontsize=14)
    ax.set_title(title, fontsize=13)

    # Annotate cells
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if not np.isnan(data[i, j]):
                # Determine text colour based on background brightness
                if vmin is not None and vmax is not None:
                    norm_val = (data[i, j] - vmin) / (vmax - vmin)
                else:
                    # For auto-scaled, get normalized value from the image's norm
                    norm_val = im.norm(data[i, j])
                color = cmap(norm_val)
                text_color = 'white' if get_brightness(color) < 0.5 else 'black'

                if epochs is not None:
                    text = f"{data[i, j]:.4f}\nepoch={int(epochs[i, j])}"
                else:
                    text = f"{data[i, j]:+.4f}"
                ax.text(j, i, text, ha='center', va='center', fontsize=13.5, color=text_color, fontweight='normal')
    return im

def plot_3d_scatter_pair(ax, x, y, z, c, title):
    """
    Draw a 3D scatter plot on the given axes.
    """
    sc = ax.scatter(x, y, z, c=c, cmap='viridis', s=40, alpha=0.7)
    ax.set_xlabel('Learning Rate')
    ax.set_ylabel('Batch Size')
    ax.set_zlabel('Epochs')
    ax.set_title(title)
    return sc

# ------------------------------------------------------------------------------
# Main script
# ------------------------------------------------------------------------------

def main():
    # --- Configuration -------------------------------------------------------
    HEATMAP = True
    SCATTER = False

    # Base directories where results are stored
    SWARM_BASE = "new_results"
    CENTRAL_BASE = "new_results"

    # CSV filenames (adjust if needed)
    SWARM_CSV = "swarm_results_entire_testset.csv"
    CENTRAL_CSV = "central_results.csv"

    # Dataset subdirectories (must match the names used in your experiments)
    datasets = {
        "mimiciii_total": "mimiciii_total_entire_testset_epochs",
        "mimiciv_total": "mimiciv_total_entire_testset_epochs",
        "mimiciv_fixed": "mimiciv_fixed_12500_entire_testset_epochs",
    }

    output_dir = Path("plots_results/epoch_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Colormap from the original script
    colors = ["#ffc6ba", "#ffd479", "#ffff66", "#a5db6f", "#7288cd"]
    cmap_heat = LinearSegmentedColormap.from_list("custom_cmap", colors)

    # -------------------------------------------------------------------------
    for dataset_name, subdir in datasets.items():
        print(f"\nProcessing {dataset_name}...")

        # Load swarm results
        swarm_results = load_results_for_config(SWARM_BASE, subdir, SWARM_CSV)
        # Load central results
        central_results = load_results_for_config(CENTRAL_BASE, subdir, CENTRAL_CSV)

        if not swarm_results or not central_results:
            print(f"Missing results for {dataset_name}. Skipping.")
            continue

        # Get common learning rates and batch sizes
        lr_list, bs_list = get_common_lr_bs(swarm_results, central_results)
        if not lr_list or not bs_list:
            print(f"No overlapping hyperparameters for {dataset_name}. Skipping.")
            continue

        # Compute best AUC and best epoch matrices for both configurations
        best_auc_swarm, best_epoch_swarm = compute_best_epoch_and_auc(swarm_results, lr_list, bs_list)
        best_auc_central, best_epoch_central = compute_best_epoch_and_auc(central_results, lr_list, bs_list)

        # Determine common colour scale for swarm and central heatmaps
        vmin = min(np.nanmin(best_auc_swarm), np.nanmin(best_auc_central))
        vmax = max(np.nanmax(best_auc_swarm), np.nanmax(best_auc_central))

        # Compute difference matrix (Swarm - Central)
        diff = best_auc_swarm - best_auc_central

        # --- Heatmaps (3 subplots side by side) ------------------------------
        if HEATMAP:
            fig, axes = plt.subplots(1, 3, figsize=(30, 10))

            # Swarm heatmap
            im1 = plot_heatmap(axes[0], best_auc_swarm, best_epoch_swarm, lr_list, bs_list, f"Swarm – {dataset_name}", cmap_heat, vmin, vmax)
            cbar1 = fig.colorbar(im1, ax=axes[0], orientation='vertical', shrink=0.8)
            cbar1.set_label('Best AUC')

            # Central heatmap
            im2 = plot_heatmap(axes[1], best_auc_central, best_epoch_central, lr_list, bs_list, f"Central – {dataset_name}", cmap_heat, vmin, vmax)
            cbar2 = fig.colorbar(im2, ax=axes[1], orientation='vertical', shrink=0.8)
            cbar2.set_label('Best AUC')

            # Difference heatmap (auto-scaled)
            cmap_delta = mcolors.ListedColormap(["#8b0000", "#ff4500", "#ffff66", "#66ff66", "#006400"])
            norm_delta = mcolors.BoundaryNorm([-1, -0.05, -0.01, 0.01, 0.05, 1], cmap_delta.N)
            im3 = plot_heatmap(axes[2], diff, None, lr_list, bs_list, f"Swarm - Central\n{dataset_name}", cmap_delta, vmin = None, vmax = None, norm=norm_delta)
            cbar3 = fig.colorbar(im3, ax=axes[2], orientation='vertical', shrink=0.8)
            cbar3.set_label('Δ AUC')

            fig.suptitle(f"{dataset_name}\nBest AUC per (lr, bs) across epochs", fontsize=16, y=1.02)
            plt.tight_layout()
            heatmap_file = output_dir / f"{dataset_name}_best_auc_heatmap_3panel.png"
            plt.savefig(heatmap_file, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Saved 3-panel heatmap: {heatmap_file}")

        # --- 3D Scatter plots (2 subplots side by side) ----------------------
        if SCATTER:
            fig = plt.figure(figsize=(24, 12))
            ax1 = fig.add_subplot(121, projection='3d')
            ax2 = fig.add_subplot(122, projection='3d')

            x_s, y_s, z_s, c_s = prepare_3d_data(swarm_results, lr_list, bs_list)
            x_c, y_c, z_c, c_c = prepare_3d_data(central_results, lr_list, bs_list)

            if len(x_s) > 0:
                sc1 = plot_3d_scatter_pair(ax1, x_s, y_s, z_s, c_s, f"Swarm – {dataset_name}")
                cbar1 = fig.colorbar(sc1, ax=ax1, shrink=0.6)
                cbar1.set_label('AUC')
            else:
                ax1.text(0.5, 0.5, 0.5, "No data", transform=ax1.transAxes)

            if len(x_c) > 0:
                sc2 = plot_3d_scatter_pair(ax2, x_c, y_c, z_c, c_c, f"Central – {dataset_name}")
                cbar2 = fig.colorbar(sc2, ax=ax2, shrink=0.6)
                cbar2.set_label('AUC')
            else:
                ax2.text(0.5, 0.5, 0.5, "No data", transform=ax2.transAxes)

            fig.suptitle(f"{dataset_name}\nAUC vs (LR, BS, Epochs)", fontsize=16, y=1.02)
            plt.tight_layout()
            scatter_file = output_dir / f"{dataset_name}_3d_scatter_pair.png"
            plt.savefig(scatter_file, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Saved 3D scatter pair: {scatter_file}")

if __name__ == "__main__":
    main()