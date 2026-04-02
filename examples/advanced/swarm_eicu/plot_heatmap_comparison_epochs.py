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
    New format: one folder per (total_epochs, iteration, lr, bs), one CSV with many epochs.
    Returns a nested dictionary: results[lr][bs][epoch] = list of iteration AUCs.
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
            _ = parse_iteration_from_dir(dir_name)
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

            if 'epoch' not in df.columns:
                continue

            # One CSV contains multiple checkpoints (epochs).
            # Rebuild the same comparison space as before by treating each epoch
            # as a candidate and collecting one AUC value per iteration.
            for epoch_value, df_epoch in df.groupby('epoch'):
                iter_auc = compute_iteration_auc(df_epoch)
                epoch_int = int(epoch_value)
                results.setdefault(lr, {}).setdefault(bs, {}).setdefault(epoch_int, []).append(iter_auc)

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

def compute_auc_at_epoch(results, lr_list, bs_list, epoch):
    """
    For each (lr, bs), compute average AUC at a specific epoch.
    Returns a 2D array (len(bs_list) x len(lr_list)).
    """
    auc_matrix = np.full((len(bs_list), len(lr_list)), np.nan)
    for i, bs in enumerate(bs_list):
        for j, lr in enumerate(lr_list):
            if lr not in results or bs not in results[lr]:
                continue
            if epoch not in results[lr][bs]:
                continue
            auc_list = results[lr][bs][epoch]
            if len(auc_list) > 0:
                auc_matrix[i, j] = np.mean(auc_list)
    return auc_matrix

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


def get_mean_curve(results, lr, bs):
    """
    Build mean AUC curve by epoch for one (lr, bs).
    Returns two arrays: epochs_sorted, auc_mean_sorted.
    """
    if lr not in results or bs not in results[lr]:
        return np.array([]), np.array([])

    epoch_auc = []
    for epoch_value, auc_list in results[lr][bs].items():
        if len(auc_list) == 0:
            continue
        epoch_auc.append((int(epoch_value), float(np.mean(auc_list))))

    if not epoch_auc:
        return np.array([]), np.array([])

    epoch_auc.sort(key=lambda x: x[0])
    epochs_sorted = np.array([x[0] for x in epoch_auc])
    auc_mean_sorted = np.array([x[1] for x in epoch_auc])
    return epochs_sorted, auc_mean_sorted

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

    output_root = Path("plots_results/epochs_analysis")
    output_root.mkdir(parents=True, exist_ok=True)

    # Colormap from the original script
    colors = ["#ffc6ba", "#ffd479", "#ffff66", "#a5db6f", "#7288cd"]
    cmap_heat = LinearSegmentedColormap.from_list("custom_cmap", colors)

    # -------------------------------------------------------------------------
    for dataset_name, subdir in datasets.items():
        print(f"\nProcessing {dataset_name}...")
        dataset_output_dir = output_root / dataset_name
        dataset_output_dir.mkdir(parents=True, exist_ok=True)

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
            heatmap_file = dataset_output_dir / f"{dataset_name}_best_auc_heatmap_3panel.png"
            plt.savefig(heatmap_file, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Saved 3-panel heatmap: {heatmap_file}")

            # --- Per-epoch heatmap matrix ------------------------------------
            # Collect all epochs present in either swarm or central results
            all_epochs = set()
            for lr in lr_list:
                for bs in bs_list:
                    if lr in swarm_results and bs in swarm_results.get(lr, {}):
                        all_epochs.update(swarm_results[lr][bs].keys())
                    if lr in central_results and bs in central_results.get(lr, {}):
                        all_epochs.update(central_results[lr][bs].keys())
            all_epochs = sorted(all_epochs)
            # Only proceed 10, 20, 30, etc
            all_epochs = [epoch for epoch in all_epochs if epoch % 10 == 0]

            if all_epochs:
                # Precompute AUC matrices for every epoch to get a global vmin/vmax
                swarm_epoch_matrices = {}
                central_epoch_matrices = {}
                for epoch in all_epochs:
                    swarm_epoch_matrices[epoch] = compute_auc_at_epoch(swarm_results, lr_list, bs_list, epoch)
                    central_epoch_matrices[epoch] = compute_auc_at_epoch(central_results, lr_list, bs_list, epoch)

                all_swarm_vals = np.concatenate([m.flatten() for m in swarm_epoch_matrices.values()])
                all_central_vals = np.concatenate([m.flatten() for m in central_epoch_matrices.values()])
                epoch_vmin = np.nanmin(np.concatenate([all_swarm_vals, all_central_vals]))
                epoch_vmax = np.nanmax(np.concatenate([all_swarm_vals, all_central_vals]))

                cmap_delta = mcolors.ListedColormap(["#8b0000", "#ff4500", "#ffff66", "#66ff66", "#006400"])
                norm_delta = mcolors.BoundaryNorm([-1, -0.05, -0.01, 0.01, 0.05, 1], cmap_delta.N)

                n_epochs = len(all_epochs)
                fig, axes = plt.subplots(
                    n_epochs, 3,
                    figsize=(30, 5 * n_epochs),
                    squeeze=False
                )

                for row_idx, epoch in enumerate(all_epochs):
                    auc_swarm_ep = swarm_epoch_matrices[epoch]
                    auc_central_ep = central_epoch_matrices[epoch]
                    diff_ep = np.where(
                        ~np.isnan(auc_swarm_ep) & ~np.isnan(auc_central_ep),
                        auc_swarm_ep - auc_central_ep,
                        np.nan
                    )

                    im_s = plot_heatmap(
                        axes[row_idx, 0], auc_swarm_ep, None, lr_list, bs_list,
                        f"Swarm – epoch {epoch}", cmap_heat, epoch_vmin, epoch_vmax
                    )
                    fig.colorbar(im_s, ax=axes[row_idx, 0], orientation='vertical', shrink=0.8).set_label('AUC')

                    im_c = plot_heatmap(
                        axes[row_idx, 1], auc_central_ep, None, lr_list, bs_list,
                        f"Central – epoch {epoch}", cmap_heat, epoch_vmin, epoch_vmax
                    )
                    fig.colorbar(im_c, ax=axes[row_idx, 1], orientation='vertical', shrink=0.8).set_label('AUC')

                    im_d = plot_heatmap(
                        axes[row_idx, 2], diff_ep, None, lr_list, bs_list,
                        f"Swarm - Central – epoch {epoch}", cmap_delta, None, None, norm=norm_delta
                    )
                    fig.colorbar(im_d, ax=axes[row_idx, 2], orientation='vertical', shrink=0.8).set_label('Δ AUC')

                fig.suptitle(f"{dataset_name}\nAUC per (lr, bs) at each epoch", fontsize=18, y=1.002)
                plt.tight_layout()
                epoch_heatmap_file = dataset_output_dir / f"{dataset_name}_epoch_evolution_heatmaps.png"
                plt.savefig(epoch_heatmap_file, dpi=150, bbox_inches='tight')
                plt.close()
                print(f"Saved per-epoch heatmap matrix: {epoch_heatmap_file}")

        # --- Line plots per (lr, bs): x=Epoch, y=AUC ------------------------
        line_plots_dir = dataset_output_dir / "line_plots"
        line_plots_dir.mkdir(parents=True, exist_ok=True)

        for lr in lr_list:
            for bs in bs_list:
                epochs_swarm, auc_swarm = get_mean_curve(swarm_results, lr, bs)
                epochs_central, auc_central = get_mean_curve(central_results, lr, bs)

                if len(epochs_swarm) == 0 and len(epochs_central) == 0:
                    continue

                fig, ax = plt.subplots(figsize=(16, 8))

                if len(epochs_swarm) > 0:
                    ax.plot(epochs_swarm, auc_swarm, marker='o', linewidth=2, markersize=4, label='Swarm')

                if len(epochs_central) > 0:
                    ax.plot(epochs_central, auc_central, marker='s', linewidth=2, markersize=4, label='Central')

                ax.set_xlim(0, 155)
                ax.set_xticks(np.arange(0, 155, 5))
                ax.set_ylim(0.4, 0.9)
                ax.set_xlabel('Epoch', fontsize=12)
                ax.set_ylabel('AUC', fontsize=12)
                ax.set_title(f"{dataset_name} | lr={lr:.5f}, bs={bs}", fontsize=12)
                ax.grid(True, alpha=0.3)
                ax.legend(loc='lower right')

                lr_label = f"{lr:.5f}".replace('.', '-')
                line_file = line_plots_dir / f"line_lr{lr_label}_bs{bs}.png"
                plt.tight_layout()
                plt.savefig(line_file, dpi=300, bbox_inches='tight')
                plt.close()
                print(f"Saved line graph: {line_file}")

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
            scatter_file = dataset_output_dir / f"{dataset_name}_3d_scatter_pair.png"
            plt.savefig(scatter_file, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Saved 3D scatter pair: {scatter_file}")

if __name__ == "__main__":
    main()