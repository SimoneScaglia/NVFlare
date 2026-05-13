import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
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


def compute_separate_testset_iteration_aucs(df_epoch):
    """
    Compute one AUC per iteration for separate-testset evaluations.

    In separate-testset mode, each epoch has one row per user/site.
    We first aggregate users within each iteration+epoch, then average
    across iterations later in the pipeline.
    """
    if "iteration" in df_epoch.columns:
        return [compute_iteration_auc(df_iter) for _, df_iter in df_epoch.groupby("iteration")]
    return [compute_iteration_auc(df_epoch)]


def is_separate_testset_epoch(df_epoch):
    """
    Detect separate-testset rows by checking whether we have per-user rows
    rather than a pre-aggregated global user (e.g. swarm_global/central).
    """
    if "user" not in df_epoch.columns:
        return False

    users = {str(u).strip().lower() for u in df_epoch["user"].dropna().astype(str)}
    if not users:
        return False

    if "swarm_global" in users:
        return False
    if users == {"central"}:
        return False

    return len(users) > 1


def get_existing_csv_path(dir_path, csv_filenames):
    """Return the first existing CSV path from a filename or list of filenames."""
    if isinstance(csv_filenames, str):
        csv_filenames = [csv_filenames]

    for filename in csv_filenames:
        csv_path = dir_path / filename
        if csv_path.exists():
            return csv_path
    return None

def load_results_for_config(base_dir, subdir, csv_filenames):
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

        csv_path = get_existing_csv_path(dir_path, csv_filenames)
        if csv_path is None:
            continue

        try:
            df = pd.read_csv(csv_path)

            if 'epoch' not in df.columns:
                continue

            # One CSV contains multiple checkpoints (epochs).
            # Entire-testset mode: one pre-aggregated row per epoch.
            # Separate-testset mode: one row per user/site per epoch.
            # In separate mode we aggregate users first, then keep one value per
            # iteration so later code can average across iterations.
            for epoch_value, df_epoch in df.groupby('epoch'):
                epoch_int = int(epoch_value)
                if is_separate_testset_epoch(df_epoch):
                    iter_aucs = compute_separate_testset_iteration_aucs(df_epoch)
                else:
                    iter_aucs = [compute_iteration_auc(df_epoch)]

                results.setdefault(lr, {}).setdefault(bs, {}).setdefault(epoch_int, []).extend(iter_aucs)

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


def compute_max_mask(data):
    """Return a boolean mask selecting all cells equal to the matrix maximum."""
    if data.size == 0 or np.all(np.isnan(data)):
        return np.zeros_like(data, dtype=bool)
    max_value = np.nanmax(data)
    return np.isclose(data, max_value, rtol=1e-10, atol=1e-12)


def create_continuous_delta_cmap(vmin, vmax):
    """
    Create a continuous colormap for difference heatmaps that preserves
    the visual thresholds around [-0.05, -0.01, 0.01, 0.05] but blends
    smoothly between them. Returns (cmap, norm).
    """
    # Core colors: deep red, orange, yellow (for near-zero), yellow again,
    # light green, dark green
    colors = ["#8b0000", "#ff4500", "#ffff66", "#ffff66", "#66ff66", "#006400"]

    # Make symmetric range around zero so thresholds are absolute values
    sym = max(abs(vmin), abs(vmax)) if (vmin is not None and vmax is not None) else 1.0
    # Define absolute breakpoints we want to emphasize
    breakpoints = [-sym, -0.05, -0.01, 0.01, 0.05, sym]

    # Map breakpoints into 0..1 positions for the colormap
    if sym == 0:
        positions = [0.0 for _ in breakpoints]
    else:
        positions = [float((bp - (-sym)) / (2 * sym)) for bp in breakpoints]

    # Clamp positions to [0, 1]
    positions = [min(max(p, 0.0), 1.0) for p in positions]

    # Build a continuous LinearSegmentedColormap with explicit stops
    cmap = LinearSegmentedColormap.from_list("delta_continuous", list(zip(positions, colors)), N=256)

    # Use a symmetric Normalize to center zero
    norm = mcolors.Normalize(vmin=-sym, vmax=sym)
    return cmap, norm

# ------------------------------------------------------------------------------
# Plotting functions
# ------------------------------------------------------------------------------

def plot_heatmap(ax, data, epochs, lr_list, bs_list, title, cmap, vmin=None, vmax=None, norm=None, bold_mask=None):
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
                font_weight = 'bold' if (bold_mask is not None and bold_mask[i, j]) else 'normal'
                ax.text(j, i, text, ha='center', va='center', fontsize=13.5, color=text_color, fontweight=font_weight)
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

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Plot Swarm vs Central epoch analysis heatmaps. "
            "By default, processes the built-in dataset mapping."
        )
    )
    parser.add_argument(
        "--swarm-subdir",
        type=str,
        default=None,
        help="Optional swarm results subdirectory under new_results/",
    )
    parser.add_argument(
        "--central-subdir",
        type=str,
        default=None,
        help="Optional central results subdirectory under new_results/",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default=None,
        help="Optional output dataset label when using custom subdirs.",
    )

    args = parser.parse_args()

    if (args.swarm_subdir is None) ^ (args.central_subdir is None):
        parser.error("--swarm-subdir and --central-subdir must be provided together.")

    return args

def main():
    args = parse_args()

    # --- Configuration -------------------------------------------------------
    HEATMAP = True
    LINEPLOTS = False
    LINEPLOTS_GROUPED = False
    SCATTER = False

    # Base directories where results are stored
    SWARM_BASE = "new_results"
    CENTRAL_BASE = "new_results"

    # CSV filenames (adjust if needed)
    SWARM_CSV = ["swarm_results_entire_testset.csv", "swarm_results.csv"]
    CENTRAL_CSV = ["central_results.csv"]

    # Dataset subdirectories (must match the names used in your experiments)
    datasets = {
        # Homogeneous datasets
        "mimiciii_total": "mimiciii_total_entire_testset_epochs",
        "mimiciv_total": "mimiciv_total_entire_testset_epochs",
        "mimiciv_fixed": "mimiciv_fixed_12500_entire_testset_epochs",
        # Heterogeneous datasets
        # Not including diagnosis in the first 48 hours
        "eicu_data_20k_entire_testset": "eicu_data_20k_entire_testset",
        "eicu_data_20k_separate_testset": "eicu_data_20k_separate_testset",
        "eicu_data_20k_fixed_rows_entire_testset": "eicu_data_20k_fixed_rows_entire_testset",
        "eicu_data_20k_fixed_rows_separate_testset": "eicu_data_20k_fixed_rows_separate_testset",
        # Including diagnosis in the first 48 hours
        "eicu_data_entire_testset": "eicu_data_entire_testset",
        "eicu_data_separate_testset": "eicu_data_separate_testset",
        "eicu_data_fixed_rows_entire_testset": "eicu_data_fixed_rows_entire_testset",
        "eicu_data_fixed_rows_separate_testset": "eicu_data_fixed_rows_separate_testset"
    }

    if args.swarm_subdir and args.central_subdir:
        custom_name = args.dataset_name or f"swarm_{args.swarm_subdir}__central_{args.central_subdir}"
        dataset_jobs = [(custom_name, args.swarm_subdir, args.central_subdir)]
    else:
        dataset_jobs = []
        for name, subdir in datasets.items():
            dataset_jobs.append((name, subdir, subdir))

    output_root = Path("plots_results/epochs_analysis")
    output_root.mkdir(parents=True, exist_ok=True)

    # Colormap from the original script
    colors = ["#ffc6ba", "#ffd479", "#ffff66", "#a5db6f", "#7288cd"]
    cmap_heat = LinearSegmentedColormap.from_list("custom_cmap", colors)

    # -------------------------------------------------------------------------
    for dataset_name, swarm_subdir, central_subdir in dataset_jobs:
        print(f"\nProcessing {dataset_name}...")
        if swarm_subdir != central_subdir:
            print(f"  swarm from: {swarm_subdir}")
            print(f"  central from: {central_subdir}")
        dataset_output_dir = output_root / dataset_name
        dataset_output_dir.mkdir(parents=True, exist_ok=True)

        # Load swarm results
        swarm_results = load_results_for_config(SWARM_BASE, swarm_subdir, SWARM_CSV)
        # Load central results
        central_results = load_results_for_config(CENTRAL_BASE, central_subdir, CENTRAL_CSV)

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
        bold_swarm_best = compute_max_mask(best_auc_swarm)
        bold_central_best = compute_max_mask(best_auc_central)

        # --- Heatmaps (3 subplots side by side) ------------------------------
        if HEATMAP:
            fig, axes = plt.subplots(1, 3, figsize=(30, 10))

            # Swarm heatmap
            im1 = plot_heatmap(
                axes[0],
                best_auc_swarm,
                best_epoch_swarm,
                lr_list,
                bs_list,
                f"Swarm – {dataset_name}",
                cmap_heat,
                vmin,
                vmax,
                bold_mask=bold_swarm_best,
            )
            cbar1 = fig.colorbar(im1, ax=axes[0], orientation='vertical', shrink=0.8)
            cbar1.set_label('Best AUC')

            # Central heatmap
            im2 = plot_heatmap(
                axes[1],
                best_auc_central,
                best_epoch_central,
                lr_list,
                bs_list,
                f"Central – {dataset_name}",
                cmap_heat,
                vmin,
                vmax,
                bold_mask=bold_central_best,
            )
            cbar2 = fig.colorbar(im2, ax=axes[1], orientation='vertical', shrink=0.8)
            cbar2.set_label('Best AUC')

            # Difference heatmap (continuous, smooth transitions around key deltas)
            sym_diff = max(abs(np.nanmin(diff)), abs(np.nanmax(diff)))
            cmap_delta, norm_delta = create_continuous_delta_cmap(-sym_diff, sym_diff)
            im3 = plot_heatmap(axes[2], diff, None, lr_list, bs_list, f"Swarm - Central\n{dataset_name}", cmap_delta, vmin=None, vmax=None, norm=norm_delta)
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

                # continuous cmap for per-epoch diffs will be created per-epoch below

                n_epochs = len(all_epochs)
                fig, axes = plt.subplots(
                    n_epochs, 3,
                    figsize=(30, 5 * n_epochs),
                    squeeze=False
                )

                for row_idx, epoch in enumerate(all_epochs):
                    auc_swarm_ep = swarm_epoch_matrices[epoch]
                    auc_central_ep = central_epoch_matrices[epoch]
                    bold_swarm_epoch = compute_max_mask(auc_swarm_ep)
                    bold_central_epoch = compute_max_mask(auc_central_ep)
                    diff_ep = np.where(
                        ~np.isnan(auc_swarm_ep) & ~np.isnan(auc_central_ep),
                        auc_swarm_ep - auc_central_ep,
                        np.nan
                    )

                    im_s = plot_heatmap(
                        axes[row_idx, 0], auc_swarm_ep, None, lr_list, bs_list,
                        f"Swarm – epoch {epoch}", cmap_heat, epoch_vmin, epoch_vmax, bold_mask=bold_swarm_epoch
                    )
                    fig.colorbar(im_s, ax=axes[row_idx, 0], orientation='vertical', shrink=0.8).set_label('AUC')

                    im_c = plot_heatmap(
                        axes[row_idx, 1], auc_central_ep, None, lr_list, bs_list,
                        f"Central – epoch {epoch}", cmap_heat, epoch_vmin, epoch_vmax, bold_mask=bold_central_epoch
                    )
                    fig.colorbar(im_c, ax=axes[row_idx, 1], orientation='vertical', shrink=0.8).set_label('AUC')

                    # Create a continuous delta colormap specific to this epoch's value range
                    sym_ep = max(
                        abs(np.nanmin(diff_ep)) if not np.all(np.isnan(diff_ep)) else 0,
                        abs(np.nanmax(diff_ep)) if not np.all(np.isnan(diff_ep)) else 0,
                    )
                    cmap_delta_ep, norm_delta_ep = create_continuous_delta_cmap(-sym_ep, sym_ep)
                    im_d = plot_heatmap(
                        axes[row_idx, 2], diff_ep, None, lr_list, bs_list,
                        f"Swarm - Central – epoch {epoch}", cmap_delta_ep, None, None, norm=norm_delta_ep
                    )
                    fig.colorbar(im_d, ax=axes[row_idx, 2], orientation='vertical', shrink=0.8).set_label('Δ AUC')

                fig.suptitle(f"{dataset_name}\nAUC per (lr, bs) at each epoch", fontsize=18, y=1.002)
                plt.tight_layout()
                epoch_heatmap_file = dataset_output_dir / f"{dataset_name}_epoch_evolution_heatmaps.png"
                plt.savefig(epoch_heatmap_file, dpi=150, bbox_inches='tight')
                plt.close()
                print(f"Saved per-epoch heatmap matrix: {epoch_heatmap_file}")

        # --- Line plots per (lr, bs): x=Epoch, y=AUC ------------------------
        if LINEPLOTS:
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
                        swarm_line, = ax.plot(epochs_swarm, auc_swarm, marker='o', linewidth=2, markersize=4, label='Swarm')
                        swarm_max_mask = np.isclose(auc_swarm, np.nanmax(auc_swarm), rtol=1e-10, atol=1e-12)
                        ax.scatter(
                            epochs_swarm[swarm_max_mask],
                            auc_swarm[swarm_max_mask],
                            s=130,
                            color=swarm_line.get_color(),
                            marker='o',
                            edgecolors='black',
                            linewidths=0.8,
                            zorder=6,
                        )

                    if len(epochs_central) > 0:
                        central_line, = ax.plot(epochs_central, auc_central, marker='s', linewidth=2, markersize=4, label='Central')
                        central_max_mask = np.isclose(auc_central, np.nanmax(auc_central), rtol=1e-10, atol=1e-12)
                        ax.scatter(
                            epochs_central[central_max_mask],
                            auc_central[central_max_mask],
                            s=130,
                            color=central_line.get_color(),
                            marker='s',
                            edgecolors='black',
                            linewidths=0.8,
                            zorder=6,
                        )

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

        if LINEPLOTS_GROUPED:
            # --- Line plots fixing LR: compare all BS (Swarm vs Central) --------
            line_plots_fix_lr_dir = dataset_output_dir / "line_plots_fix_lr"
            line_plots_fix_lr_dir.mkdir(parents=True, exist_ok=True)

            bs_colors = plt.cm.tab20(np.linspace(0, 1, max(len(bs_list), 1)))

            for lr in lr_list:
                fig, ax = plt.subplots(figsize=(18, 9))
                plotted_any = False

                for idx, bs in enumerate(bs_list):
                    color = bs_colors[idx % len(bs_colors)]
                    epochs_swarm, auc_swarm = get_mean_curve(swarm_results, lr, bs)
                    epochs_central, auc_central = get_mean_curve(central_results, lr, bs)

                    if len(epochs_swarm) > 0:
                        ax.plot(
                            epochs_swarm,
                            auc_swarm,
                            color=color,
                            linestyle='-',
                            linewidth=2.0,
                            marker='o',
                            markersize=3,
                            label=f"bs={bs} | Swarm",
                        )
                        swarm_max_mask = np.isclose(auc_swarm, np.nanmax(auc_swarm), rtol=1e-10, atol=1e-12)
                        ax.scatter(
                            epochs_swarm[swarm_max_mask],
                            auc_swarm[swarm_max_mask],
                            s=130,
                            color=color,
                            marker='o',
                            edgecolors='black',
                            linewidths=0.8,
                            zorder=6,
                        )
                        plotted_any = True

                    # if len(epochs_central) > 0:
                    #     ax.plot(
                    #         epochs_central,
                    #         auc_central,
                    #         color=color,
                    #         linestyle='--',
                    #         linewidth=2.0,
                    #         marker='s',
                    #         markersize=3,
                    #         label=f"bs={bs} | Central",
                    #     )
                    #     plotted_any = True

                if not plotted_any:
                    plt.close(fig)
                    continue

                ax.set_xlim(0, 155)
                ax.set_xticks(np.arange(0, 155, 5))
                ax.set_ylim(0.4, 0.9)
                ax.set_xlabel('Epoch', fontsize=12)
                ax.set_ylabel('AUC', fontsize=12)
                ax.set_title(f"{dataset_name} | fixed lr={lr:.5f} | colors=bs, style=Swarm/Central", fontsize=12)
                ax.grid(True, alpha=0.3)
                ax.legend(loc='center left', bbox_to_anchor=(1.01, 0.5), fontsize=9)

                lr_label = f"{lr:.5f}".replace('.', '-')
                line_fix_lr_file = line_plots_fix_lr_dir / f"line_fix_lr{lr_label}_all_bs.png"
                plt.tight_layout()
                plt.savefig(line_fix_lr_file, dpi=300, bbox_inches='tight')
                plt.close()
                print(f"Saved fixed-lr line graph: {line_fix_lr_file}")

            # --- Line plots fixing BS: compare all LR (Swarm vs Central) --------
            line_plots_fix_bs_dir = dataset_output_dir / "line_plots_fix_bs"
            line_plots_fix_bs_dir.mkdir(parents=True, exist_ok=True)

            lr_colors = plt.cm.tab20(np.linspace(0, 1, max(len(lr_list), 1)))

            for bs in bs_list:
                fig, ax = plt.subplots(figsize=(18, 9))
                plotted_any = False

                for idx, lr in enumerate(lr_list):
                    color = lr_colors[idx % len(lr_colors)]
                    epochs_swarm, auc_swarm = get_mean_curve(swarm_results, lr, bs)
                    epochs_central, auc_central = get_mean_curve(central_results, lr, bs)

                    if len(epochs_swarm) > 0:
                        ax.plot(
                            epochs_swarm,
                            auc_swarm,
                            color=color,
                            linestyle='-',
                            linewidth=2.0,
                            marker='o',
                            markersize=3,
                            label=f"lr={lr:.5f} | Swarm",
                        )
                        swarm_max_mask = np.isclose(auc_swarm, np.nanmax(auc_swarm), rtol=1e-10, atol=1e-12)
                        ax.scatter(
                            epochs_swarm[swarm_max_mask],
                            auc_swarm[swarm_max_mask],
                            s=130,
                            color=color,
                            marker='o',
                            edgecolors='black',
                            linewidths=0.8,
                            zorder=6,
                        )
                        plotted_any = True

                    # if len(epochs_central) > 0:
                    #     ax.plot(
                    #         epochs_central,
                    #         auc_central,
                    #         color=color,
                    #         linestyle='--',
                    #         linewidth=2.0,
                    #         marker='s',
                    #         markersize=3,
                    #         label=f"lr={lr:.5f} | Central",
                    #     )
                    #     plotted_any = True

                if not plotted_any:
                    plt.close(fig)
                    continue

                ax.set_xlim(0, 155)
                ax.set_xticks(np.arange(0, 155, 5))
                ax.set_ylim(0.4, 0.9)
                ax.set_xlabel('Epoch', fontsize=12)
                ax.set_ylabel('AUC', fontsize=12)
                ax.set_title(f"{dataset_name} | fixed bs={bs} | colors=lr, style=Swarm/Central", fontsize=12)
                ax.grid(True, alpha=0.3)
                ax.legend(loc='center left', bbox_to_anchor=(1.01, 0.5), fontsize=9)

                line_fix_bs_file = line_plots_fix_bs_dir / f"line_fix_bs{bs}_all_lr.png"
                plt.tight_layout()
                plt.savefig(line_fix_bs_file, dpi=300, bbox_inches='tight')
                plt.close()
                print(f"Saved fixed-bs line graph: {line_fix_bs_file}")

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