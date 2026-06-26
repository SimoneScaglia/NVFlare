#!/usr/bin/env python3
"""
Plot comparisons between Swarm (with FedProx mu from CSV column) and Central.
Swarm results are expected under fedprox_new_results/<dataset>/... with CSV containing 'fedprox_mu' column.
Central results under new_results/<dataset>/... (no mu column).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.colors as mcolors
from collections import defaultdict
import re

# ------------------------------------------------------------------------------
# Helper functions (adapted from original plot_heatmap_comparison_epochs.py)
# ------------------------------------------------------------------------------

def get_brightness(color):
    return 0.299 * color[0] + 0.587 * color[1] + 0.114 * color[2]

def parse_lr_from_dir(dir_name):
    lr_part = dir_name.split('lr')[-1].split('_')[0]
    lr_str = lr_part.replace('-', '.')
    return float(lr_str)

def parse_bs_from_dir(dir_name):
    bs_part = dir_name.split('bs')[-1]
    return int(bs_part)

def parse_epochs_from_dir(dir_name):
    return int(dir_name.split('_')[0])

def parse_iteration_from_dir(dir_name):
    return int(dir_name.split('_')[1])

def compute_iteration_auc(df):
    if "splits" in df.columns:
        split_sum = df["splits"].sum()
        if split_sum == 100:
            return (df["auc"] * df["splits"]).sum() / split_sum
        else:
            return df["auc"].mean()
    else:
        return df["auc"].mean()

def compute_separate_testset_iteration_aucs(df_epoch):
    if "iteration" in df_epoch.columns:
        return [compute_iteration_auc(df_iter) for _, df_iter in df_epoch.groupby("iteration")]
    return [compute_iteration_auc(df_epoch)]

def is_separate_testset_epoch(df_epoch):
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
    if isinstance(csv_filenames, str):
        csv_filenames = [csv_filenames]
    for filename in csv_filenames:
        csv_path = dir_path / filename
        if csv_path.exists():
            return csv_path
    return None

def load_central_results(base_dir, dataset_name, csv_filenames):
    """Load central results (no mu) from new_results/<dataset_name>/..."""
    base_path = Path(base_dir) / dataset_name
    if not base_path.exists():
        return {}
    results = {}
    for dir_path in base_path.iterdir():
        if not dir_path.is_dir():
            continue
        dir_name = dir_path.name
        try:
            _ = parse_iteration_from_dir(dir_name)
            lr = parse_lr_from_dir(dir_name)
            bs = parse_bs_from_dir(dir_name)
        except Exception:
            continue
        csv_path = get_existing_csv_path(dir_path, csv_filenames)
        if csv_path is None:
            continue
        try:
            df = pd.read_csv(csv_path)
            if 'epoch' not in df.columns:
                continue
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

def load_swarm_results_with_mu(base_dir, dataset_name, csv_filenames):
    """
    Load swarm results from fedprox_new_results/<dataset_name>/
    Expects subdirectories like <epochs>_<iter>_lr<lr>_bs<bs> each containing a CSV
    with a 'fedprox_mu' column.
    Returns dict: mu_value -> results (lr -> bs -> epoch -> list of AUCs)
    """
    base_path = Path(base_dir) / dataset_name
    if not base_path.exists():
        return {}
    mu_results = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list))))
    for dir_path in base_path.iterdir():
        if not dir_path.is_dir():
            continue
        dir_name = dir_path.name
        try:
            _ = parse_iteration_from_dir(dir_name)
            lr = parse_lr_from_dir(dir_name)
            bs = parse_bs_from_dir(dir_name)
        except Exception:
            continue
        csv_path = get_existing_csv_path(dir_path, csv_filenames)
        if csv_path is None:
            continue
        try:
            df = pd.read_csv(csv_path)
            if 'epoch' not in df.columns or 'fedprox_mu' not in df.columns:
                continue
            # Group by mu value
            for mu_val, df_mu in df.groupby('fedprox_mu'):
                for epoch_value, df_epoch in df_mu.groupby('epoch'):
                    epoch_int = int(epoch_value)
                    if is_separate_testset_epoch(df_epoch):
                        iter_aucs = compute_separate_testset_iteration_aucs(df_epoch)
                    else:
                        iter_aucs = [compute_iteration_auc(df_epoch)]
                    mu_results[mu_val][lr][bs][epoch_int].extend(iter_aucs)
        except Exception as e:
            print(f"Error processing {csv_path}: {e}")
    # Convert to regular dict
    return {mu: dict(res) for mu, res in mu_results.items()}

def compute_best_epoch_and_auc(results, lr_list, bs_list):
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
            best = max(epoch_auc.items(), key=lambda x: x[1])
            best_epoch[i, j] = best[0]
            best_auc[i, j] = best[1]
    return best_auc, best_epoch

def compute_auc_at_epoch(results, lr_list, bs_list, epoch):
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

def get_common_lr_bs(results_a, results_b):
    lr_set = set(results_a.keys()).intersection(set(results_b.keys()))
    lr_list = sorted(lr_set)
    bs_set = set()
    for lr in lr_list:
        bs_set.update(results_a[lr].keys())
        bs_set.update(results_b[lr].keys())
    bs_list = sorted(bs_set)
    return lr_list, bs_list

def get_mean_curve(results, lr, bs):
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
    if data.size == 0 or np.all(np.isnan(data)):
        return np.zeros_like(data, dtype=bool)
    max_value = np.nanmax(data)
    return np.isclose(data, max_value, rtol=1e-10, atol=1e-12)

def create_continuous_delta_cmap(vmin, vmax):
    colors = ["#8b0000", "#ff4500", "#ffff66", "#ffff66", "#66ff66", "#006400"]
    sym = max(abs(vmin), abs(vmax)) if (vmin is not None and vmax is not None) else 1.0
    breakpoints = [-sym, -0.05, -0.01, 0.01, 0.05, sym]
    positions = [float((bp - (-sym)) / (2 * sym)) for bp in breakpoints]
    positions = [min(max(p, 0.0), 1.0) for p in positions]
    cmap = LinearSegmentedColormap.from_list("delta_continuous", list(zip(positions, colors)), N=256)
    norm = mcolors.Normalize(vmin=-sym, vmax=sym)
    return cmap, norm

def plot_heatmap(ax, data, epochs, lr_list, bs_list, title, cmap, vmin=None, vmax=None, norm=None, bold_mask=None):
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
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if not np.isnan(data[i, j]):
                if vmin is not None and vmax is not None:
                    norm_val = (data[i, j] - vmin) / (vmax - vmin)
                else:
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

# ------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Plot Swarm (with mu from CSV) vs Central")
    parser.add_argument("--swarm-root", type=str, default="fedprox_new_results", help="Root for swarm results with mu column")
    parser.add_argument("--central-root", type=str, default="new_results", help="Root for central results")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset subdirectory name (e.g., eicu_data_20k_entire_testset)")
    parser.add_argument("--mu-list", nargs="+", type=float, help="Restrict to specific mu values (default: all found)")
    parser.add_argument("--swarm-csv", nargs="+", default=["swarm_results_entire_testset.csv", "swarm_results.csv"])
    parser.add_argument("--central-csv", nargs="+", default=["central_results.csv"])
    parser.add_argument("--output-dir", type=str, default="plots_results/mu_analysis", help="Root output directory")
    parser.add_argument("--heatmaps", action="store_true", default=True)
    parser.add_argument("--epoch-evolution", action="store_true", default=True)
    parser.add_argument("--lineplots", action="store_true", default=False)
    parser.add_argument("--compare-mu", action="store_true", default=False)
    return parser.parse_args()

def main():
    args = parse_args()

    # Load swarm results with mu
    swarm_mu_results = load_swarm_results_with_mu(args.swarm_root, args.dataset, args.swarm_csv)
    if not swarm_mu_results:
        print(f"No swarm results found for dataset {args.dataset} under {args.swarm_root}")
        return

    # Filter mu list
    if args.mu_list:
        swarm_mu_results = {mu: res for mu, res in swarm_mu_results.items() if mu in args.mu_list}
        if not swarm_mu_results:
            print("No matching mu values after filtering.")
            return

    # Load central results
    central_results = load_central_results(args.central_root, args.dataset, args.central_csv)
    if not central_results:
        print(f"No central results found for {args.dataset} under {args.central_root}")
        # Continue anyway, will plot only swarm per mu

    out_root = Path(args.output_dir) / args.dataset
    out_root.mkdir(parents=True, exist_ok=True)

    colors = ["#ffc6ba", "#ffd479", "#ffff66", "#a5db6f", "#7288cd"]
    cmap_auc = LinearSegmentedColormap.from_list("custom_cmap", colors)

    # For each mu, generate comparisons with central
    for mu, swarm_res in sorted(swarm_mu_results.items()):
        print(f"\nProcessing mu = {mu}")

        if central_results:
            lr_list, bs_list = get_common_lr_bs(swarm_res, central_results)
        else:
            lr_list = sorted(swarm_res.keys())
            bs_list = sorted(set(bs for lr in lr_list for bs in swarm_res[lr].keys()))

        if not lr_list or not bs_list:
            print(f"  No hyperparameters for mu={mu}, skip")
            continue

        best_auc_swarm, best_epoch_swarm = compute_best_epoch_and_auc(swarm_res, lr_list, bs_list)
        if central_results:
            best_auc_central, best_epoch_central = compute_best_epoch_and_auc(central_results, lr_list, bs_list)
            vmin = min(np.nanmin(best_auc_swarm), np.nanmin(best_auc_central))
            vmax = max(np.nanmax(best_auc_swarm), np.nanmax(best_auc_central))
            diff = best_auc_swarm - best_auc_central
            bold_swarm = compute_max_mask(best_auc_swarm)
            bold_central = compute_max_mask(best_auc_central)
        else:
            vmin = np.nanmin(best_auc_swarm)
            vmax = np.nanmax(best_auc_swarm)

        mu_out = out_root / f"mu_{mu}"
        mu_out.mkdir(exist_ok=True)

        if args.heatmaps:
            if central_results:
                fig, axes = plt.subplots(1, 3, figsize=(30, 10))
                im1 = plot_heatmap(axes[0], best_auc_swarm, best_epoch_swarm, lr_list, bs_list,
                                   f"Swarm (mu={mu})", cmap_auc, vmin, vmax, bold_mask=bold_swarm)
                fig.colorbar(im1, ax=axes[0]).set_label('Best AUC')
                im2 = plot_heatmap(axes[1], best_auc_central, best_epoch_central, lr_list, bs_list,
                                   "Central", cmap_auc, vmin, vmax, bold_mask=bold_central)
                fig.colorbar(im2, ax=axes[1]).set_label('Best AUC')
                sym_diff = max(abs(np.nanmin(diff)), abs(np.nanmax(diff)))
                cmap_diff, norm_diff = create_continuous_delta_cmap(-sym_diff, sym_diff)
                im3 = plot_heatmap(axes[2], diff, None, lr_list, bs_list,
                                   f"Swarm - Central (mu={mu})", cmap_diff, norm=norm_diff)
                fig.colorbar(im3, ax=axes[2]).set_label('Δ AUC')
                fig.suptitle(f"{args.dataset} – mu={mu}", fontsize=16, y=1.02)
                plt.tight_layout()
                plt.savefig(mu_out / "best_auc_3panel.png", dpi=300, bbox_inches='tight')
                plt.close()
            else:
                fig, ax = plt.subplots(figsize=(10, 8))
                im = plot_heatmap(ax, best_auc_swarm, best_epoch_swarm, lr_list, bs_list,
                                  f"Swarm (mu={mu})", cmap_auc, vmin, vmax)
                fig.colorbar(im, ax=ax).set_label('Best AUC')
                plt.tight_layout()
                plt.savefig(mu_out / "best_auc_swarm_only.png", dpi=300)
                plt.close()

        if args.epoch_evolution and central_results:
            # Collect epochs
            all_epochs = set()
            for lr in lr_list:
                for bs in bs_list:
                    if lr in swarm_res and bs in swarm_res[lr]:
                        all_epochs.update(swarm_res[lr][bs].keys())
                    if lr in central_results and bs in central_results[lr]:
                        all_epochs.update(central_results[lr][bs].keys())
            all_epochs = sorted([e for e in all_epochs if e % 10 == 0])
            if all_epochs:
                swarm_epoch_mats = {e: compute_auc_at_epoch(swarm_res, lr_list, bs_list, e) for e in all_epochs}
                central_epoch_mats = {e: compute_auc_at_epoch(central_results, lr_list, bs_list, e) for e in all_epochs}
                all_vals = np.concatenate([m.flatten() for m in swarm_epoch_mats.values()] +
                                          [m.flatten() for m in central_epoch_mats.values()])
                epoch_vmin, epoch_vmax = np.nanmin(all_vals), np.nanmax(all_vals)
                n_epochs = len(all_epochs)
                fig, axes = plt.subplots(n_epochs, 3, figsize=(30, 5 * n_epochs), squeeze=False)
                for row, epoch in enumerate(all_epochs):
                    auc_sw = swarm_epoch_mats[epoch]
                    bold_sw = compute_max_mask(auc_sw)
                    im_sw = plot_heatmap(axes[row, 0], auc_sw, None, lr_list, bs_list,
                                         f"Swarm (mu={mu}) epoch {epoch}", cmap_auc, epoch_vmin, epoch_vmax, bold_mask=bold_sw)
                    fig.colorbar(im_sw, ax=axes[row, 0]).set_label('AUC')
                    auc_cen = central_epoch_mats[epoch]
                    bold_cen = compute_max_mask(auc_cen)
                    im_cen = plot_heatmap(axes[row, 1], auc_cen, None, lr_list, bs_list,
                                          f"Central epoch {epoch}", cmap_auc, epoch_vmin, epoch_vmax, bold_mask=bold_cen)
                    fig.colorbar(im_cen, ax=axes[row, 1]).set_label('AUC')
                    diff_ep = auc_sw - auc_cen
                    sym_ep = max(abs(np.nanmin(diff_ep)), abs(np.nanmax(diff_ep))) if not np.all(np.isnan(diff_ep)) else 0
                    cmap_diff_ep, norm_diff_ep = create_continuous_delta_cmap(-sym_ep, sym_ep)
                    im_diff = plot_heatmap(axes[row, 2], diff_ep, None, lr_list, bs_list,
                                           f"Swarm - Central (mu={mu}) epoch {epoch}", cmap_diff_ep, norm=norm_diff_ep)
                    fig.colorbar(im_diff, ax=axes[row, 2]).set_label('Δ AUC')
                fig.suptitle(f"{args.dataset} – mu={mu}", fontsize=18, y=1.002)
                plt.tight_layout()
                plt.savefig(mu_out / "epoch_evolution_heatmaps.png", dpi=150, bbox_inches='tight')
                plt.close()

        if args.lineplots and len(swarm_mu_results) > 1:
            # For fixed (lr, bs) plot curves for different mu
            line_dir = mu_out / "lineplots_across_mu"
            line_dir.mkdir(exist_ok=True)
            # Sample a few hyperparameters
            sample_lrs = lr_list[:3] if len(lr_list) > 3 else lr_list
            sample_bss = bs_list[:3] if len(bs_list) > 3 else bs_list
            for lr in sample_lrs:
                for bs in sample_bss:
                    fig, ax = plt.subplots(figsize=(10, 6))
                    plotted = False
                    for mu_val, res in sorted(swarm_mu_results.items()):
                        epochs, aucs = get_mean_curve(res, lr, bs)
                        if len(epochs) > 0:
                            ax.plot(epochs, aucs, marker='o', label=f"mu={mu_val}")
                            plotted = True
                    if plotted:
                        ax.set_xlabel('Epoch')
                        ax.set_ylabel('Mean AUC')
                        ax.set_title(f"{args.dataset} – lr={lr:.5f}, bs={bs}")
                        ax.legend()
                        ax.grid(True, alpha=0.3)
                        plt.tight_layout()
                        plt.savefig(line_dir / f"line_lr{lr:.5f}_bs{bs}.png", dpi=300)
                        plt.close()

    # Cross-mu comparison (directly among swarm results)
    if args.compare_mu and len(swarm_mu_results) > 1:
        cross_out = out_root / "cross_mu"
        cross_out.mkdir(exist_ok=True)
        # Find common hyperparameters across all mu
        all_lrs = set()
        all_bss = set()
        for res in swarm_mu_results.values():
            all_lrs.update(res.keys())
            for lr in res:
                all_bss.update(res[lr].keys())
        all_lrs = sorted(all_lrs)
        all_bss = sorted(all_bss)
        if not all_lrs or not all_bss:
            print("No common hyperparameters for cross-mu comparison")
        else:
            mus = sorted(swarm_mu_results.keys())
            best_auc_3d = np.full((len(mus), len(all_bss), len(all_lrs)), np.nan)
            for i, mu in enumerate(mus):
                auc_mat, _ = compute_best_epoch_and_auc(swarm_mu_results[mu], all_lrs, all_bss)
                best_auc_3d[i, :, :] = auc_mat
            best_overall = np.nanmax(best_auc_3d, axis=0)
            for idx, mu in enumerate(mus):
                diff_mu = best_auc_3d[idx] - best_overall
                fig, ax = plt.subplots(figsize=(10, 8))
                sym = max(abs(np.nanmin(diff_mu)), abs(np.nanmax(diff_mu)))
                cmap_diff, norm_diff = create_continuous_delta_cmap(-sym, sym)
                im = plot_heatmap(ax, diff_mu, None, all_lrs, all_bss,
                                  f"mu={mu} – difference from best mu", cmap_diff, norm=norm_diff)
                fig.colorbar(im, ax=ax).set_label('Δ AUC (best = 0)')
                plt.tight_layout()
                plt.savefig(cross_out / f"diff_from_best_mu_{mu}.png", dpi=300)
                plt.close()

    print(f"\nAll plots saved under {out_root}")

if __name__ == "__main__":
    main()