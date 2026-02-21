#!/usr/bin/env python3
"""
Generate comparison plots for local, central and swarm experiments.

Creates 2 images:
  - plots_results/plot_auc.png   (AUC vs num_nodes)
  - plots_results/plot_loss.png  (Loss vs num_nodes)

Each image contains 4 vertically-stacked subplots (5nodes, 10nodes, 20nodes, 40nodes).
Three lines per subplot: Local (mean across nodes), Central, Swarm (mean across sites).
Shaded bands show ±1 std.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ── Configuration ──────────────────────────────────────────────────────
CONFIGS = {
    "mimic_iv": [
        {"name": "5nodes",  "max_nodes": 5,  "label": "5 nodes - 8000 rows/node"},
        {"name": "10nodes", "max_nodes": 10, "label": "10 nodes - 4000 rows/node"},
        {"name": "20nodes", "max_nodes": 20, "label": "20 nodes - 2000 rows/node"},
        {"name": "40nodes", "max_nodes": 40, "label": "40 nodes - 1000 rows/node"},
    ],
    "mimic_iii": [
        {"name": "5nodes",  "max_nodes": 5,  "label": "5 nodes - 2500 rows/node"},
        {"name": "10nodes", "max_nodes": 10, "label": "10 nodes - 1250 rows/node"},
        {"name": "20nodes", "max_nodes": 20, "label": "20 nodes - 625 rows/node"},
        {"name": "40nodes", "max_nodes": 40, "label": "40 nodes - 312 rows/node"},
    ],
}

SCRIPT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = SCRIPT_DIR / "results"
OUTPUT_DIR = SCRIPT_DIR / "plots_results"


def load_and_aggregate(csv_path: Path, group_cols: list[str], metric: str):
    """
    Load a results CSV, group by `group_cols`, and compute mean/std of `metric`.
    Returns a DataFrame with columns: *group_cols, mean, std.
    """
    if not csv_path.is_file():
        print(f"WARNING: {csv_path} not found – skipping")
        return pd.DataFrame()

    df = pd.read_csv(csv_path)
    agg = df.groupby(group_cols)[metric].agg(["mean", "std"]).reset_index()
    return agg


def plot_metric(metric: str, ylabel: str, output_filename: str, dataset: str, ylim: tuple = None, shared_x: bool = False):
    """Create a 4-row figure for the given metric and save to disk.
    If shared_x=True, all subplots use x-axis 2..80.
    """

    # Load CSVs once
    local_df = pd.read_csv(RESULTS_DIR / dataset / "local_results.csv") if (RESULTS_DIR / dataset / "local_results.csv").is_file() else pd.DataFrame()
    central_df = pd.read_csv(RESULTS_DIR / dataset / "central_results.csv") if (RESULTS_DIR / dataset / "central_results.csv").is_file() else pd.DataFrame()
    swarm_df = pd.read_csv(RESULTS_DIR / dataset / "swarm_results.csv") if (RESULTS_DIR / dataset / "swarm_results.csv").is_file() else pd.DataFrame()

    global_max = max(c["max_nodes"] for c in CONFIGS[dataset])  # 80
    fig, axes = plt.subplots(4, 1, figsize=(16, 15))

    for idx, cfg in enumerate(CONFIGS[dataset]):
        ax = axes[idx]
        config_name = cfg["name"]
        max_nodes = cfg["max_nodes"]
        x_range = np.arange(2, max_nodes + 1)
        x_axis_max = global_max if shared_x else max_nodes

        # ── Local ──
        if not local_df.empty and config_name in local_df["configuration"].values:
            loc = local_df[local_df["configuration"] == config_name]
            loc_agg = loc.groupby("num_nodes")[metric].agg(["mean", "std"]).reset_index()
            loc_agg = loc_agg.set_index("num_nodes").reindex(x_range).reset_index()
            ax.plot(loc_agg["num_nodes"], loc_agg["mean"], "o-", color="#e74c3c", label="Local", linewidth=2, markersize=4)
            ax.fill_between(loc_agg["num_nodes"], loc_agg["mean"] - loc_agg["std"], loc_agg["mean"] + loc_agg["std"], color="#e74c3c", alpha=0.15)

        # ── Central ──
        if not central_df.empty and config_name in central_df["configuration"].values:
            cen = central_df[central_df["configuration"] == config_name]
            cen_agg = cen.groupby("num_nodes")[metric].agg(["mean", "std"]).reset_index()
            cen_agg = cen_agg.set_index("num_nodes").reindex(x_range).reset_index()
            ax.plot(cen_agg["num_nodes"], cen_agg["mean"], "s-", color="#2ecc71", label="Central", linewidth=2, markersize=4)
            ax.fill_between(cen_agg["num_nodes"], cen_agg["mean"] - cen_agg["std"], cen_agg["mean"] + cen_agg["std"], color="#2ecc71", alpha=0.15)

        # ── Swarm ──
        if not swarm_df.empty and config_name in swarm_df["configuration"].values:
            swm = swarm_df[swarm_df["configuration"] == config_name]
            swm_agg = swm.groupby("num_nodes")[metric].agg(["mean", "std"]).reset_index()
            swm_agg = swm_agg.set_index("num_nodes").reindex(x_range).reset_index()
            ax.plot(swm_agg["num_nodes"], swm_agg["mean"], "^-", color="#3498db", label="Swarm", linewidth=2, markersize=4)
            ax.fill_between(swm_agg["num_nodes"], swm_agg["mean"] - swm_agg["std"], swm_agg["mean"] + swm_agg["std"], color="#3498db", alpha=0.15)

        # ── Axes formatting ──
        ax.set_xlim(1.5, x_axis_max + 0.5)
        ax.set_xticks(np.arange(2, (global_max if shared_x else max_nodes) + 1))
        ax.tick_params(axis='x', labelsize=7)
        ax.set_xlabel("Number of nodes", fontsize=12, fontweight="bold")
        ax.set_ylabel(ylabel, fontsize=12, fontweight="bold")
        ax.set_title(cfg["label"], fontsize=13, fontweight="bold", pad=8)
        if ylim:
            ax.set_ylim(ylim)
        handles, _ = ax.get_legend_handles_labels()
        if handles:
            ax.legend(fontsize=11, loc="best")
        ax.grid(True, alpha=0.3)
        _annotate_hyperparams(ax, dataset, config_name, max_nodes,
                              local_df, central_df, swarm_df)

    plt.tight_layout()

    (OUTPUT_DIR / dataset).mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / dataset / output_filename
    fig.savefig(str(out_path), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def _compute_mean_iter_time(df: pd.DataFrame, config_name: str, x_range):
    """
    For each num_nodes, compute the mean time (in minutes) between consecutive
    iterations (0→1, 1→2, ..., 8→9).

    Within a single (config, num_nodes, iteration) there can be multiple rows
    (one per node/site). We take the max datetime per iteration (= when that
    iteration finished), then diff consecutive iterations.
    """
    if df.empty or config_name not in df["configuration"].values:
        return None

    sub = df[df["configuration"] == config_name].copy()
    sub["datetime"] = pd.to_datetime(sub["datetime"])

    records = []
    for nn in x_range:
        nn_df = sub[sub["num_nodes"] == nn]
        if nn_df.empty:
            records.append({"num_nodes": nn, "mean_minutes": np.nan})
            continue

        # max datetime per iteration (when it finished)
        iter_times = nn_df.groupby("iteration")["datetime"].max().sort_index()

        if len(iter_times) < 2:
            records.append({"num_nodes": nn, "mean_minutes": np.nan})
            continue

        diffs = iter_times.diff().dropna().abs().dt.total_seconds() / 60.0
        records.append({"num_nodes": nn, "mean_minutes": diffs.mean()})

    return pd.DataFrame(records)


# ── Hyperparameter extraction from results CSVs ──────────────────────
GROUP_SIZE = 5


def _get_params_from_results(df, config_name, num_nodes):
    """Extract the (lr, batch_size) used for a given (config, num_nodes) from a results CSV.

    All rows for the same (config, num_nodes) share the same hyperparameters,
    so we just take the first row.  Returns (lr, bs) or None.
    """
    if df.empty or 'lr' not in df.columns or 'batch_size' not in df.columns:
        return None
    sub = df[(df['configuration'] == config_name) & (df['num_nodes'] == num_nodes)]
    if sub.empty:
        return None
    row = sub.iloc[0]
    return row['lr'], int(row['batch_size'])


def _annotate_hyperparams(ax, dataset, config_name, max_nodes,
                          local_df, central_df, swarm_df):
    """Add hyperparameter annotations and group separators to a subplot.

    Reads lr / batch_size directly from the results DataFrames.
    - Vertical dashed lines at group boundaries (every GROUP_SIZE nodes)
    - Per-group annotations for swarm (blue) and central (green) params
    - Single annotation for local params (red), constant across all nodes
    """
    trans = ax.get_xaxis_transform()  # x in data coords, y in axes coords

    # Local params (same for entire configuration – pick from any num_nodes)
    local_params = None
    for nn in range(2, max_nodes + 1):
        local_params = _get_params_from_results(local_df, config_name, nn)
        if local_params:
            break
    if local_params:
        lr, bs = local_params
        ax.text(0.02, 0.04, f"Local: lr={lr:g}  bs={bs}",
                transform=ax.transAxes, fontsize=7, va='bottom', ha='left',
                color='#e74c3c', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                          edgecolor='#e74c3c', alpha=0.85, linewidth=0.6))

    # Per-group annotations for swarm & central
    for group_end in range(GROUP_SIZE, max_nodes + 1, GROUP_SIZE):
        group_start = max(2, group_end - GROUP_SIZE + 1)

        # Alternating background bands
        if (group_end // GROUP_SIZE) % 2 == 0:
            ax.axvspan(group_start - 0.5, group_end + 0.5,
                       alpha=0.04, color='steelblue', zorder=0)

        # Vertical separator at group boundary
        if group_end < max_nodes:
            ax.axvline(x=group_end + 0.5, color='gray',
                       linestyle=':', alpha=0.4, linewidth=0.8)

        # Pick params from any num_nodes in this group (they all share the same)
        swarm_params = None
        central_params = None
        for nn in range(group_start, group_end + 1):
            if swarm_params is None:
                swarm_params = _get_params_from_results(swarm_df, config_name, nn)
            if central_params is None:
                central_params = _get_params_from_results(central_df, config_name, nn)

        x_center = (group_start + group_end) / 2
        parts = []
        if swarm_params:
            lr, bs = swarm_params
            parts.append((f"S: lr={lr:g} bs={bs}", '#3498db'))
        if central_params:
            lr, bs = central_params
            parts.append((f"C: lr={lr:g} bs={bs}", '#2ecc71'))

        for i, (text, color) in enumerate(parts):
            y_pos = 0.96 - i * 0.09
            ax.text(x_center, y_pos, text, transform=trans,
                    fontsize=6, ha='center', va='top',
                    color=color, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                              edgecolor=color, alpha=0.8, linewidth=0.5))


def plot_time(dataset: str, shared_x: bool = False):
    """Create a 4-row figure with mean iteration time (minutes) vs num_nodes."""

    local_df = pd.read_csv(RESULTS_DIR / dataset / "local_results.csv") if (RESULTS_DIR / dataset / "local_results.csv").is_file() else pd.DataFrame()
    central_df = pd.read_csv(RESULTS_DIR / dataset / "central_results.csv") if (RESULTS_DIR / dataset / "central_results.csv").is_file() else pd.DataFrame()
    swarm_df = pd.read_csv(RESULTS_DIR / dataset / "swarm_results.csv") if (RESULTS_DIR / dataset / "swarm_results.csv").is_file() else pd.DataFrame()

    global_max = max(c["max_nodes"] for c in CONFIGS[dataset])
    fig, axes = plt.subplots(4, 1, figsize=(16, 15))

    for idx, cfg in enumerate(CONFIGS[dataset]):
        ax = axes[idx]
        config_name = cfg["name"]
        max_nodes = cfg["max_nodes"]
        x_range = np.arange(2, max_nodes + 1)
        x_axis_max = global_max if shared_x else max_nodes

        # Local
        loc_time = _compute_mean_iter_time(local_df, config_name, x_range)
        if loc_time is not None:
            ax.plot(loc_time["num_nodes"], loc_time["mean_minutes"], "o-", color="#e74c3c", label="Local", linewidth=2, markersize=4)

        # Central
        cen_time = _compute_mean_iter_time(central_df, config_name, x_range)
        if cen_time is not None:
            ax.plot(cen_time["num_nodes"], cen_time["mean_minutes"], "s-", color="#2ecc71", label="Central", linewidth=2, markersize=4)

        # Swarm
        swm_time = _compute_mean_iter_time(swarm_df, config_name, x_range)
        if swm_time is not None:
            ax.plot(swm_time["num_nodes"], swm_time["mean_minutes"], "^-", color="#3498db", label="Swarm", linewidth=2, markersize=4)

        # Axes formatting
        ax.set_xlim(1.5, x_axis_max + 0.5)
        ax.set_xticks(np.arange(2, (global_max if shared_x else max_nodes) + 1))
        ax.tick_params(axis='x', labelsize=7)
        ax.set_xlabel("Number of nodes", fontsize=12, fontweight="bold")
        ax.set_ylabel("Time per iteration (min)", fontsize=12, fontweight="bold")
        ax.set_title(cfg["label"], fontsize=13, fontweight="bold", pad=8)
        ax.set_ylim(0, 15)
        handles, _ = ax.get_legend_handles_labels()
        if handles:
            ax.legend(fontsize=11, loc="best")
        ax.grid(True, alpha=0.3)
        _annotate_hyperparams(ax, dataset, config_name, max_nodes,
                              local_df, central_df, swarm_df)

    plt.tight_layout()

    (OUTPUT_DIR / dataset).mkdir(parents=True, exist_ok=True)
    prefix = "aligned_x_" if shared_x else "own_x_"
    out_path = OUTPUT_DIR / dataset / f"{prefix}plot_time.png"
    fig.savefig(str(out_path), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


# ── Swarm comparison: all configs on one graph ─────────────────────────
SWARM_COLORS = {
    "5nodes":  "#f39c12",
    "10nodes": "#e74c3c",
    "20nodes": "#2ecc71",
    "40nodes": "#3498db",
}


def plot_swarm_comparison(metric: str, ylabel: str, output_filename: str, dataset: str, ylim: tuple = None):
    """Single graph with one line per configuration (swarm only)."""
    swarm_csv = RESULTS_DIR / dataset / "swarm_results.csv"
    if not swarm_csv.is_file():
        print(f"WARNING: {swarm_csv} not found – skipping swarm comparison")
        return

    swarm_df = pd.read_csv(swarm_csv)
    fig, ax = plt.subplots(figsize=(16, 6))

    for cfg in CONFIGS[dataset]:
        config_name = cfg["name"]
        max_nodes = cfg["max_nodes"]
        x_range = np.arange(2, max_nodes + 1)

        if config_name not in swarm_df["configuration"].values:
            continue

        sub = swarm_df[swarm_df["configuration"] == config_name]
        agg = sub.groupby("num_nodes")[metric].agg(["mean", "std"]).reset_index()
        agg = agg.set_index("num_nodes").reindex(x_range).reset_index()

        color = SWARM_COLORS[config_name]
        ax.plot(agg["num_nodes"], agg["mean"], "o-", color=color,
                label=cfg["label"], linewidth=2, markersize=4)
        ax.fill_between(agg["num_nodes"],
                        agg["mean"] - agg["std"],
                        agg["mean"] + agg["std"],
                        color=color, alpha=0.12)

    global_max = max(c["max_nodes"] for c in CONFIGS[dataset])
    ax.set_xlim(1.5, global_max + 0.5)
    ax.set_xticks(np.arange(2, global_max + 1))
    ax.tick_params(axis='x', labelsize=7)
    ax.set_xlabel("Number of nodes", fontsize=12, fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=12, fontweight="bold")
    ax.set_title(f"Swarm Learning – {ylabel} comparison", fontsize=14, fontweight="bold", pad=10)
    if ylim:
        ax.set_ylim(ylim)
    ax.legend(fontsize=11, loc="best")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    (OUTPUT_DIR / dataset).mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / dataset / output_filename
    fig.savefig(str(out_path), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_swarm_time_comparison(dataset: str):
    """Single graph comparing swarm iteration time across all configs."""
    swarm_csv = RESULTS_DIR / dataset / "swarm_results.csv"
    if not swarm_csv.is_file():
        print(f"WARNING: {swarm_csv} not found – skipping swarm time comparison")
        return

    swarm_df = pd.read_csv(swarm_csv)
    fig, ax = plt.subplots(figsize=(16, 6))

    global_max = max(c["max_nodes"] for c in CONFIGS[dataset])

    for cfg in CONFIGS[dataset]:
        config_name = cfg["name"]
        max_nodes = cfg["max_nodes"]
        x_range = np.arange(2, max_nodes + 1)

        time_data = _compute_mean_iter_time(swarm_df, config_name, x_range)
        if time_data is None:
            continue

        color = SWARM_COLORS[config_name]
        ax.plot(time_data["num_nodes"], time_data["mean_minutes"], "o-",
                color=color, label=cfg["label"], linewidth=2, markersize=4)

    ax.set_xlim(1.5, global_max + 0.5)
    ax.set_xticks(np.arange(2, global_max + 1))
    ax.tick_params(axis='x', labelsize=7)
    ax.set_xlabel("Number of nodes", fontsize=12, fontweight="bold")
    ax.set_ylabel("Time per iteration (min)", fontsize=12, fontweight="bold")
    ax.set_title("Swarm Learning – Time per iteration comparison", fontsize=14, fontweight="bold", pad=10)
    handles, _ = ax.get_legend_handles_labels()
    if handles:
        ax.legend(fontsize=11, loc="best")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    (OUTPUT_DIR / dataset).mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / dataset / "swarm_comparison_plot_time.png"
    fig.savefig(str(out_path), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def main(dataset: str):
    # Per-config plots (own x-range)
    plot_metric("auc", "AUC (mean)", f"own_x_plot_auc.png", dataset, ylim=(0.7, 0.95))
    plot_metric("loss", "Loss (mean)", f"own_x_plot_loss.png", dataset, ylim=(0.15, 0.5))
    plot_time(dataset)

    # Per-config plots (aligned x-range 2..40)
    plot_metric("auc", "AUC (mean)", f"aligned_x_plot_auc.png", dataset, ylim=(0.7, 0.95), shared_x=True)
    plot_metric("loss", "Loss (mean)", f"aligned_x_plot_loss.png", dataset, ylim=(0.15, 0.5), shared_x=True)
    plot_time(dataset, shared_x=True)

    # Swarm comparison (all configs on one graph)
    plot_swarm_comparison("auc", "AUC (mean)", f"swarm_comparison_plot_auc.png", dataset, ylim=(0.7, 0.95))
    plot_swarm_comparison("loss", "Loss (mean)", f"swarm_comparison_plot_loss.png", dataset, ylim=(0.15, 0.5))
    plot_swarm_time_comparison(dataset)

    print("\nDone - all plots created.")


if __name__ == "__main__":
    main("mimic_iv")
    main("mimic_iii")
