#!/usr/bin/env python3
"""
Generate comparison plots for local, central and swarm experiments.

Supports multiple datasets: mimic_iii, mimic_iv, mimic_iv_fixed, eicu.

Creates images in plots_results/<dataset>/:
- mosaic_plot_auc.png                  (compact mosaic layout)
- swarm_comparison_nodes_plot_auc.png  (swarm lines by num_nodes)
- swarm_comparison_rows_plot_auc.png   (swarm lines by total rows)

Each per-configuration subplot shows three lines: Local, Central, Swarm
(mean across nodes/sites).
"""

import os
import re
from pathlib import Path

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import numpy as np
import pandas as pd

# ── Configuration ──────────────────────────────────────────────────────
CONFIGS = {
    "mimic_iv": [
        {"name": "5nodes",  "max_nodes": 5,  "label": "5 nodes - 8000 rows/node"},
        {"name": "10nodes", "max_nodes": 10, "label": "10 nodes - 4000 rows/node"},
        {"name": "20nodes", "max_nodes": 20, "label": "20 nodes - 2000 rows/node"},
        {"name": "40nodes", "max_nodes": 40, "label": "40 nodes - 1000 rows/node"},
        {"name": "80nodes", "max_nodes": 80, "label": "80 nodes - 500 rows/node"},
    ],
    "mimic_iii": [
        {"name": "5nodes",  "max_nodes": 5,  "label": "5 nodes - 2500 rows/node"},
        {"name": "10nodes", "max_nodes": 10, "label": "10 nodes - 1250 rows/node"},
        {"name": "20nodes", "max_nodes": 20, "label": "20 nodes - 625 rows/node"},
        {"name": "40nodes", "max_nodes": 40, "label": "40 nodes - 312 rows/node"},
        {"name": "80nodes", "max_nodes": 80, "label": "80 nodes - 156 rows/node"},
    ],
    "mimic_iv_fixed": [
        {"name": "5nodes",  "max_nodes": 5,  "label": "5 nodes - 2500 rows/node"},
        {"name": "10nodes", "max_nodes": 10, "label": "10 nodes - 1250 rows/node"},
        {"name": "20nodes", "max_nodes": 20, "label": "20 nodes - 625 rows/node"},
        {"name": "40nodes", "max_nodes": 40, "label": "40 nodes - 312 rows/node"},
        {"name": "80nodes", "max_nodes": 80, "label": "80 nodes - 156 rows/node"},
    ],
    "eicu": [
        {"name": "5nodes",  "max_nodes": 5,  "label": "5 nodes - 2500 rows/node"},
        {"name": "10nodes", "max_nodes": 10, "label": "10 nodes - 1250 rows/node"},
        {"name": "20nodes", "max_nodes": 20, "label": "20 nodes - 625 rows/node"},
        {"name": "25nodes", "max_nodes": 25, "label": "25 nodes - 500 rows/node"},
    ],
}

DATASET_NAMES = {
    "mimic_iii": "MIMIC-III",
    "mimic_iv_fixed": "MIMIC-IV",
    "mimic_iv": "MIMIC-IV Full",
    "eicu": "eICU",
}

SWARM_COLORS = {
    "5nodes":  "#f39c12",
    "10nodes": "#e74c3c",
    "20nodes": "#2ecc71",
    "25nodes": "#3498db",
    "40nodes": "#3498db",
    "80nodes": "#9b59b6",
}

AUC_YLIM = {
    "eicu": (0.6, 0.85),
    "mimic_iii": (0.85, 0.89),
    "mimic_iv": (0.85, 0.89),
    "mimic_iv_fixed": (0.85, 0.89),
}

AUC_YLIM_COMPACT = {
    "eicu": (0.65, 0.85),
    "mimic_iii": (0.85, 0.89),
    "mimic_iv": (0.85, 0.89),
    "mimic_iv_fixed": (0.85, 0.89),
}

SCRIPT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = SCRIPT_DIR / "results"
OUTPUT_DIR = SCRIPT_DIR / "plots_results"

GROUP_SIZE = 5


# ── Helpers ────────────────────────────────────────────────────────────

def _rows_per_node_from_label(label: str) -> int:
    """Extract rows/node integer from labels like '5 nodes - 8000 rows/node'."""
    m = re.search(r"(\d+)\s*rows/node", label)
    if m is None:
        raise ValueError(f"Cannot parse rows/node from label: {label!r}")
    return int(m.group(1))


def _drop_zero_auc_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows with auc == 0 when auc column exists."""
    if df.empty or "auc" not in df.columns:
        return df
    return df[df["auc"] != 0].copy()


def _load_results(dataset: str):
    """Load local/central/swarm CSVs for a dataset, dropping zero-AUC rows."""
    dfs = {}
    for kind in ("local", "central", "swarm"):
        path = RESULTS_DIR / dataset / f"{kind}_results.csv"
        df = pd.read_csv(path) if path.is_file() else pd.DataFrame()
        dfs[kind] = _drop_zero_auc_rows(df)
    return dfs["local"], dfs["central"], dfs["swarm"]


def _agg_metric(df, config_name, x_range, metric):
    """Aggregate metric by num_nodes for a given configuration."""
    if df.empty or config_name not in df["configuration"].values:
        return None
    sub = df[df["configuration"] == config_name]
    agg = sub.groupby("num_nodes")[metric].agg(["mean", "std"]).reset_index()
    agg = agg.set_index("num_nodes").reindex(x_range).reset_index()
    return agg


def _plot_three_series(ax, local_df, central_df, swarm_df, config_name, x_range, metric, linewidth=2, markersize=4):
    """Plot Local / Central / Swarm series on a single axis."""
    series = [
        (local_df,   "o-", "#e74c3c", "Local"),
        (central_df, "s-", "#2ecc71", "Central"),
        (swarm_df,   "^-", "#3498db", "Swarm"),
    ]
    for df, marker, color, label in series:
        agg = _agg_metric(df, config_name, x_range, metric)
        if agg is not None:
            ax.plot(agg["num_nodes"], agg["mean"], marker, color=color, label=label, linewidth=linewidth, markersize=markersize)


# ── Time computation ───────────────────────────────────────────────────

def _compute_mean_iter_time(df: pd.DataFrame, config_name: str, x_range):
    """Compute per-num_nodes mean time (min) between consecutive iterations."""
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
        iter_times = nn_df.groupby("iteration")["datetime"].max().sort_index()
        if len(iter_times) < 2:
            records.append({"num_nodes": nn, "mean_minutes": np.nan})
            continue
        diffs = iter_times.diff().dropna().abs().dt.total_seconds() / 60.0
        records.append({"num_nodes": nn, "mean_minutes": diffs.mean()})
    return pd.DataFrame(records)


# ── Hyperparameter annotations ────────────────────────────────────────

def _get_params_from_results(df, config_name, num_nodes):
    """Extract (lr, batch_size) for a (config, num_nodes) from results CSV."""
    if df.empty or "lr" not in df.columns or "batch_size" not in df.columns:
        return None
    sub = df[(df["configuration"] == config_name) & (df["num_nodes"] == num_nodes)]
    if sub.empty:
        return None
    row = sub.iloc[0]
    return row["lr"], int(row["batch_size"])


def _annotate_hyperparams(ax, dataset, config_name, max_nodes, local_df, central_df, swarm_df):
    """Add hyperparameter annotations and group separators to a subplot."""
    trans = ax.get_xaxis_transform()

    # Local params
    local_params = None
    for nn in range(2, max_nodes + 1):
        local_params = _get_params_from_results(local_df, config_name, nn)
        if local_params:
            break
    if local_params:
        lr, bs = local_params
        ax.text(
            0.02, 0.04, f"Local: lr={lr:g}  bs={bs}",
            transform=ax.transAxes, fontsize=10, va="bottom", ha="left",
            color="#e74c3c", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#e74c3c", alpha=0.85, linewidth=0.6),
        )

    # Per-group annotations
    for group_end in range(GROUP_SIZE, max_nodes + 1, GROUP_SIZE):
        group_start = max(2, group_end - GROUP_SIZE + 1)
        if (group_end // GROUP_SIZE) % 2 == 0:
            ax.axvspan(group_start - 0.5, group_end + 0.5, alpha=0.04, color="steelblue", zorder=0)
        if group_end < max_nodes:
            ax.axvline(x=group_end + 0.5, color="gray", linestyle=":", alpha=0.4, linewidth=0.8)

        swarm_params = central_params = None
        for nn in range(group_start, group_end + 1):
            if swarm_params is None:
                swarm_params = _get_params_from_results(swarm_df, config_name, nn)
            if central_params is None:
                central_params = _get_params_from_results(central_df, config_name, nn)

        x_center = (group_start + group_end) / 2
        parts = []
        if swarm_params:
            lr, bs = swarm_params
            parts.append((f"S: lr={lr:g} bs={bs}", "#3498db"))
        if central_params:
            lr, bs = central_params
            parts.append((f"C: lr={lr:g} bs={bs}", "#2ecc71"))

        for i, (text, color) in enumerate(parts):
            y_pos = 0.96 - i * 0.09
            ax.text(
                x_center, y_pos, text, transform=trans,
                fontsize=10, ha="center", va="top",
                color=color, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor=color, alpha=0.8, linewidth=0.5),
            )


# ── Plot functions ─────────────────────────────────────────────────────

def plot_metric(metric, ylabel, output_filename, dataset, ylim=None, shared_x=False):
    """Vertically-stacked subplots, one per configuration."""
    local_df, central_df, swarm_df = _load_results(dataset)
    cfgs = CONFIGS[dataset]
    dataset_name = DATASET_NAMES.get(dataset, dataset.upper())
    global_max = max(c["max_nodes"] for c in cfgs)
    fig, axes = plt.subplots(len(cfgs), 1, figsize=(16, 4 * len(cfgs)))
    if len(cfgs) == 1:
        axes = [axes]
    fig.suptitle(f"{dataset_name} - {ylabel}", fontsize=16, fontweight="bold", y=0.995)

    for idx, cfg in enumerate(cfgs):
        ax = axes[idx]
        config_name = cfg["name"]
        max_nodes = cfg["max_nodes"]
        x_range = np.arange(2, max_nodes + 1)
        x_axis_max = global_max if shared_x else max_nodes

        _plot_three_series(ax, local_df, central_df, swarm_df, config_name, x_range, metric)

        ax.set_xlim(1.5, x_axis_max + 0.5)
        ax.set_xticks(np.arange(2, (global_max if shared_x else max_nodes) + 1))
        ax.tick_params(axis="x", labelsize=7)
        ax.set_xlabel("Number of nodes", fontsize=12, fontweight="bold")
        ax.set_ylabel(ylabel, fontsize=12, fontweight="bold")
        ax.set_title(cfg["label"], fontsize=13, fontweight="bold", pad=8)
        if ylim:
            ax.set_ylim(ylim)
        handles, _ = ax.get_legend_handles_labels()
        if handles:
            ax.legend(fontsize=11, loc="best")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    _save(fig, dataset, output_filename)


def plot_metric_mosaic(metric, ylabel, output_filename, dataset, ylim=None):
    """Compact mosaic layout adapting to the number of configurations.

    5 configs → top: 3 plots (2/6 cols each), bottom: 2 plots (3/6 cols each)
    4 configs → top: 3 plots (2/6 cols each), bottom: 1 plot (full width)
    3 configs → single row of 3 plots
    """
    local_df, central_df, swarm_df = _load_results(dataset)
    cfgs = CONFIGS[dataset]
    dataset_name = DATASET_NAMES.get(dataset, dataset.upper())
    n = len(cfgs)

    fig = plt.figure(figsize=(16, 6.5))
    # Reserve space at the top for the shared legend and title
    gs = gridspec.GridSpec(2, 6, hspace=0.30, wspace=0.45, top=0.86, bottom=0.09, left=0.06, right=0.98)
    fig.suptitle(f"{dataset_name} - {ylabel}", fontsize=14, fontweight="bold", y=0.96)

    if n >= 5:
        positions = [gs[0, 0:2], gs[0, 2:4], gs[0, 4:6], gs[1, 0:3], gs[1, 3:6]]
    elif n == 4:
        positions = [gs[0, 0:2], gs[0, 2:4], gs[0, 4:6], gs[1, 0:6]]
    elif n == 3:
        positions = [gs[0, 0:2], gs[0, 2:4], gs[0, 4:6]]
    else:
        positions = [gs[0, i * 3:(i + 1) * 3] for i in range(n)]

    # Auto-compute shared Y range from actual data across all configs
    if ylim is None:
        all_means = []
        for cfg in cfgs:
            x_range = np.arange(2, cfg["max_nodes"] + 1)
            for df in (local_df, central_df, swarm_df):
                agg = _agg_metric(df, cfg["name"], x_range, metric)
                if agg is not None:
                    valid = agg["mean"].dropna()
                    if not valid.empty:
                        all_means.extend(valid.tolist())
        if all_means:
            margin = (max(all_means) - min(all_means)) * 0.05
            ylim = (
                np.floor((min(all_means) - margin) * 100) / 100,
                np.ceil((max(all_means) + margin) * 100) / 100,
            )

    first_ax = None
    for idx, cfg in enumerate(cfgs):
        if idx >= len(positions):
            break
        ax = fig.add_subplot(positions[idx])
        if first_ax is None:
            first_ax = ax
        config_name = cfg["name"]
        max_nodes = cfg["max_nodes"]
        x_range = np.arange(2, max_nodes + 1)

        _plot_three_series(ax, local_df, central_df, swarm_df, config_name, x_range, metric, linewidth=1.5, markersize=3)

        ax.set_xlim(1.5, max_nodes + 0.5)
        step = max(1, max_nodes // 10)
        xticks = np.arange(2, max_nodes + 1, step)
        if xticks[-1] != max_nodes:
            xticks = np.append(xticks, max_nodes)
        ax.set_xticks(xticks)
        ax.tick_params(axis="x", labelsize=6)
        ax.tick_params(axis="y", labelsize=7)
        ax.set_xlabel("Num nodes", fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9, labelpad=2)
        ax.set_title(cfg["label"], fontsize=10, fontweight="bold", pad=4)
        if ylim:
            ax.set_ylim(ylim)
            ymin, ymax = ylim
            yrange = ymax - ymin
            if yrange <= 0.1:
                ystep = 0.01
            elif yrange <= 0.2:
                ystep = 0.02
            else:
                ystep = 0.05
            ax.set_yticks(np.arange(ymin, ymax + ystep / 2, ystep))
        ax.grid(True, alpha=0.3)

    # Shared horizontal legend at the top of the figure
    if first_ax is not None:
        handles, labels = first_ax.get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, loc="upper center", ncol=len(handles), fontsize=11, frameon=False, bbox_to_anchor=(0.5, 0.93))

    _save(fig, dataset, output_filename)


def plot_metric_rows(metric, ylabel, output_filename, dataset, ylim=None):
    """Vertically-stacked subplots with x-axis = total rows."""
    local_df, central_df, swarm_df = _load_results(dataset)
    cfgs = CONFIGS[dataset]
    dataset_name = DATASET_NAMES.get(dataset, dataset.upper())
    global_max_rows = max(c["max_nodes"] * _rows_per_node_from_label(c["label"]) for c in cfgs)
    fig, axes = plt.subplots(len(cfgs), 1, figsize=(16, 4 * len(cfgs)))
    if len(cfgs) == 1:
        axes = [axes]
    fig.suptitle(f"{dataset_name} - {ylabel}", fontsize=16, fontweight="bold", y=0.995)

    for idx, cfg in enumerate(cfgs):
        ax = axes[idx]
        config_name = cfg["name"]
        max_nodes = cfg["max_nodes"]
        x_range = np.arange(2, max_nodes + 1)
        rows_per_node = _rows_per_node_from_label(cfg["label"])

        for df, marker, color, label in [
            (local_df,   "o-", "#e74c3c", "Local"),
            (central_df, "s-", "#2ecc71", "Central"),
            (swarm_df,   "^-", "#3498db", "Swarm"),
        ]:
            agg = _agg_metric(df, config_name, x_range, metric)
            if agg is not None:
                agg["total_rows"] = agg["num_nodes"] * rows_per_node
                ax.plot(agg["total_rows"], agg["mean"], marker, color=color, label=label, linewidth=2, markersize=4)

        ax.set_xlim(0, global_max_rows * 1.02)
        ax.tick_params(axis="x", labelsize=7)
        ax.set_xlabel("Total number of rows", fontsize=12, fontweight="bold")
        ax.set_ylabel(ylabel, fontsize=12, fontweight="bold")
        ax.set_title(cfg["label"], fontsize=13, fontweight="bold", pad=8)
        if ylim:
            ax.set_ylim(ylim)
        handles, _ = ax.get_legend_handles_labels()
        if handles:
            ax.legend(fontsize=11, loc="best")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    _save(fig, dataset, output_filename)


def plot_time(dataset, shared_x=False):
    """Vertically-stacked subplots for mean iteration time (min)."""
    local_df, central_df, swarm_df = _load_results(dataset)
    cfgs = CONFIGS[dataset]
    dataset_name = DATASET_NAMES.get(dataset, dataset.upper())
    global_max = max(c["max_nodes"] for c in cfgs)
    fig, axes = plt.subplots(len(cfgs), 1, figsize=(16, 4 * len(cfgs)))
    if len(cfgs) == 1:
        axes = [axes]
    fig.suptitle(f"{dataset_name} - Time per iteration", fontsize=16, fontweight="bold", y=0.995)

    for idx, cfg in enumerate(cfgs):
        ax = axes[idx]
        config_name = cfg["name"]
        max_nodes = cfg["max_nodes"]
        x_range = np.arange(2, max_nodes + 1)
        x_axis_max = global_max if shared_x else max_nodes

        for df, marker, color, label in [
            (local_df,   "o-", "#e74c3c", "Local"),
            (central_df, "s-", "#2ecc71", "Central"),
            (swarm_df,   "^-", "#3498db", "Swarm"),
        ]:
            time_data = _compute_mean_iter_time(df, config_name, x_range)
            if time_data is not None:
                ax.plot(time_data["num_nodes"], time_data["mean_minutes"], marker, color=color, label=label, linewidth=2, markersize=4)

        ax.set_xlim(1.5, x_axis_max + 0.5)
        ax.set_xticks(np.arange(2, (global_max if shared_x else max_nodes) + 1))
        ax.tick_params(axis="x", labelsize=7)
        ax.set_xlabel("Number of nodes", fontsize=12, fontweight="bold")
        ax.set_ylabel("Time per iteration (min)", fontsize=12, fontweight="bold")
        ax.set_title(cfg["label"], fontsize=13, fontweight="bold", pad=8)
        ax.set_ylim(0, 15)
        handles, _ = ax.get_legend_handles_labels()
        if handles:
            ax.legend(fontsize=11, loc="best")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    prefix = "aligned_x_" if shared_x else "own_x_"
    _save(fig, dataset, f"{prefix}plot_time.png")


def plot_swarm_comparison_nodes(metric, ylabel, output_filename, dataset, ylim=None):
    """Single graph with one coloured line per configuration (swarm only)."""
    swarm_csv = RESULTS_DIR / dataset / "swarm_results.csv"
    if not swarm_csv.is_file():
        print(f"WARNING: {swarm_csv} not found - skipping")
        return
    swarm_df = _drop_zero_auc_rows(pd.read_csv(swarm_csv))
    dataset_name = DATASET_NAMES.get(dataset, dataset.upper())
    fig, ax = plt.subplots(figsize=(16, 4))
    cfgs = CONFIGS[dataset]
    global_max = max(c["max_nodes"] for c in cfgs)

    for cfg in cfgs:
        config_name = cfg["name"]
        x_range = np.arange(2, cfg["max_nodes"] + 1)
        agg = _agg_metric(swarm_df, config_name, x_range, metric)
        if agg is None:
            continue
        color = SWARM_COLORS.get(config_name, "#333333")
        ax.plot(agg["num_nodes"], agg["mean"], "o-", color=color, label=cfg["label"], linewidth=2, markersize=4)

    ax.set_xlim(1.5, global_max + 0.5)
    ax.set_xticks(np.arange(2, global_max + 1))
    ax.tick_params(axis="x", labelsize=7)
    ax.set_xlabel("Number of nodes", fontsize=12, fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=12, fontweight="bold")
    ax.set_title(f"{dataset_name} - Swarm Learning - {ylabel} comparison", fontsize=14, fontweight="bold", pad=10)
    ax.legend(fontsize=11, loc="best")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    _save(fig, dataset, output_filename)


def plot_swarm_comparison_rows(metric, ylabel, output_filename, dataset, ylim=None):
    """Single graph with one coloured line per configuration (swarm only),
    x-axis = total rows (num_nodes × rows_per_node)."""
    swarm_csv = RESULTS_DIR / dataset / "swarm_results.csv"
    if not swarm_csv.is_file():
        print(f"WARNING: {swarm_csv} not found - skipping")
        return
    swarm_df = _drop_zero_auc_rows(pd.read_csv(swarm_csv))
    dataset_name = DATASET_NAMES.get(dataset, dataset.upper())
    fig, ax = plt.subplots(figsize=(16, 4))
    max_total_rows = 0

    for cfg in CONFIGS[dataset]:
        config_name = cfg["name"]
        max_nodes = cfg["max_nodes"]
        x_range = np.arange(2, max_nodes + 1)
        agg = _agg_metric(swarm_df, config_name, x_range, metric)
        if agg is None:
            continue
        rows_per_node = _rows_per_node_from_label(cfg["label"])
        agg["total_rows"] = agg["num_nodes"] * rows_per_node
        color = SWARM_COLORS.get(config_name, "#333333")
        ax.plot(agg["total_rows"], agg["mean"], "o-", color=color, label="Up to " + cfg["label"], linewidth=2, markersize=4)
        current_max = rows_per_node * max_nodes
        if current_max > max_total_rows:
            max_total_rows = current_max

    ax.set_xlim(-max_total_rows * 0.01, max_total_rows * 1.01)
    ax.xaxis.set_major_locator(MultipleLocator(2000))
    ax.tick_params(axis="x", labelsize=7)
    ax.set_xlabel("Total number of rows", fontsize=12, fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=12, fontweight="bold")
    ax.set_title(f"{dataset_name} - Swarm Learning - {ylabel} comparison by total rows", fontsize=14, fontweight="bold", pad=10)
    ax.legend(fontsize=11, loc="lower right", ncol=2)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    _save(fig, dataset, output_filename)


def plot_swarm_time_comparison(dataset):
    """Single graph comparing swarm iteration time across all configs."""
    swarm_csv = RESULTS_DIR / dataset / "swarm_results.csv"
    if not swarm_csv.is_file():
        print(f"WARNING: {swarm_csv} not found - skipping")
        return
    swarm_df = _drop_zero_auc_rows(pd.read_csv(swarm_csv))
    dataset_name = DATASET_NAMES.get(dataset, dataset.upper())
    fig, ax = plt.subplots(figsize=(16, 6))
    cfgs = CONFIGS[dataset]
    global_max = max(c["max_nodes"] for c in cfgs)

    for cfg in cfgs:
        config_name = cfg["name"]
        x_range = np.arange(2, cfg["max_nodes"] + 1)
        time_data = _compute_mean_iter_time(swarm_df, config_name, x_range)
        if time_data is None:
            continue
        color = SWARM_COLORS.get(config_name, "#333333")
        ax.plot(time_data["num_nodes"], time_data["mean_minutes"], "o-", color=color, label=cfg["label"], linewidth=2, markersize=4)

    ax.set_xlim(1.5, global_max + 0.5)
    ax.set_xticks(np.arange(2, global_max + 1))
    ax.tick_params(axis="x", labelsize=7)
    ax.set_xlabel("Number of nodes", fontsize=12, fontweight="bold")
    ax.set_ylabel("Time per iteration (min)", fontsize=12, fontweight="bold")
    ax.set_title(f"{dataset_name} - Swarm Learning - Time per iteration comparison", fontsize=14, fontweight="bold", pad=10)
    handles, _ = ax.get_legend_handles_labels()
    if handles:
        ax.legend(fontsize=11, loc="best")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    _save(fig, dataset, "swarm_comparison_plot_time.png")


# ── Save helper ────────────────────────────────────────────────────────

def _save(fig, dataset, filename):
    (OUTPUT_DIR / dataset).mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / dataset / filename
    fig.savefig(str(out_path), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


# ── Main ───────────────────────────────────────────────────────────────

def main(dataset: str):
    plot_metric_mosaic("auc", "AUC (mean)", "mosaic_plot_auc.png", dataset)
    plot_swarm_comparison_nodes("auc", "AUC (mean)", "swarm_comparison_nodes_plot_auc.png", dataset)
    plot_swarm_comparison_rows("auc", "AUC (mean)", "swarm_comparison_rows_plot_auc.png", dataset)

    print(f"\nDone - all plots created for {dataset}.")


if __name__ == "__main__":
    for ds in CONFIGS:
        main(ds)
