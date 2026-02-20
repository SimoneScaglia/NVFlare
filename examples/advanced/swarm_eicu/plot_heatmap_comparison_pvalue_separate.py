import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import matplotlib
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.colors as mcolors
from scipy import stats

os.chdir(Path(__file__).resolve().parent)

def nadeau_bengio_corrected_ttest(scores_a, scores_b, n_train, n_test):
    """
    Esegue il t-test accoppiato corretto di Nadeau & Bengio.
    """
    diff = np.array(scores_a) - np.array(scores_b)
    n = len(diff)

    mu_diff = np.mean(diff)
    sigma2_diff = np.var(diff, ddof=1)
    correction_factor = (1 / n) + (n_test / n_train)
    t_stat = mu_diff / np.sqrt(correction_factor * sigma2_diff)
    p_val = stats.t.sf(np.abs(t_stat), df=n - 1) * 2
    return t_stat, p_val

def parse_lr_from_dir(dir_path):
    """Extract learning rate from directory name"""
    dir_name = os.path.basename(dir_path)
    lr_part = dir_name.split('lr')[-1].split('_')[0]
    lr_str = lr_part.replace('-', '.')
    return float(lr_str)

def parse_bs_from_dir(dir_path):
    """Extract batch size from directory name"""
    dir_name = os.path.basename(dir_path)
    bs_part = dir_name.split('bs')[-1]
    return int(bs_part)

def load_results_for_ttest(directories, csv_filename, learning_rates=None, batch_sizes=None):
    """Load all individual results for t-test"""
    auc_results = {bs: {lr: [] for lr in learning_rates} for bs in batch_sizes}

    for directory in directories:
        csv_path = os.path.join(directory, csv_filename)

        if os.path.exists(csv_path):
            try:
                df = pd.read_csv(csv_path)
                lr = parse_lr_from_dir(directory)
                bs = parse_bs_from_dir(directory)

                if bs in auc_results and lr in auc_results[bs]:
                    auc_results[bs][lr].extend(df.groupby("iteration")["auc"].mean().tolist())

            except Exception as e:
                print(f"Error processing {directory}: {e}")
        else:
            print(f"File not found: {csv_path}")

    return auc_results

def load_mean_results(directories, csv_filename, learning_rates=None, batch_sizes=None):
    """Load mean AUC per (bs, lr) across iterations"""
    auc_results = {bs: {lr: 0 for lr in learning_rates} for bs in batch_sizes}

    for directory in directories:
        csv_path = os.path.join(directory, csv_filename)
        if not os.path.exists(csv_path):
            continue

        try:
            df = pd.read_csv(csv_path)
            lr = parse_lr_from_dir(directory)
            bs = parse_bs_from_dir(directory)
            if bs not in auc_results or lr not in auc_results[bs]:
                continue

            iteration_aucs = []
            for _, iter_df in df.groupby("iteration"):
                if "splits" in iter_df.columns:
                    split_sum = iter_df["splits"].sum()
                    if split_sum == 100:
                        iter_auc = (iter_df["auc"] * iter_df["splits"]).sum() / split_sum
                    else:
                        iter_auc = iter_df["auc"].mean()
                else:
                    iter_auc = iter_df["auc"].mean()
                iteration_aucs.append(iter_auc)

            if iteration_aucs:
                auc_results[bs][lr] = sum(iteration_aucs) / len(iteration_aucs)

        except Exception as e:
            print(f"Error processing {directory}: {e}")

    return auc_results

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

def prepare_matrices(comparison, metric, n_training_samples, n_test_samples):
    learning_rates = [0.0001, 0.00025, 0.0005, 0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.075, 0.1]
    batch_sizes = [8, 16, 32, 64, 128, 256, 512]

    conf1_dirs = [os.path.join(comparison['dir1'], d) for d in build_directories(comparison['dir1'], learning_rates, batch_sizes, comparison['node1'])]
    conf2_dirs = [os.path.join(comparison['dir2'], d) for d in build_directories(comparison['dir2'], learning_rates, batch_sizes, comparison['node2'])]

    conf1_individual = load_results_for_ttest(conf1_dirs, comparison['file1'], learning_rates, batch_sizes)
    conf2_individual = load_results_for_ttest(conf2_dirs, comparison['file2'], learning_rates, batch_sizes)
    conf1_mean = load_mean_results(conf1_dirs, comparison['file1'], learning_rates, batch_sizes)
    conf2_mean = load_mean_results(conf2_dirs, comparison['file2'], learning_rates, batch_sizes)

    batch_sizes = sorted(list(conf1_mean.keys()))
    learning_rates = sorted(list(next(iter(conf1_mean.values())).keys()))

    conf1_values = np.zeros((len(batch_sizes), len(learning_rates)))
    conf2_values = np.zeros((len(batch_sizes), len(learning_rates)))
    delta_values = np.zeros((len(batch_sizes), len(learning_rates)))
    p_values = np.zeros((len(batch_sizes), len(learning_rates)))

    for i, bs in enumerate(batch_sizes):
        for j, lr in enumerate(learning_rates):
            conf1_values[i, j] = conf1_mean[bs][lr]
            conf2_values[i, j] = conf2_mean[bs][lr]
            delta_values[i, j] = conf1_values[i, j] - conf2_values[i, j]

            conf1_vals = conf1_individual[bs][lr]
            conf2_vals = conf2_individual[bs][lr]
            if len(conf1_vals) > 1 and len(conf2_vals) > 1 and len(conf1_vals) == len(conf2_vals):
                try:
                    _, p_val = nadeau_bengio_corrected_ttest(
                        conf1_vals,
                        conf2_vals,
                        n_training_samples,
                        n_test_samples
                    )
                    p_values[i, j] = p_val
                except Exception as e:
                    print(f"Error in t-test for bs={bs}, lr={lr}: {e}")
                    p_values[i, j] = np.nan
            else:
                p_values[i, j] = np.nan

    return {
        "learning_rates": learning_rates,
        "batch_sizes": batch_sizes,
        "conf1_values": conf1_values,
        "conf2_values": conf2_values,
        "delta_values": delta_values,
        "p_values": p_values,
        "metric_title": metric.upper(),
    }

def get_brightness(color):
    return 0.299 * color[0] + 0.587 * color[1] + 0.114 * color[2]

def annotate_heatmap(ax, data, p_values, cmap, vmin, vmax, title, mean_value, learning_rates, batch_sizes, annotate_p=False):
    im = ax.imshow(data, aspect='auto', cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xticks(np.arange(len(learning_rates)))
    ax.set_yticks(np.arange(len(batch_sizes)))
    ax.set_xticklabels([f"{lr:.5f}" for lr in learning_rates], rotation=45, fontsize=12)
    ax.set_yticklabels(batch_sizes, fontsize=12)
    ax.set_xlabel('Learning Rate', fontsize=12)
    ax.set_ylabel('Batch Size', fontsize=12)
    ax.set_title(f"{title}\nmean={mean_value:.4f}", fontsize=18)

    cmap_obj = matplotlib.colormaps.get_cmap(cmap)
    max_val = np.nanmax(data)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            color = cmap_obj((data[i, j] - vmin) / (vmax - vmin)) if vmax > vmin else cmap_obj(0.5)
            text_color = 'white' if get_brightness(color) < 0.5 else 'black'
            is_max = np.isfinite(data[i, j]) and np.isclose(data[i, j], max_val)
            if annotate_p and not np.isnan(p_values[i, j]):
                text = f"{data[i, j]:+.4f}\np={p_values[i, j]:.3f}{'*' if p_values[i, j] < 0.05 else ''}"
            else:
                text = f"{data[i, j]:.4f}"
            ax.text(j, i, text, ha='center', va='center', fontsize=14, color=text_color, fontweight=('bold' if is_max else 'normal'))
    return im

def create_separate_heatmaps(comparison, metric, n_training_samples, n_test_samples):
    matrices = prepare_matrices(comparison, metric, n_training_samples, n_test_samples)
    learning_rates = matrices['learning_rates']
    batch_sizes = matrices['batch_sizes']
    conf1_values = matrices['conf1_values']
    conf2_values = matrices['conf2_values']
    delta_values = matrices['delta_values']
    p_values = matrices['p_values']
    metric_title = matrices['metric_title']

    mean_A = np.mean(conf1_values)
    mean_B = np.mean(conf2_values)
    mean_delta = np.mean(delta_values)

    colors = ["#ffc6ba", "#ffd479", "#ffff66", "#a5db6f", "#7288cd"]
    cmap_A = LinearSegmentedColormap.from_list("cmap_A", colors)
    cmap_B = LinearSegmentedColormap.from_list("cmap_B", colors)

    global_max = max(conf1_values.max(), conf2_values.max())
    vmin_common = global_max - 0.3
    vmin_A, vmax_A = vmin_common, global_max
    vmin_B, vmax_B = vmin_common, global_max

    # delta_bounds = [-1, -0.05, -0.01, 0.01, 0.05, 1]
    # delta_colors = ["#8b0000", "#ff4500", "#ffff66", "#ffff66", "#66ff66", "#006400"]
    # delta_min, delta_max = delta_bounds[0], delta_bounds[-1]
    # delta_positions = [(b - delta_min) / (delta_max - delta_min) for b in delta_bounds]
    # cmap_delta = LinearSegmentedColormap.from_list("cmap_delta", list(zip(delta_positions, delta_colors)))
    # norm_delta = mcolors.Normalize(vmin=delta_min, vmax=delta_max)
    cmap_delta = mcolors.ListedColormap(["#8b0000", "#ff4500", "#ffff66", "#66ff66", "#006400"])
    norm_delta = mcolors.BoundaryNorm([-1, -0.05, -0.01, 0.01, 0.05, 1], cmap_delta.N)

    out_dir = 'plots_results/plot_heatmap_comparison_pvalue_separate'
    os.makedirs(out_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(42, 14))

    im_A = annotate_heatmap(
        axes[0],
        conf1_values,
        p_values,
        cmap_A,
        vmin_A,
        vmax_A,
        f"A ({metric_title})",
        mean_A,
        learning_rates,
        batch_sizes,
        annotate_p=False,
    )
    cbar_A = fig.colorbar(im_A, ax=axes[0], orientation='vertical', shrink=0.8)
    cbar_A.set_label(f'A ({metric_title})', fontsize=12)

    im_B = annotate_heatmap(
        axes[1],
        conf2_values,
        p_values,
        cmap_B,
        vmin_B,
        vmax_B,
        f"B ({metric_title})",
        mean_B,
        learning_rates,
        batch_sizes,
        annotate_p=False,
    )
    cbar_B = fig.colorbar(im_B, ax=axes[1], orientation='vertical', shrink=0.8)
    cbar_B.set_label(f'B ({metric_title})', fontsize=12)

    im_delta = axes[2].imshow(delta_values, aspect='auto', cmap=cmap_delta, norm=norm_delta)
    axes[2].set_xticks(np.arange(len(learning_rates)))
    axes[2].set_yticks(np.arange(len(batch_sizes)))
    axes[2].set_xticklabels([f"{lr:.5f}" for lr in learning_rates], rotation=45, fontsize=12)
    axes[2].set_yticklabels(batch_sizes, fontsize=12)
    axes[2].set_xlabel('Learning Rate', fontsize=12)
    axes[2].set_ylabel('Batch Size', fontsize=12)
    axes[2].set_title(f"Δ (A - B) ({metric_title})\nmean={mean_delta:.4f}", fontsize=18)

    delta_max = np.nanmax(delta_values)
    for i in range(delta_values.shape[0]):
        for j in range(delta_values.shape[1]):
            color = im_delta.cmap(im_delta.norm(delta_values[i, j]))
            text_color = 'white' if get_brightness(color) < 0.5 else 'black'
            is_max = np.isfinite(delta_values[i, j]) and np.isclose(delta_values[i, j], delta_max)
            if not np.isnan(p_values[i, j]):
                text = f"{delta_values[i, j]:+.4f}\np={p_values[i, j]:.3f}{'*' if p_values[i, j] < 0.05 else ''}"
            else:
                text = f"{delta_values[i, j]:+.4f}\np=N/A"
            axes[2].text(j, i, text, ha='center', va='center', fontsize=14, color=text_color, fontweight=('bold' if is_max else 'normal'))

    cbar_delta = fig.colorbar(im_delta, ax=axes[2], orientation='vertical', shrink=0.8)
    cbar_delta.set_label(f'Δ ({metric_title}: A - B)', fontsize=12)

    exp = comparison['exp']
    fig.suptitle(
        f'{metric_title} Separate Heatmaps: {exp}\n'
        f'Means — A: {mean_A:.4f}, B: {mean_B:.4f}, Δ: {mean_delta:.4f}\n'
        f'Corrected Dependent t-test (train={n_training_samples}, test={n_test_samples})',
        fontsize=20,
        y=0.98,
    )

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    exp_replaced = exp.replace(' ', '_').replace('=', 'vs').replace('(', '').replace(')', '').replace(',', '').replace('/', '_')
    filename = f"{out_dir}/{exp_replaced}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

    # Log highest average combination for reference
    avg_values = (conf1_values + conf2_values) / 2
    max_avg_indices = np.where(avg_values == np.max(avg_values))
    if max_avg_indices[0].size > 0 and max_avg_indices[1].size > 0:
        i, j = max_avg_indices[0][0], max_avg_indices[1][0]
        max_avg_lr = learning_rates[j]
        max_avg_bs = batch_sizes[i]
        max_avg_value = avg_values[i, j]
        print(f"Highest average value: {max_avg_value:.4f} at LR={max_avg_lr:.5f}, BS={max_avg_bs}")

def try_plot_separate_comparison(comparison):
    try:
        print(f"Plotting separate heatmaps for: {comparison['exp']}")
        create_separate_heatmaps(
            comparison,
            metric='auc',
            n_training_samples=comparison['n_training_samples'],
            n_test_samples=comparison['n_test_samples']
        )
    except Exception as e:
        print(f"An error occurred while plotting separate heatmaps for {comparison['exp']}: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    comparisons = [
        # {
        #     'exp': 'A = Swarm 5 nodes (~3000 rows/node) vs B = Central (~18000 rows) 70k var rows',
        #     'dir1': 'results/70k_var_rows_separate_testset',
        #     'file1': 'swarm_results.csv',
        #     'node1': 5,
        #     'dir2': 'results/70k_var_rows_separate_testset',
        #     'file2': 'central_results.csv',
        #     'node2': 5,
        #     'n_training_samples': 3000,
        #     'n_test_samples': 1000
        # },
        {
            'exp': 'A = Swarm 5 nodes (~3000 rows/node) vs B = Central (~18000 rows) 70k var rows entire test set',
            'dir1': 'results/70k_var_rows_separate_testset',
            'file1': 'swarm_results_entire_testset.csv',
            'node1': 5,
            'dir2': 'results/70k_var_rows_separate_testset',
            'file2': 'central_results.csv',
            'node2': 5,
            'n_training_samples': 3000,
            'n_test_samples': 5000
        },
        # {
        #     'exp': 'A = Swarm 5 nodes (2500 rows/node) vs B = Central (12500 rows) 70k fixed rows',
        #     'dir1': 'results/heatmap_experiments_5nodes_fixed_rows',
        #     'file1': 'swarm_results.csv',
        #     'node1': 5,
        #     'dir2': 'results/heatmap_experiments_5nodes_fixed_rows',
        #     'file2': 'central_results.csv',
        #     'node2': 5,
        #     'n_training_samples': 2500,
        #     'n_test_samples': 1000
        # },
        {
            'exp': 'A = Swarm 5 nodes (2500 rows/node) vs B = Central (12500 rows) 70k fixed rows entire test set',
            'dir1': 'results/70k_fixed_rows_separate_entire_testset',
            'file1': 'swarm_results_entire_testset.csv',
            'node1': 5,
            'dir2': 'results/70k_fixed_rows_separate_entire_testset',
            'file2': 'central_results.csv',
            'node2': 5,
            'n_training_samples': 2500,
            'n_test_samples': 1000
        },
        # {
        #     'exp': 'A = Swarm 5 nodes (7155 rows/node) vs B = Central (35775 rows) mimic iv entire test set',
        #     'dir1': 'results/mimiciv_total_entire_testset',
        #     'file1': 'swarm_results_entire_testset.csv',
        #     'node1': 5,
        #     'dir2': 'results/mimiciv_total_entire_testset',
        #     'file2': 'central_results.csv',
        #     'node2': 5,
        #     'n_training_samples': 7155,
        #     'n_test_samples': 3066
        # },
        {
            'exp': 'A = Swarm 5 nodes (2500 rows/node) vs B = Central (12500 rows) mimic iv fixed 12500',
            'dir1': 'results/mimiciv_fixed_12500_entire_testset',
            'file1': 'swarm_results_entire_testset.csv',
            'node1': 5,
            'dir2': 'results/mimiciv_fixed_12500_entire_testset',
            'file2': 'central_results.csv',
            'node2': 5,
            'n_training_samples': 2500,
            'n_test_samples': 1071
        },
        {
            'exp': 'A = Swarm 5 nodes (2459 rows/node) vs B = Central (12295 rows) mimic iii entire test set',
            'dir1': 'results/mimiciii_total_entire_testset',
            'file1': 'swarm_results_entire_testset.csv',
            'node1': 5,
            'dir2': 'results/mimiciii_total_entire_testset',
            'file2': 'central_results.csv',
            'node2': 5,
            'n_training_samples': 2459,
            'n_test_samples': 1054
        },
    ]

    for comp in comparisons:
        try_plot_separate_comparison(comp)
