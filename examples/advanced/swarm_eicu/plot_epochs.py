import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import StringIO
import os

def read_csv_clean(path):
    if not os.path.exists(path):
        return pd.DataFrame()
    with open(path, 'r', encoding='utf-8') as f:
        lines = [ln for ln in f.readlines() if not ln.lstrip().startswith('//')]
    return pd.read_csv(StringIO(''.join(lines)))

def prepare_series(df, epoch_col='epochs', metric_col='auc'):
    df = df.copy()
    df[epoch_col] = df[epoch_col].astype(int)
    # average if multiple rows per epoch/iteration
    ser = df.groupby([epoch_col, 'iteration'])[metric_col].mean().unstack(fill_value=np.nan)
    return ser

def compute_means(central_df, swarm_df, max_epoch, metric_col):
    x = np.arange(1, max_epoch + 1)

    if not central_df.empty:
        central_ser = prepare_series(central_df, epoch_col='epoch', metric_col=metric_col).reindex(x)
        central_mean = central_ser.mean(axis=1, skipna=True)
    else:
        central_mean = pd.Series(index=x, dtype=float)

    if not swarm_df.empty:
        nodes = []
        node_lines = {}
        for node in sorted(swarm_df['user'].unique()):
            node_df = swarm_df[swarm_df['user'] == node]
            ser = prepare_series(node_df, epoch_col='epoch', metric_col=metric_col).reindex(x)
            node_lines[node] = ser
            nodes.append(node)
        node_df_concat = pd.concat([node_lines[n] for n in nodes], axis=1)
        swarm_mean = node_df_concat.mean(axis=1, skipna=True)
    else:
        nodes = []
        node_lines = {}
        swarm_mean = pd.Series(index=x, dtype=float)

    if metric_col == 'auc':
        best_central_value = central_mean.max()
        best_central_epoch = central_mean.idxmax()
        best_swarm_value = swarm_mean.max()
        best_swarm_epoch = swarm_mean.idxmax()
    else:
        best_central_value = central_mean.min()
        best_central_epoch = central_mean.idxmin()
        best_swarm_value = swarm_mean.min()
        best_swarm_epoch = swarm_mean.idxmin()

    return x, central_mean, swarm_mean, best_central_epoch, best_central_value, best_swarm_epoch, best_swarm_value


def plot_auc_loss_stacked(central_path, swarm_path, max_epoch, auc_ylim, loss_ylim, fig_name, title_suffix):
    central_df = read_csv_clean(central_path)
    swarm_df = read_csv_clean(swarm_path)

    x_auc, central_auc, swarm_auc, best_c_ep_auc, best_c_val_auc, best_s_ep_auc, best_s_val_auc = compute_means(
        central_df, swarm_df, max_epoch, metric_col='auc'
    )
    x_loss, central_loss, swarm_loss, best_c_ep_loss, best_c_val_loss, best_s_ep_loss, best_s_val_loss = compute_means(
        central_df, swarm_df, max_epoch, metric_col='loss'
    )

    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

    # AUC subplot
    ax = axes[0]
    ax.plot(x_auc, central_auc.values, label='central_mean', color='black', linewidth=2)
    ax.plot(x_auc, swarm_auc.values, label='swarm_mean', color='magenta', linewidth=2)
    ax.scatter([best_c_ep_auc], [best_c_val_auc], color='blue', label=f'Best Central: {best_c_val_auc:.4f} (Epoch {best_c_ep_auc})')
    ax.scatter([best_s_ep_auc], [best_s_val_auc], color='red', label=f'Best Swarm: {best_s_val_auc:.4f} (Epoch {best_s_ep_auc})')
    ax.set_xlim(0, max_epoch + 1)
    ax.set_ylim(*auc_ylim)
    ax.set_xlabel('epochs')
    ax.set_ylabel('auc')
    ax.set_title(f'AUC per epoch (mean over 10 iterations)\n{title_suffix}')
    ax.grid(True, linestyle=':', alpha=0.5)
    ax.legend(loc='best', fontsize='small')

    # Loss subplot
    ax = axes[1]
    ax.plot(x_loss, central_loss.values, label='central_mean', color='black', linewidth=2)
    ax.plot(x_loss, swarm_loss.values, label='swarm_mean', color='magenta', linewidth=2)
    ax.scatter([best_c_ep_loss], [best_c_val_loss], color='blue', label=f'Best Central: {best_c_val_loss:.4f} (Epoch {best_c_ep_loss})')
    ax.scatter([best_s_ep_loss], [best_s_val_loss], color='red', label=f'Best Swarm: {best_s_val_loss:.4f} (Epoch {best_s_ep_loss})')
    ax.set_xlim(0, max_epoch + 1)
    ax.set_ylim(*loss_ylim)
    ax.set_xlabel('epochs')
    ax.set_ylabel('loss')
    ax.set_title(f'Loss per epoch (mean over 10 iterations)\n{title_suffix}')
    ax.grid(True, linestyle=':', alpha=0.5)
    ax.legend(loc='lower right', fontsize='small')

    plt.tight_layout()
    out_dir = 'plots_results/plot_epochs'
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(out_dir + '/' + fig_name, dpi=350)

if __name__ == '__main__':
    # # # Fixed rows experiment: combined AUC + Loss
    # plot_auc_loss_stacked(
    #     'results/70k_fixed_rows_separate_testset_epochs/5_0_lr0.00025_bs8/central_results.csv',
    #     'results/70k_fixed_rows_separate_testset_epochs/5_0_lr0.00500_bs8/swarm_results.csv',
    #     250,
    #     (0.5, 0.9),
    #     (0.15, 0.5),
    #     'plot_epochs_5nodes_70k_fixed_rows_auc_loss_separate.png',
    #     'Swarm (5 nodes/ 2500 rows) vs Central (12500 rows)'
    # )
    # plot_auc_loss_stacked(
    #     'results/70k_fixed_rows_entire_testset_epochs/5_0_lr0.00025_bs8/central_results.csv',
    #     'results/70k_fixed_rows_entire_testset_epochs/5_0_lr0.07500_bs256/swarm_results.csv',
    #     250,
    #     (0.5, 0.9),
    #     (0.15, 0.5),
    #     'plot_epochs_5nodes_70k_fixed_rows_auc_loss_entire.png',
    #     'Swarm (5 nodes/ 2500 rows) vs Central (12500 rows)'
    # )
    # # # Variable rows experiment: combined AUC + Loss
    # plot_auc_loss_stacked(
    #     'results/70k_var_rows_separate_testset_epochs/5_0_lr0.01000_bs128/central_results.csv',
    #     'results/70k_var_rows_separate_testset_epochs/5_0_lr0.00500_bs8/swarm_results.csv',
    #     250,
    #     (0.5, 0.9),
    #     (0.15, 0.5),
    #     'plot_epochs_5nodes_70k_var_rows_auc_loss.png',
    #     'Swarm (5 nodes/ ~3000 rows) vs Central (14436 rows)'
    # )
    # plot_auc_loss_stacked(
    #     'results/mimiciii_total_entire_testset_epochs/5_0_lr0.00100_bs16/central_results.csv',
    #     'results/mimiciii_total_entire_testset_epochs/5_0_lr0.00250_bs32/swarm_results.csv',
    #     250,
    #     (0.5, 0.9),
    #     (0.15, 0.5),
    #     'mimiciii_total_entire_testset_epochs.png',
    #     'Swarm (5 nodes/ 2459 rows) vs Central (12295 rows)'
    # )
    # plot_auc_loss_stacked(
    #     'results/mimiciv_total_entire_testset_epochs/5_0_lr0.00100_bs16/central_results.csv',
    #     'results/mimiciv_total_entire_testset_epochs/5_0_lr0.00100_bs8/swarm_results.csv',
    #     250,
    #     (0.5, 0.9),
    #     (0.15, 0.5),
    #     'mimiciv_total_entire_testset_epochs.png',
    #     'Swarm (5 nodes/ 7155 rows) vs Central (37775 rows)'
    # )
    # plot_auc_loss_stacked(
    #     'results/mimiciv_fixed_12500_entire_testset_epochs/5_0_lr0.00100_bs64/central_results.csv',
    #     'results/mimiciv_fixed_12500_entire_testset_epochs/5_0_lr0.00500_bs8/swarm_results.csv',
    #     250,
    #     (0.7, 0.95),
    #     (0.2, 0.45),
    #     'mimiciv_fixed_12500_entire_testset_epochs.png',
    #     'Swarm (5 nodes/ 2500 rows) vs Central (12500 rows)'
    # )
    # plot_auc_loss_stacked(
    #     'results/70k_var_rows_entire_testset_epochs/5_0_lr0.01000_bs128/central_results.csv',
    #     'results/70k_var_rows_entire_testset_epochs/5_0_lr0.01000_bs32/swarm_results.csv',
    #     250,
    #     (0.5, 0.9),
    #     (0.15, 0.5),
    #     'plot_epochs_5nodes_70k_var_rows_auc_loss_entire.png',
    #     'Swarm (5 nodes/ ~3000 rows) vs Central (14436 rows)'
    # )
    plot_auc_loss_stacked(
        'results/70k_fixed_rows_entire_testset_epochs/5_0_lr0.00025_bs8/central_results.csv',
        'results/70k_fixed_rows_entire_testset_epochs/5_0_lr0.01000_bs8/swarm_results.csv',
        250,
        (0.5, 0.9),
        (0.15, 0.5),
        'plot_epochs_5nodes_70k_fixed_rows_auc_loss_entire_different_lr_bs.png',
        'Swarm (5 nodes/ 2500 rows) vs Central (12500 rows)'
    )
    plot_auc_loss_stacked(
        'results/70k_fixed_rows_entire_testset_epochs/5_0_lr0.01000_bs8/swarm_results.csv',
        'results/70k_fixed_rows_entire_testset_epochs/5_0_lr0.07500_bs256/swarm_results.csv',
        250,
        (0.5, 0.9),
        (0.15, 0.5),
        'plot_epochs_5nodes_70k_fixed_rows_auc_loss_entire_swarm_lr_bs.png',
        'Swarm (5 nodes/ 2500 rows) LR = 0.01000, BS = 8 vs Swarm (5 nodes/ 2500 rows) LR = 0.07500, BS = 256'
    )
