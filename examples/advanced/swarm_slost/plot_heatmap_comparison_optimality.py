import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import t
from math import sqrt
from statistics import stdev
import matplotlib
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.colors as mcolors
from scipy import stats
import matplotlib.patches as patches
from scipy import ndimage

os.chdir(Path(__file__).resolve().parent)

def nadeau_bengio_corrected_ttest(scores_a, scores_b, n_train, n_test):
    """
    Esegue il t-test accoppiato corretto di Nadeau & Bengio.
    
    Args:
        scores_a (list/array): Lista delle AUC del metodo A per le 10 run.
        scores_b (list/array): Lista delle AUC del metodo B per le 10 run.
        n_train (int): Dimensione totale del training set (somma di tutti i client).
        n_test (int): Dimensione del test set.
        
    Returns:
        t_stat, p_value
    """
    diff = np.array(scores_a) - np.array(scores_b)
    n = len(diff)  # Numero di run (es. 10)
    
    mu_diff = np.mean(diff)
    sigma2_diff = np.var(diff, ddof=1) # Varianza campionaria non distorta
    
    # Calcolo del fattore di correzione
    # Questo termine riduce il t-value per compensare la correlazione tra run
    correction_factor = (1/n) + (n_test / n_train)
    
    # Calcolo della t-statistic corretta
    t_stat = mu_diff / np.sqrt(correction_factor * sigma2_diff)
    
    # Calcolo del p-value (distribuzione t con n-1 gradi di libertà)
    # Moltiplichiamo per 2 per un test a due code
    p_val = stats.t.sf(np.abs(t_stat), df=n-1) * 2
    
    return t_stat, p_val

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

def load_results_for_ttest(directories, csv_filename, learning_rates=None, batch_sizes=None):
    """Load all individual results for t-test"""
    auc_results = {bs: {lr: [] for lr in learning_rates} for bs in batch_sizes}
    
    for directory in directories:
        csv_path = os.path.join(directory, csv_filename)
        
        if os.path.exists(csv_path):
            try:
                df = pd.read_csv(csv_path)

                # Extract parameters
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
    """Load mean results from directories with given CSV filename"""
    auc_results = {bs: {lr: 0 for lr in learning_rates} for bs in batch_sizes}
    
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

            except Exception as e:
                print(f"Error processing {directory}: {e}")
        else:
            print(f"File not found: {csv_path}")

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

def create_triple_heatmap(comparison, metric, n_training_samples, n_test_samples):
    """
    Original triple heatmap function - creates the triple heatmap without optimality band.
    """
    learning_rates = [0.0001, 0.00025, 0.0005, 0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.075, 0.1]
    batch_sizes = [8, 16, 32, 64, 128, 256, 512]
    
    conf1_dirs = [os.path.join(comparison['dir1'], d) for d in build_directories(comparison['dir1'], learning_rates, batch_sizes, comparison['node1'])]
    conf2_dirs = [os.path.join(comparison['dir2'], d) for d in build_directories(comparison['dir2'], learning_rates, batch_sizes, comparison['node2'])]
    
    # Load individual results for t-test
    conf1_individual = load_results_for_ttest(conf1_dirs, comparison['file1'], learning_rates, batch_sizes)
    conf2_individual = load_results_for_ttest(conf2_dirs, comparison['file2'], learning_rates, batch_sizes)
    
    # Load mean results for display
    conf1_mean = load_mean_results(conf1_dirs, comparison['file1'], learning_rates, batch_sizes)
    conf2_mean = load_mean_results(conf2_dirs, comparison['file2'], learning_rates, batch_sizes)
    
    conf1_matrix = conf1_mean
    conf2_matrix = conf2_mean
    metric_title = 'AUC'
    
    batch_sizes = sorted(list(conf1_matrix.keys()), reverse=True)
    learning_rates = sorted(list(next(iter(conf1_matrix.values())).keys()))
    
    conf1_values = np.zeros((len(batch_sizes), len(learning_rates)))
    conf2_values = np.zeros((len(batch_sizes), len(learning_rates)))
    delta_values = np.zeros((len(batch_sizes), len(learning_rates)))
    p_values = np.zeros((len(batch_sizes), len(learning_rates)))
    
    for i, bs in enumerate(batch_sizes):
        for j, lr in enumerate(learning_rates):
            conf1_values[i, j] = conf1_matrix[bs][lr]
            conf2_values[i, j] = conf2_matrix[bs][lr]
            delta_values[i, j] = conf1_values[i, j] - conf2_values[i, j]
            
            # Calculate corrected dependent t-test p-value if we have individual data
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
    
    # Trova gli indici dei massimi
    max_A_indices = np.where(conf1_values == np.max(conf1_values))
    max_B_indices = np.where(conf2_values == np.max(conf2_values))
    max_delta_indices = np.where(delta_values == np.max(delta_values))
    
    # Create colormaps for each section
    colors = [
        "#ffc6ba",  # 1
        "#ffd479",  # 2
        "#ffff66",  # 3
        "#a5db6f",  # 4
        "#7288cd"   # 5
    ]
    cmap_A = LinearSegmentedColormap.from_list("my_custom_cmap", colors)
    cmap_B = LinearSegmentedColormap.from_list("my_custom_cmap", colors)
    vmin_A, vmax_A = conf1_values.max() - 0.1, conf1_values.max()
    vmin_B, vmax_B = conf2_values.max() - 0.1, conf2_values.max()

    cmap_delta = mcolors.ListedColormap(["#8b0000", "#ff4500", "#ffff66", "#66ff66", "#006400"])
    norm_delta = mcolors.BoundaryNorm([-1, -0.05, -0.01, 0.01, 0.05, 1], cmap_delta.N)
    
    out_dir = 'plots_results/plot_heatmap_comparison_pvalue'
    os.makedirs(out_dir, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(30, 16))
    
    # Create ScalarMappable for delta colorbar
    sm_delta = plt.cm.ScalarMappable(cmap=cmap_delta, norm=norm_delta)
    sm_delta.set_array([])
    
    # Plot each section separately
    for i in range(len(batch_sizes)):
        for j in range(len(learning_rates)):
            # Define rectangles for each third of the cell
            cell_width = 1
            cell_height = 1
            
            # Top third: A
            color_A = matplotlib.colormaps.get_cmap(cmap_A)((conf1_values[i, j] - vmin_A) / (vmax_A - vmin_A))
            rect_A = plt.Rectangle((j, i + 2/3), cell_width, cell_height/3, facecolor=color_A, edgecolor='black', linewidth=0.5)
            ax.add_patch(rect_A)
            
            # Middle third: B
            color_B = matplotlib.colormaps.get_cmap(cmap_B)((conf2_values[i, j] - vmin_B) / (vmax_B - vmin_B))
            rect_B = plt.Rectangle((j, i + 1/3), cell_width, cell_height/3, facecolor=color_B, edgecolor='black', linewidth=0.5)
            ax.add_patch(rect_B)
            
            # Bottom third: Delta
            color_delta = sm_delta.to_rgba(delta_values[i, j])
            rect_delta = plt.Rectangle((j, i), cell_width, cell_height/3, facecolor=color_delta, edgecolor='black', linewidth=0.5)
            ax.add_patch(rect_delta)

            macro_cell_border = plt.Rectangle((j, i), cell_width, cell_height, facecolor='none', edgecolor='black', linewidth=4, alpha=0.8)
            ax.add_patch(macro_cell_border)
            
            # Function to calculate brightness of a color
            def get_brightness(color):
                # Convert RGB to brightness (perceived luminance)
                # Using the formula: 0.299*R + 0.587*G + 0.114*B
                return 0.299 * color[0] + 0.587 * color[1] + 0.114 * color[2]
            
            # Determine text color based on background brightness
            brightness_A = get_brightness(color_A)
            brightness_B = get_brightness(color_B)
            brightness_delta = get_brightness(color_delta)
            
            # Use white text for dark backgrounds, black for light backgrounds
            text_color_A = 'white' if brightness_A < 0.5 else 'black'
            text_color_B = 'white' if brightness_B < 0.5 else 'black'
            text_color_delta = 'white' if brightness_delta < 0.5 else 'black'
            
            # Add text annotations
            # Format p-value
            p_val = p_values[i, j]
            p_text = f"p={p_val:.3f}" if not np.isnan(p_val) else "p=N/A"
            sig_star = "*" if p_val < 0.05 and not np.isnan(p_val) else ""
            
            # A section text
            ax.text(j + 0.5, i + 5/6, f"A: {conf1_values[i, j]:.4f}", ha='center', va='center', fontsize=18, color=text_color_A, fontweight='bold' if (i, j) in zip(max_A_indices[0], max_A_indices[1]) else 'normal')
            
            # B section text
            ax.text(j + 0.5, i + 0.5, f"B: {conf2_values[i, j]:.4f}", ha='center', va='center', fontsize=18, color=text_color_B, fontweight='bold' if (i, j) in zip(max_B_indices[0], max_B_indices[1]) else 'normal')
            
            # Delta section text with p-value
            delta_text = f"Δ: {delta_values[i, j]:+.4f}\n{p_text}{sig_star}"
            ax.text(j + 0.5, i + 1/6, delta_text, ha='center', va='center', fontsize=14, color=text_color_delta, fontweight='bold' if (i, j) in zip(max_delta_indices[0], max_delta_indices[1]) else 'normal')
    
    # Set limits and labels
    ax.set_xlim(0, len(learning_rates))
    ax.set_ylim(0, len(batch_sizes))
    
    # Set ticks
    ax.set_xticks(np.arange(len(learning_rates)) + 0.5)
    ax.set_yticks(np.arange(len(batch_sizes)) + 0.5)
    ax.set_xticklabels([f"{lr:.5f}" for lr in learning_rates], rotation=45, fontsize=14)
    ax.set_yticklabels(batch_sizes, fontsize=14)
    
    ax.set_xlabel('Learning Rate', fontsize=14)
    ax.set_ylabel('Batch Size', fontsize=14)
    
    # Create colorbars
    # Colorbar for A
    sm_A = plt.cm.ScalarMappable(cmap=cmap_A, norm=plt.Normalize(vmin=vmin_A, vmax=vmax_A))
    sm_A.set_array([])
    cbar_A = fig.colorbar(sm_A, ax=ax, orientation='vertical', shrink=0.9, pad=-0.06)
    cbar_A.set_label(f'A ({metric_title})', fontsize=12)
    
    # Colorbar for B
    sm_B = plt.cm.ScalarMappable(cmap=cmap_B, norm=plt.Normalize(vmin=vmin_B, vmax=vmax_B))
    sm_B.set_array([])
    cbar_B = fig.colorbar(sm_B, ax=ax, orientation='vertical', shrink=0.9, pad=-0.05)
    cbar_B.set_label(f'B ({metric_title})', fontsize=12)
    
    # Colorbar for Delta
    cbar_delta = fig.colorbar(sm_delta, ax=ax, orientation='vertical', shrink=0.9, pad=0.02)
    cbar_delta.set_label(f'Δ ({metric_title}: A - B)', fontsize=12)
    
    # Title
    exp = comparison['exp']
    plt.title(f'{metric_title} Triple Heatmap Comparison: {exp}\n' +
            f'Corrected Dependent t-test (train={n_training_samples}, test={n_test_samples})\n' +
            '(Each cell divided into: Top=A, Middle=B, Bottom=Δ with p-value)', 
            fontsize=16, pad=20)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save plot
    exp_replaced = exp.replace(' ', '_').replace('=', 'vs').replace('(', '').replace(')', '').replace(',', '').replace('/', '_')
    filename = f'{out_dir}/triple_heatmap_{metric}_{exp_replaced}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    return conf1_values, conf2_values, delta_values, p_values, batch_sizes, learning_rates

def identify_optimality_band(conf1_values, conf2_values):
    # Calculate mean of both configurations
    mean_values = (conf1_values + conf2_values) / 2.0
    
    # Optimality band: mean(conf1, conf2) > max mean - 0.05
    threshold = mean_values.max() - 0.02
    optimality_mask = mean_values > threshold
    return optimality_mask, mean_values

def plot_triple_heatmap_with_optimality(ax, conf1_values, conf2_values, delta_values, p_values,
                                        batch_sizes, learning_rates, optimality_mask,
                                        vmin_A, vmax_A, vmin_B, vmax_B, cmap_A, cmap_B, cmap_delta, norm_delta):
    """
    Plot the triple heatmap with optimality band highlighted.
    """
    # Create ScalarMappable for delta colorbar
    sm_delta = plt.cm.ScalarMappable(cmap=cmap_delta, norm=norm_delta)
    sm_delta.set_array([])
    
    # First, draw the optimality band background
    for i in range(len(batch_sizes)):
        for j in range(len(learning_rates)):
            if optimality_mask[i, j]:
                # Draw light green background for optimal cells
                opt_rect = patches.Rectangle(
                    (j, i), 1, 1,
                    facecolor='lightgreen', alpha=0.3,
                    edgecolor='darkgreen', linewidth=2
                )
                ax.add_patch(opt_rect)
            else:
                # Draw light red background for non-optimal cells
                non_opt_rect = patches.Rectangle(
                    (j, i), 1, 1,
                    facecolor='lightcoral', alpha=0.2,
                    edgecolor='darkred', linewidth=1, linestyle='--'
                )
                ax.add_patch(non_opt_rect)
    
    # Now plot the triple heatmap cells on top
    for i in range(len(batch_sizes)):
        for j in range(len(learning_rates)):
            cell_width = 1
            cell_height = 1
            
            # Top third: A
            color_A = matplotlib.colormaps.get_cmap(cmap_A)((conf1_values[i, j] - vmin_A) / (vmax_A - vmin_A))
            rect_A = patches.Rectangle(
                (j, i + 2/3), cell_width, cell_height/3,
                facecolor=color_A, edgecolor='black', linewidth=0.5
            )
            ax.add_patch(rect_A)
            
            # Middle third: B
            color_B = matplotlib.colormaps.get_cmap(cmap_B)((conf2_values[i, j] - vmin_B) / (vmax_B - vmin_B))
            rect_B = patches.Rectangle(
                (j, i + 1/3), cell_width, cell_height/3,
                facecolor=color_B, edgecolor='black', linewidth=0.5
            )
            ax.add_patch(rect_B)
            
            # Bottom third: Delta
            color_delta = sm_delta.to_rgba(delta_values[i, j])
            rect_delta = patches.Rectangle(
                (j, i), cell_width, cell_height/3,
                facecolor=color_delta, edgecolor='black', linewidth=0.5
            )
            ax.add_patch(rect_delta)
            
            # Add macro cell border
            border_color = 'darkgreen' if optimality_mask[i, j] else 'black'
            border_width = 3 if optimality_mask[i, j] else 1
            macro_border = patches.Rectangle(
                (j, i), cell_width, cell_height,
                facecolor='none', edgecolor=border_color,
                linewidth=border_width, alpha=0.8
            )
            ax.add_patch(macro_border)
            
            # Function to calculate brightness
            def get_brightness(color):
                return 0.299 * color[0] + 0.587 * color[1] + 0.114 * color[2]
            
            # Add text annotations
            brightness_A = get_brightness(color_A)
            brightness_B = get_brightness(color_B)
            brightness_delta = get_brightness(color_delta)
            
            text_color_A = 'white' if brightness_A < 0.5 else 'black'
            text_color_B = 'white' if brightness_B < 0.5 else 'black'
            text_color_delta = 'white' if brightness_delta < 0.5 else 'black'
            
            # Format p-value
            p_val = p_values[i, j]
            p_text = f"p={p_val:.3f}" if not np.isnan(p_val) else "p=N/A"
            sig_star = "*" if p_val < 0.05 and not np.isnan(p_val) else ""
            
            # A section text
            ax.text(
                j + 0.5, i + 5/6, f"A: {conf1_values[i, j]:.4f}",
                ha='center', va='center', fontsize=12,
                color=text_color_A, fontweight='bold' if optimality_mask[i, j] else 'normal'
            )
            
            # B section text
            ax.text(
                j + 0.5, i + 0.5, f"B: {conf2_values[i, j]:.4f}",
                ha='center', va='center', fontsize=12,
                color=text_color_B, fontweight='bold' if optimality_mask[i, j] else 'normal'
            )
            
            # Delta section text
            delta_text = f"Δ: {delta_values[i, j]:+.4f}\n{p_text}{sig_star}"
            ax.text(
                j + 0.5, i + 1/6, delta_text,
                ha='center', va='center', fontsize=10,
                color=text_color_delta, fontweight='bold' if optimality_mask[i, j] else 'normal'
            )
    
    # Set axis properties
    ax.set_xlim(0, len(learning_rates))
    ax.set_ylim(0, len(batch_sizes))
    ax.set_xticks(np.arange(len(learning_rates)) + 0.5)
    ax.set_yticks(np.arange(len(batch_sizes)) + 0.5)
    ax.set_xticklabels([f"{lr:.5f}" for lr in learning_rates], rotation=45, fontsize=10)
    ax.set_yticklabels(batch_sizes, fontsize=10)
    ax.set_xlabel('Learning Rate', fontsize=12)
    ax.set_ylabel('Batch Size', fontsize=12)
    ax.set_title('Triple Heatmap with Optimality Band', fontsize=14, pad=20)
    
    return sm_delta

def plot_combined_score_heatmap(ax, combined_scores, optimality_mask, batch_sizes, learning_rates):
    """
    Plot heatmap of combined scores showing optimality gradient.
    """
    # Reverse the batch sizes and adjust the matrix accordingly
    batch_sizes = batch_sizes[::-1]
    combined_scores = combined_scores[::-1]
    optimality_mask = optimality_mask[::-1]
    
    # Adjust the figure size to make the heatmap more compact
    aspect_ratio = len(learning_rates) / len(batch_sizes)
    ax.set_aspect(aspect_ratio * 0.5)  # Reduce width scaling
    
    # Plot the heatmap with no colormap scaling
    im = ax.imshow(combined_scores, cmap='Greys', aspect='auto', interpolation='nearest', vmin=0.5, vmax=0.9)
    
    # Add optimality band outline and background color
    for i in range(len(batch_sizes)):
        for j in range(len(learning_rates)):
            # Set background color based on optimality mask
            if optimality_mask[i, j]:
                rect = patches.Rectangle(
                    (j - 0.5, i - 0.5), 1, 1,
                    facecolor='lightgreen', edgecolor='black',
                    linewidth=2, alpha=0.3
                )
            else:
                rect = patches.Rectangle(
                    (j - 0.5, i - 0.5), 1, 1,
                    facecolor='lightcoral', edgecolor='black',
                    linewidth=2, alpha=0.3
                )
            ax.add_patch(rect)
    
    # Add text annotations with scores
    for i in range(len(batch_sizes)):
        for j in range(len(learning_rates)):
            score = combined_scores[i, j]
            if not np.isinf(score):
                ax.text(j, i, f"{score:.4f}", ha='center', va='center', color='black', fontsize=14, fontweight='bold')
    
    # Set axis properties
    ax.set_xticks(np.arange(len(learning_rates)))
    ax.set_yticks(np.arange(len(batch_sizes)))
    ax.set_xticklabels([f"{lr:.5f}" for lr in learning_rates], rotation=45, fontsize=8)
    ax.set_yticklabels(batch_sizes, fontsize=8)
    ax.set_xlabel('Learning Rate', fontsize=10)
    ax.set_ylabel('Batch Size', fontsize=10)
    ax.set_title('Combined Optimality Score\n(Mean performance weighted by p-value)', fontsize=12, pad=15)
    
    return im

def create_triple_heatmap_with_optimality_band(comparison, metric, n_training_samples, n_test_samples):
    learning_rates = [0.0001, 0.00025, 0.0005, 0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.075, 0.1]
    batch_sizes = [8, 16, 32, 64, 128, 256, 512]
    
    conf1_dirs = [os.path.join(comparison['dir1'], d) for d in build_directories(comparison['dir1'], learning_rates, batch_sizes, comparison['node1'])]
    conf2_dirs = [os.path.join(comparison['dir2'], d) for d in build_directories(comparison['dir2'], learning_rates, batch_sizes, comparison['node2'])]
    
    # Load individual results for t-test
    conf1_individual = load_results_for_ttest(conf1_dirs, comparison['file1'], learning_rates, batch_sizes)
    conf2_individual = load_results_for_ttest(conf2_dirs, comparison['file2'], learning_rates, batch_sizes)
    
    # Load mean results for display
    conf1_mean = load_mean_results(conf1_dirs, comparison['file1'], learning_rates, batch_sizes)
    conf2_mean = load_mean_results(conf2_dirs, comparison['file2'], learning_rates, batch_sizes)
    
    conf1_matrix = conf1_mean
    conf2_matrix = conf2_mean
    metric_title = 'AUC'
    
    batch_sizes = sorted(list(conf1_matrix.keys()), reverse=True)
    learning_rates = sorted(list(next(iter(conf1_matrix.values())).keys()))
    
    conf1_values = np.zeros((len(batch_sizes), len(learning_rates)))
    conf2_values = np.zeros((len(batch_sizes), len(learning_rates)))
    delta_values = np.zeros((len(batch_sizes), len(learning_rates)))
    p_values = np.zeros((len(batch_sizes), len(learning_rates)))
    
    for i, bs in enumerate(batch_sizes):
        for j, lr in enumerate(learning_rates):
            conf1_values[i, j] = conf1_matrix[bs][lr]
            conf2_values[i, j] = conf2_matrix[bs][lr]
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
    
    # Identify optimality band
    optimality_mask, combined_scores = identify_optimality_band(conf1_values, conf2_values)
    
    # Create colormaps for each section
    colors = [
        "#ffc6ba",  # 1
        "#ffd479",  # 2
        "#ffff66",  # 3
        "#a5db6f",  # 4
        "#7288cd"   # 5
    ]
    cmap_A = LinearSegmentedColormap.from_list("my_custom_cmap", colors)
    cmap_B = LinearSegmentedColormap.from_list("my_custom_cmap", colors)
    vmin_A, vmax_A = conf1_values.max() - 0.1, conf1_values.max()
    vmin_B, vmax_B = conf2_values.max() - 0.1, conf2_values.max()

    cmap_delta = mcolors.ListedColormap(["#8b0000", "#ff4500", "#ffff66", "#66ff66", "#006400"])
    norm_delta = mcolors.BoundaryNorm([-1, -0.05, -0.01, 0.01, 0.05, 1], cmap_delta.N)
    
    # Create figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(30, 14), gridspec_kw={'width_ratios': [3, 2]})
    
    # Plot 1: Original triple heatmap with optimality band highlighted
    ax1 = axes[0]
    sm_delta = plot_triple_heatmap_with_optimality(
        ax1, conf1_values, conf2_values, delta_values, p_values,
        batch_sizes, learning_rates, optimality_mask,
        vmin_A, vmax_A, vmin_B, vmax_B, cmap_A, cmap_B, cmap_delta, norm_delta
    )
    
    # Add colorbars to the first subplot
    sm_A = plt.cm.ScalarMappable(cmap=cmap_A, norm=plt.Normalize(vmin=vmin_A, vmax=vmax_A))
    sm_A.set_array([])
    cbar_A = fig.colorbar(sm_A, ax=ax1, orientation='vertical', shrink=0.9, pad=-0.06)
    cbar_A.set_label(f'A ({metric_title})', fontsize=12)
    
    sm_B = plt.cm.ScalarMappable(cmap=cmap_B, norm=plt.Normalize(vmin=vmin_B, vmax=vmax_B))
    sm_B.set_array([])
    cbar_B = fig.colorbar(sm_B, ax=ax1, orientation='vertical', shrink=0.9, pad=-0.05)
    cbar_B.set_label(f'B ({metric_title})', fontsize=12)
    
    cbar_delta = fig.colorbar(sm_delta, ax=ax1, orientation='vertical', shrink=0.9, pad=0.02)
    cbar_delta.set_label(f'Δ ({metric_title}: A - B)', fontsize=12)
    
    # Plot 2: Combined score heatmap showing optimality gradient
    ax2 = axes[1]
    im = plot_combined_score_heatmap(
        ax2, combined_scores, optimality_mask,
        batch_sizes, learning_rates
    )
    
    # Add main title
    exp = comparison['exp']
    fig.suptitle(
        f'{metric} Analysis with Optimality Band\n'
        f'Experiment: {exp}\n'
        f'Corrected Dependent t-test (train={n_training_samples}, test={n_test_samples})\n'
        f'Optimality',
        fontsize=16, y=1.02
    )
    
    # Adjust layout and save
    plt.tight_layout()
    out_dir = 'plots_results/plot_optimality_analysis'
    os.makedirs(out_dir, exist_ok=True)
    
    exp_replaced = exp.replace(' ', '_').replace('=', 'vs').replace('(', '').replace(')', '').replace(',', '').replace('/', '_')
    filename = f'{out_dir}/optimality_analysis_{metric}_{exp_replaced}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Optimality band covers {np.sum(optimality_mask)}/{optimality_mask.size} cells")
    print(f"Optimality percentage: {100 * np.sum(optimality_mask) / optimality_mask.size:.1f}%")
    
    return optimality_mask, combined_scores, conf1_values, conf2_values, p_values

def try_plot_triple_comparison(comparison):
    """Wrapper function with error handling"""
    try:
        _, _, _, _, _ = create_triple_heatmap_with_optimality_band(
            comparison=comparison,
            metric='AUC',
            n_training_samples=comparison['n_training_samples'],
            n_test_samples=comparison['n_test_samples'],
        )
    except Exception as e:
        print(f"An error occurred while plotting triple comparison for {comparison['exp']}: {e}")
        import traceback
        traceback.print_exc()

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
            'node2': 2,
            'n_training_samples': 2000,
            'n_test_samples': 10000
        },
        {
            'exp': 'SLOST_NVFLARE_A = Swarm 2 nodes NVFlare (2000 rows/node) vs B = Swarm 2 nodes SLOST (2000 rows/node)',
            'dir1': 'results/heatmap_experiments_2000rows_2nodes',
            'file1': 'swarm_results.csv',
            'node1': 2,
            'dir2': '/home/swarm-learning/repos/SLOST-local/results/heatmap_experiments_2000rows_2nodes',
            'file2': 'swarm_results.csv',
            'node2': 2,
            'n_training_samples': 4000,
            'n_test_samples': 10000
        },
        {
            'exp': 'SLOST_NVFLARE_A = Swarm 2 nodes NVFlare (4000 rows/node) vs B = Swarm 2 nodes SLOST (4000 rows/node)',
            'dir1': 'results/heatmap_experiments_4000rows_2nodes',
            'file1': 'swarm_results.csv',
            'node1': 2,
            'dir2': '/home/swarm-learning/repos/SLOST-local/results/heatmap_experiments_4000rows_2nodes',
            'file2': 'swarm_results.csv',
            'node2': 2,
            'n_training_samples': 8000,
            'n_test_samples': 10000
        },
        {
            'exp': 'SLOST_NVFLARE_A = Swarm 5 nodes NVFlare (1000 rows/node) vs B = Swarm 5 nodes SLOST (1000 rows/node)',
            'dir1': 'results/heatmap_experiments_1000rows_5nodes',
            'file1': 'swarm_results.csv',
            'node1': 5,
            'dir2': '/home/swarm-learning/repos/SLOST-local/results/heatmap_experiments_1000rows_5nodes',
            'file2': 'swarm_results.csv',
            'node2': 5,
            'n_training_samples': 5000,
            'n_test_samples': 10000
        },
        {
            'exp': 'SLOST_NVFLARE_A = Swarm 5 nodes NVFlare (2000 rows/node) vs B = Swarm 5 nodes SLOST (2000 rows/node)',
            'dir1': 'results/heatmap_experiments_2000rows_5nodes',
            'file1': 'swarm_results.csv',
            'node1': 5,
            'dir2': '/home/swarm-learning/repos/SLOST-local/results/heatmap_experiments_2000rows_5nodes',
            'file2': 'swarm_results.csv',
            'node2': 5,
            'n_training_samples': 10000,
            'n_test_samples': 10000
        },
        {
            'exp': 'SLOST_NVFLARE_A = Swarm 5 nodes NVFlare (4000 rows/node) vs B = Swarm 5 nodes SLOST (4000 rows/node)',
            'dir1': 'results/heatmap_experiments_4000rows_5nodes',
            'file1': 'swarm_results.csv',
            'node1': 5,
            'dir2': '/home/swarm-learning/repos/SLOST-local/results/heatmap_experiments_4000rows_5nodes',
            'file2': 'swarm_results.csv',
            'node2': 5,
            'n_training_samples': 20000,
            'n_test_samples': 10000
        },
        {
            'exp': 'SLOST_NVFLARE_A = Swarm 4 nodes NVFlare (entire mimic iii) vs B = Swarm 4 nodes SLOST (entire mimic iii)',
            'dir1': 'results/heatmap_experiments_4nodes_iii',
            'file1': 'swarm_results.csv',
            'node1': 4,
            'dir2': '/home/swarm-learning/repos/SLOST-local/results/heatmap_experiments_4nodes_iii',
            'file2': 'swarm_results.csv',
            'node2': 4,
            'n_training_samples': 12300,
            'n_test_samples': 5271
        },
        {
            'exp': 'SLOST_NVFLARE_A = Swarm 4 nodes NVFlare (entire mimic iv) vs B = Swarm 4 nodes SLOST (entire mimic iv)',
            'dir1': 'results/heatmap_experiments_4nodes_iv',
            'file1': 'swarm_results.csv',
            'node1': 4,
            'dir2': '/home/swarm-learning/repos/SLOST-local/results/heatmap_experiments_4nodes_iv',
            'file2': 'swarm_results.csv',
            'node2': 4,
            'n_training_samples': 35772,
            'n_test_samples': 15330
        },
        # 2 nodes
        {
            'exp': 'A = Swarm 2 nodes (1000 rows/node) vs B = Centralized (2000 rows)',
            'dir1': 'results/heatmap_experiments_1000rows_2nodes',
            'file1': 'swarm_results.csv',
            'node1': 2,
            'dir2': 'results/heatmap_experiments_1000rows_2nodes',
            'file2': 'central_results.csv',
            'node2': 2,
            'n_training_samples': 2000,
            'n_test_samples': 10000
        },
        {
            'exp': 'A = Swarm 2 nodes (2000 rows/node) vs B = Centralized (4000 rows)',
            'dir1': 'results/heatmap_experiments_2000rows_2nodes',
            'file1': 'swarm_results.csv',
            'node1': 2,
            'dir2': 'results/heatmap_experiments_2000rows_2nodes',
            'file2': 'central_results.csv',
            'node2': 2,
            'n_training_samples': 4000,
            'n_test_samples': 10000
        },
        {
            'exp': 'A = Swarm 2 nodes (4000 rows/node) vs B = Centralized (8000 rows)',
            'dir1': 'results/heatmap_experiments_4000rows_2nodes',
            'file1': 'swarm_results.csv',
            'node1': 2,
            'dir2': 'results/heatmap_experiments_4000rows_2nodes',
            'file2': 'central_results.csv',
            'node2': 2,
            'n_training_samples': 8000,
            'n_test_samples': 10000
        },
        # 5 nodes
        {
            'exp': 'A = Swarm 5 nodes (1000 rows/node) vs B = Centralized (5000 rows)',
            'dir1': 'results/heatmap_experiments_1000rows_5nodes',
            'file1': 'swarm_results.csv',
            'node1': 5,
            'dir2': 'results/heatmap_experiments_1000rows_5nodes',
            'file2': 'central_results.csv',
            'node2': 5,
            'n_training_samples': 5000,
            'n_test_samples': 10000
        },
        {
            'exp': 'A = Swarm 5 nodes (2000 rows/node) vs B = Centralized (10000 rows)',
            'dir1': 'results/heatmap_experiments_2000rows_5nodes',
            'file1': 'swarm_results.csv',
            'node1': 5,
            'dir2': 'results/heatmap_experiments_2000rows_5nodes',
            'file2': 'central_results.csv',
            'node2': 5,
            'n_training_samples': 10000,
            'n_test_samples': 10000
        },
        {
            'exp': 'A = Swarm 5 nodes (4000 rows/node) vs B = Centralized (20000 rows)',
            'dir1': 'results/heatmap_experiments_4000rows_5nodes',
            'file1': 'swarm_results.csv',
            'node1': 5,
            'dir2': 'results/heatmap_experiments_4000rows_5nodes',
            'file2': 'central_results.csv',
            'node2': 5,
            'n_training_samples': 20000,
            'n_test_samples': 10000
        },
        # entire datasets
        {
            'exp': 'A = Swarm 4 nodes vs B = Centralized (entire mimic iii)',
            'dir1': 'results/heatmap_experiments_4nodes_iii',
            'file1': 'swarm_results.csv',
            'node1': 4,
            'dir2': 'results/heatmap_experiments_4nodes_iii',
            'file2': 'central_results.csv',
            'node2': 4,
            'n_training_samples': 12300,
            'n_test_samples': 5271
        },
        {
            'exp': 'A = Swarm 4 nodes vs B = Centralized (entire mimic iv)',
            'dir1': 'results/heatmap_experiments_4nodes_iv',
            'file1': 'swarm_results.csv',
            'node1': 4,
            'dir2': 'results/heatmap_experiments_4nodes_iv',
            'file2': 'central_results.csv',
            'node2': 4,
            'n_training_samples': 35772,
            'n_test_samples': 15330
        }
    ]
    
    for comp in comparisons:
        try_plot_triple_comparison(comp)
