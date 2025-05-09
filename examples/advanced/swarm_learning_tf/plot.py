import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import glob
import itertools
from collections import defaultdict
import matplotlib.colors as mcolors
from itertools import zip_longest
import argparse

parser = argparse.ArgumentParser(description="Comparison plots for AUC values.")
parser.add_argument('main_folder', help='Cartella in cui sono contenuti i dati')

args = parser.parse_args()
main_folder = args.main_folder

# Define the metric function and labels in one place
METRIC_NAME = "AUC Comparison"  # Change this to any metric name you want
METRIC_FILENAME = "AUC_Comparison"
METRIC_VAR = "auc_comparison"

### FILES
workspace_dir = os.path.join(os.getcwd(), main_folder)  # Uses the current working directory
output_dir = os.path.join(workspace_dir, "Plots/", METRIC_FILENAME)  # Output directory for plots

##SETUP SETTINGS
# Define the ALL flag (True = run all combinations, False = manually specify)
ALL = False  # Change to False to manually control which combinations to run

DEBUG = False

# Define the boolean options for each variable
FLAGS = {
    "CONFIDENCE": [True, False],
    "SEPARATE": [True, False],
    "SWARM": [True],
    "CENTRAL": [True]
}

# Extract the keys (variable names)
KEYS = FLAGS.keys()

# If ALL is True, generate all possible True/False combinations
if ALL:
    COMBINATIONS = list(itertools.product(*FLAGS.values()))
else:
    # Manually specify the combinations you want to run
    COMBINATIONS = [
        #(True, True, False, True),  # CONFIDENCE=True, SEPARATE=True, SWARM=False, CENTRAL=True
        (False, False, True, False)  # CONFIDENCE, SEPARATE, SWARM, CENTRAL
        # Add more manual cases as needed
    ]

### METRIC FUNCTION  ###
def metric_function(values):  # Place here the metric function
    #values = np.array(values)
    return np.mean(values)  # Example: Variance as a metric

### COLORS
# Get default color cycle
default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

# Choose a manual color (e.g., green from the default cycle)
reference_color = default_colors[0]
central_color = default_colors[1]
swarm_color = default_colors[2]
avg_others_color = default_colors[3]

# List of manual colors
manual_colors = [reference_color, swarm_color, central_color, avg_others_color, default_colors[6]] #We also remove the pink as it is too similar to purple

# Define new colors to add
new_colors = ["#33FF57", "#3357FF", "#FFD700", "800080", "FF1493", "008080"] 

# Filter out the manually assigned colors
filtered_colors = [c for c in default_colors if c not in manual_colors] + new_colors

# Set new color cycle
plt.rcParams['axes.prop_cycle'] = plt.cycler(color=filtered_colors)


plt.rc('font', size=20)  # Set font size for all text
plt.rc('axes', titlesize=24, labelsize=20)  # Title and labels
plt.rc('xtick', labelsize=14)  # X-tick labels
plt.rc('ytick', labelsize=14)  # Y-tick labels
plt.rc('legend', fontsize=16)  # Legend font size
plt.rc('figure', titlesize=24)  # Figure title

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

# **Find all local_aucs.csv files in any subdirectory**
local_files = glob.glob(os.path.join(workspace_dir, "**/swarm_results/local_aucs.csv"), recursive=True)
swarm_files = glob.glob(os.path.join(workspace_dir, "**/swarm_results/aucs.csv"), recursive=True)
central_files = glob.glob(os.path.join(workspace_dir, "**/central_results/central_aucs.csv"), recursive=True)

# Define fixed pairs of subfolders for analysis
grouped_folders = [
    # ("1host_1node_data", "1host_2nodes"),
    ("1host_2nodes_data", "1host_3nodes"),
    ("1host_3nodes_data", "1host_4nodes"),
    ("1host_4nodes_data", "1host_5nodes"),
]

file_groups = defaultdict(lambda: defaultdict(lambda: {"local": [], "swarm": [], "central": []}))

for file in local_files + swarm_files:
    subfolder = os.path.dirname(file)
    subfolder_name = os.path.basename(os.path.dirname(subfolder))  # Ensure correct folder extraction
    for group in grouped_folders:
        if subfolder_name in group:
            if "local_aucs.csv" in file:
                file_groups[main_folder][group]["local"].append(file)
            else:
                file_groups[main_folder][group]["swarm"].append(file)

for central_file in central_files:
    central_subfolder = os.path.basename(os.path.dirname(os.path.dirname(central_file)))  # Extract correct parent folder
    for group in grouped_folders:
        if central_subfolder in group:
            file_groups[main_folder][group]["central"].append(central_file)


# Sort file_groups alphabetically
file_groups = {k: dict(sorted(v.items(), reverse=True)) for k, v in sorted(file_groups.items())}


def run_experiment(CONFIDENCE, SEPARATE, SWARM, CENTRAL):

    
    for pair in grouped_folders:
        print(f"\n🔹 Processing Pair: {pair}")   

        # Sort files in descending order
        local_files = sorted(file_groups[main_folder][pair]["local"], reverse=True)
        swarm_files = sorted(file_groups[main_folder][pair]["swarm"], reverse=True)
        central_files = sorted(file_groups[main_folder][pair]["central"], reverse=True)

        plot_data = {}

        # **Loop through each subfolder group and process files together**
        # for element in file_groups[main_folder][pair].items():
        for local_file, swarm_file, central_file in zip_longest(local_files, swarm_files, central_files, fillvalue=None):
            
            if os.path.dirname(local_file) != os.path.dirname(swarm_file):
                continue

            subfolder = os.path.dirname(local_file)
            
            print(f"\n🔹 Processing Local: {local_file}")
            print(f"\n🔹 Processing Swarm: {swarm_file}")
            print(f"\n🔹 Processing Central: {central_file}")

            try:
                # **Extract the relative path for output filename**
                relative_path = os.path.relpath(local_file, workspace_dir).replace("/", "_").replace("\\", "_")

                # **Read CSV file**
                df = pd.read_csv(local_file)

                # **Retrieve unique users and determine the reference user**
                users = sorted(df["user"].unique())  # Get all users
                ref_user = 1
                reference = True
                if users[0] != 1:
                    reference = False

                

                grouped = df.groupby(['user', 'splits']).agg(
                    mean_score=('auc', metric_function),  # Compute metric (e.g., variance, mean)
                    std_dev=('auc', 'std')  # Compute standard deviation for confidence interval
                ).reset_index()
                grouped.rename(columns={'mean_score': METRIC_VAR, 'std_dev': 'std_dev'}, inplace=True)  # Add standard deviation


                



                ### PRINTING FOR DEBUG PURPOSES
                if DEBUG:
                    # Extract reference node data
                    ref_data = grouped[grouped['user'] == ref_user]

                    # Compute the average Jain's Index for other nodes per split
                    avg_metric_score_data = grouped[grouped['user'] != ref_user].groupby('splits')[METRIC_VAR].mean().reset_index()

                    # Print only the reference node and the averaged values
                    print("\nReference Node Data:")
                    print(ref_data)

                    print("\nAverage " + METRIC_NAME + " for Other Nodes:")
                    print(avg_metric_score_data)
                    







                # **Reference Node (User 1) - Using Bottom X-Axis**
                if reference: 
                    ref_df = grouped[grouped['user'] == ref_user].copy()
                    ref_splits = sorted(df[df['user'] == ref_user]['splits'].unique())  # Bottom X-axis values in ascending order
                    ref_metric_score = [ref_df[ref_df['splits'] == split][METRIC_VAR].values[0] for split in ref_splits]

                    if CONFIDENCE:
                        # Compute Confidence Interval for Reference Node
                        ref_std_dev = [ref_df[ref_df['splits'] == split]['std_dev'].values[0] for split in ref_splits]  
                        ref_conf_int_low = np.array(ref_metric_score) - 1.96 * np.array(ref_std_dev)  # 95% CI lower bound
                        ref_conf_int_high = np.array(ref_metric_score) + 1.96 * np.array(ref_std_dev)  # 95% CI upper bound


                # **Other Users - Using Top X-Axis (Reversed Order)**
                other_users = [u for u in users if u in grouped['user'].unique() and u != ref_user]
                if len(other_users) > 0:
                    other_splits = sorted(df[df['user'] != ref_user]['splits'].unique())  
                    other_splits_reversed = other_splits[::-1]  

                    # **Compute top axis labels dynamically**
                    num_other_users = len(other_users)
                    other_splits_labels = [(100 - split) / num_other_users for split in ref_splits]

                    if SEPARATE:
                        user_metric_scores = {}
                        user_conf_int_low = {}
                        user_conf_int_high = {}

                        for user in other_users:
                            user_df = grouped[grouped['user'] == user]
                            user_metric_scores[user] = [user_df[user_df['splits'] == split][METRIC_VAR].values[0] for split in other_splits]

                            if CONFIDENCE:
                                user_std_dev = [user_df[user_df['splits'] == split]['std_dev'].values[0] for split in other_splits]  
                                user_conf_int_low[user] = np.array(user_metric_scores[user]) - 1.96 * np.array(user_std_dev)  
                                user_conf_int_high[user] = np.array(user_metric_scores[user]) + 1.96 * np.array(user_std_dev)  
                    else:    
                        # Compute Average Metric Score for Other Nodes Instead of Plotting Each One
                        avg_metric_score = []
                        avg_std_dev = []
                        for split in other_splits:
                            split_values = grouped[(grouped['splits'] == split) & (grouped['user'] != ref_user)][METRIC_VAR].values
                            std_values = grouped[(grouped['splits'] == split) & (grouped['user'] != ref_user)]['std_dev'].values  

                            avg_metric_score.append(np.mean(split_values) if len(split_values) > 0 else np.nan)
                            avg_std_dev.append(np.mean(std_values) if len(std_values) > 0 else np.nan)  

                        avg_conf_int_low = np.array(avg_metric_score) - 1.96 * np.array(avg_std_dev)  # 95% CI lower bound
                        avg_conf_int_high = np.array(avg_metric_score) + 1.96 * np.array(avg_std_dev)  # 95% CI upper bound
                    
                
                # **Extract the relative path for output filename**
                relative_path = os.path.relpath(swarm_file, workspace_dir).replace("/", "_").replace("\\", "_")


                if SWARM:
                    # **Read CSV file**
                    df = pd.read_csv(swarm_file)

                    # **Retrieve unique users and determine the reference user**
                    users = sorted(df["user"].unique())  # Get all users
                    swarm_user_ref = 1
                    swarm_users_others =  [u for u in users if u != swarm_user_ref]
                    
                    grouped = df.groupby(['user', 'splits']).agg(
                        mean_score=('auc', metric_function),  # Compute metric (e.g., variance, mean)
                        std_dev=('auc', 'std')  # Compute standard deviation for confidence interval
                    ).reset_index()
                    grouped.rename(columns={'mean_score': METRIC_VAR, 'std_dev': 'std_dev'}, inplace=True)  # Add standard deviation                        

                    # **All Nodes - Using Top X-Axis (Reversed Order)**
                    swarm_nodes = [u for u in users if u in grouped['user'].unique()]
                    if len(swarm_nodes) > 0:
                        swarm_splits_ref = sorted(df[df['user'] == swarm_user_ref]['splits'].unique()) 
                        swarm_splits_compl = sorted(df[df['user'] != swarm_user_ref]['splits'].unique(), reverse=True) 

                        print("\nSwarm_splits ref:")
                        print(swarm_splits_ref)

                        print("\nSwarm_splits_compl:")
                        print(swarm_splits_compl)

                        # Compute Average Metric Score for Other Nodes Instead of Plotting Each One
                        swarm_avg_metric_score = []
                        swarm_avg_std_dev = []

                        # Case 1: No reference splits are present
                        if len(swarm_splits_ref) == 0:
                            print("No reference splits found. Using complementary nodes only.")
                            for j in range(len(swarm_splits_compl)):
                                compl_rows = grouped[(grouped['user'] != swarm_user_ref) & (grouped['splits'] == swarm_splits_compl[j])]
                                if compl_rows.empty:
                                    combined_metric = np.nan
                                    combined_std = np.nan
                                else:
                                    # Since there's no reference value, simply use the complementary nodes' average
                                    combined_metric = compl_rows[METRIC_VAR].mean()
                                    combined_std = compl_rows['std_dev'].mean()


                                swarm_avg_metric_score.append(combined_metric)
                                swarm_avg_std_dev.append(combined_std)

                                print("\nsplit_others:", swarm_splits_compl[j])
                                print("\nswarm_split_value others:", compl_rows)
                                print("\ncombined metric (compl only):", combined_metric)

                        # Case 2: Both reference and complementary splits are present and have the same length
                        elif len(swarm_splits_ref) == len(swarm_splits_compl):
                            
                            ref_splits = swarm_splits_ref
                            
                            for j in range(len(swarm_splits_compl)):
                                # Get the reference row for the j-th split
                                ref_row = grouped[(grouped['user'] == swarm_user_ref) & (grouped['splits'] == swarm_splits_ref[j])]
                                if ref_row.empty:
                                    # If for some reason the reference value is missing for this split, skip it
                                    swarm_avg_metric_score.append(np.nan)
                                    swarm_avg_std_dev.append(np.nan)
                                    continue

                                ref_metric = ref_row[METRIC_VAR].values[0]
                                ref_std = ref_row['std_dev'].values[0]

                                # Get the complementary rows for the corresponding split
                                compl_rows = grouped[(grouped['user'] != swarm_user_ref) & (grouped['splits'] == swarm_splits_compl[j])]
                                compl_metric = compl_rows[METRIC_VAR].mean()  # Average over all complementary nodes
                                compl_std = compl_rows['std_dev'].mean()

                                # Weight the complementary nodes by their count.
                                # Total users = 1 (ref) + number of complementary nodes.
                                combined_metric = (ref_metric + compl_metric * len(swarm_users_others)) / len(users)
                                # Combine standard deviations assuming independent errors:
                                combined_variance = (ref_std**2 + len(swarm_users_others) * compl_std**2) / (len(users)**2)
                                combined_std = np.sqrt(combined_variance)

                                swarm_avg_metric_score.append(combined_metric)
                                swarm_avg_std_dev.append(combined_std)

                                print("\nsplit_ref:", swarm_splits_ref[j])
                                print("\nsplit_others:", swarm_splits_compl[j])
                                print("\nswarm_split_value ref:", ref_row)
                                print("\nswarm_split_value others:", compl_rows)
                                print("\ncombined metric:", combined_metric)
                        else:
                            print("Reference and complementary splits are of unequal lengths.")

                        # Optionally, compute 95% confidence intervals
                        swarm_avg_conf_int_low = np.array(swarm_avg_metric_score) - 1.96 * np.array(swarm_avg_std_dev)
                        swarm_avg_conf_int_high = np.array(swarm_avg_metric_score) + 1.96 * np.array(swarm_avg_std_dev)

                        print("\nSwarm Average Metric Scores:", np.array(swarm_avg_metric_score).tolist())
                        print("\nSwarm Average Standard Deviations:", np.array(swarm_avg_std_dev).tolist())
                        print("\n95% Confidence Intervals Low:", swarm_avg_conf_int_low)
                        print("\n95% Confidence Intervals High:", swarm_avg_conf_int_high)
                

                if CENTRAL:
                    # **Read CSV file**
                    df = pd.read_csv(central_file)

                    if reference:

                        # **Compute the Central Score (Just a Mean Value)**
                        central_score = metric_function(df["auc"])  # Compute the average AUC for the central model
                    
                    
                        # **Compute Standard Deviation & Confidence Interval (If Needed)**
                        if CONFIDENCE:
                            central_std_dev = np.std(df["auc"]) if "auc" in df.columns else 0
                            central_conf_int_low = central_score - 1.96 * central_std_dev  # 95% CI lower bound
                            central_conf_int_high = central_score + 1.96 * central_std_dev  # 95% CI upper bound

                        ### PRINTING FOR DEBUG PURPOSES
                        if DEBUG:
                            print("\nCentral " + METRIC_NAME + " (Across All Data):")
                            print(f"  ✅ Central Score: {central_score:.5f}")
                            print(f"  ✅ Confidence Interval: [{central_conf_int_low:.5f}, {central_conf_int_high:.5f}]")


                    else:

                        grouped = df.groupby(['splits']).agg(
                            mean_score=('auc', metric_function),  # Compute metric (e.g., variance, mean)
                            std_dev=('auc', 'std')  # Compute standard deviation for confidence interval
                        ).reset_index()
                        grouped.rename(columns={'mean_score': METRIC_VAR, 'std_dev': 'std_dev'}, inplace=True)  # Add standard deviation

                        central_df = grouped.copy()
                        central_splits = sorted(central_df['splits'].unique()) 

                        central_score = [central_df[central_df['splits'] == split][METRIC_VAR].values[0] for split in central_splits]
                    
                        if CONFIDENCE:
                            # Compute Confidence Interval for Reference Node
                            central_std_dev = [central_df[central_df['splits'] == split]['std_dev'].values[0] for split in central_splits]  
                            central_conf_int_low = np.array(central_score) - 1.96 * np.array(central_std_dev)  # 95% CI lower bound
                            central_conf_int_high = np.array(central_score) + 1.96 * np.array(central_std_dev)  # 95% CI upper bound



                
                
                plot_data[subfolder] = {
                    "num_users": len(users),
                    "ref_user": ref_user,
                    "ref_splits": ref_splits if reference else "",
                    "ref_metric_score": ref_metric_score if reference else "",
                    "ref_conf_int_low": ref_conf_int_low if reference and CONFIDENCE else "",
                    "ref_conf_int_high": ref_conf_int_high if reference and CONFIDENCE else "",
                    "other_users": other_users,
                    "other_splits_reversed": other_splits_reversed if len(other_users) > 0 else "",
                    "other_splits_labels": other_splits_labels if len(other_users) > 0 else "",
                    "avg_metric_score": avg_metric_score if len(other_users) > 0 and not SEPARATE else "",
                    "avg_conf_int_low": avg_conf_int_low if len(other_users) > 0  and not SEPARATE and CONFIDENCE else "",
                    "avg_conf_int_high": avg_conf_int_high if len(other_users) > 0  and not SEPARATE and CONFIDENCE else "",
                    "user_metric_scores": user_metric_scores if len(other_users) > 0 and SEPARATE else "",
                    "user_conf_int_low": user_conf_int_low if len(other_users) > 0  and SEPARATE and CONFIDENCE else "",
                    "user_conf_int_high": user_conf_int_high if len(other_users) > 0  and SEPARATE and CONFIDENCE else "",
                    "swarm_splits_ref": swarm_splits_ref if SWARM else "",
                    "swarm_splits_others": swarm_splits_compl if SWARM else "",
                    "swarm_avg_metric_score": swarm_avg_metric_score if SWARM else "",
                    "swarm_avg_conf_int_low": swarm_avg_conf_int_low if SWARM and CONFIDENCE else "",
                    "swarm_avg_conf_int_high": swarm_avg_conf_int_high if SWARM and CONFIDENCE else "",
                    "central_score": central_score if CENTRAL else "",
                    "central_conf_int_low": central_conf_int_low if CENTRAL and CONFIDENCE else "",
                    "central_conf_int_high": central_conf_int_high if CENTRAL and CONFIDENCE else "",
                    "label": f"{os.path.basename(subfolder)}"
                }

            except Exception as e:
                print(f"⚠️ Error processing '{local_file, swarm_file}': {e}")
                continue


        # **Create figure**
        fig, ax = plt.subplots(figsize=(10, 7))
        # ax.set_ylim(0.4, 0.9)

        for i, (subfolder, data) in enumerate(plot_data.items()):
            shaded_ref_color = reference_color if i == 0 else mcolors.to_rgba(reference_color, alpha=0.3 + (i / len(plot_data)) * 0.4)
            shaded_swarm_color = swarm_color if i == 0 else mcolors.to_rgba(swarm_color, alpha=0.3 + (i / len(plot_data)) * 0.4)
            shaded_central_color = central_color if i == 0 else mcolors.to_rgba(central_color, alpha=0.3 + (i / len(plot_data)) * 0.4)
            shaded_avg_others_color = avg_others_color if i == 0 else mcolors.to_rgba(avg_others_color, alpha=0.3 + (i / len(plot_data)) * 0.4)


            if SWARM:
                
                if i % 2 == 0:
                    swarm_splits = data["swarm_splits_ref"]
                    ax.plot(swarm_splits, data["swarm_avg_metric_score"], marker='o', linestyle='-', color=shaded_swarm_color, label=f'Swarm Learning')

                else:
                    ax.plot(swarm_splits, data["swarm_avg_metric_score"], marker='s', linestyle='-', color=shaded_swarm_color, label=f'Swarm Learning')
                print("Swarm ")
                print(swarm_splits)
                print(data["swarm_avg_metric_score"])

                if CONFIDENCE:
                    ax.fill_between(swarm_splits, data["swarm_avg_conf_int_low"], data["swarm_avg_conf_int_high"], color=shaded_swarm_color, alpha=0.2, label=fr'$\pm$ 95% CI (SL)')
                # else:
                #     ax.plot(data["swarm_splits"], data["swarm_avg_metric_score"], marker='o', linestyle='-', color=shaded_swarm_color, label=f'Swarm Learning')

                #     if CONFIDENCE:
                #         ax.fill_between(data["other_splits_reversed"], data["swarm_avg_conf_int_low"], data["swarm_avg_conf_int_high"], color=shaded_swarm_color, alpha=0.2, label=fr'$\pm$ 95% CI (SL)')


            if CENTRAL:
                if i % 2 == 0:
                    central_splits = data["ref_splits"]
                    ax.axhline(y=data["central_score"], color=shaded_central_color, linestyle=':', label='Central Model')
                else:
                    data["central_score"] = data["central_score"][::-1]
                    ax.plot(central_splits, data["central_score"], marker='s', linestyle=':', color=shaded_central_color, label=f'Central Model')

                if CONFIDENCE:
                    if i % 2 == 0:
                        # Generate repeated values for plotting confidence intervals
                        central_conf_low_array = np.full_like(central_splits, data["central_conf_int_low"], dtype=np.float64)  # Repeat lower bound
                        central_conf_high_array = np.full_like(central_splits, data["central_conf_int_high"], dtype=np.float64)  # Repeat upper bound
                        ax.fill_between(data["ref_splits"], central_conf_low_array, central_conf_high_array, color=shaded_central_color, alpha=0.2, label=fr'$\pm$ 95% CI (Central)')
                    else:
                        central_conf_low_array = np.full_like(central_splits, data["central_conf_int_low"], dtype=np.float64)  # Repeat lower bound
                        central_conf_high_array = np.full_like(central_splits, data["central_conf_int_high"], dtype=np.float64)  # Repeat upper bound
                        ax.fill_between(central_splits[::-1], central_conf_low_array, central_conf_high_array, color=shaded_central_color, alpha=0.2, label=fr'$\pm$ 95% CI (Central)')
            
            
            if i % 2 == 0:
                ref_splits = data["ref_splits"]
                ax.plot(ref_splits, data["ref_metric_score"], marker='o', linestyle='-', color=shaded_ref_color, label="Node " + str(data["ref_user"]) + " (New)")
                print("REF ")
                print(ref_splits)
                print(data["ref_metric_score"])

                if CONFIDENCE:
                    ax.fill_between(ref_splits, data["ref_conf_int_low"], data["ref_conf_int_high"], color=shaded_ref_color, alpha=0.2, label=fr'$\pm$ 95% CI - New')

            # **Plot Other Users**
            if len(data["other_users"]) > 0:
                #other_splits_reversed =  [98 - x * len(data["other_users"]) for x in data["other_splits_reversed"]]
                #other_splits_reversed = other_splits_reversed[::-1]
                other_splits_reversed = ref_splits[::-1]
                
                if SEPARATE:
                    for user in data["other_users"]:
                        if i % 2 == 0:
                            ax.plot(other_splits_reversed, data["user_metric_scores"][user], marker='o', linestyle='-.', label=f'Node {user}')
                        else:
                            ax.plot(other_splits_reversed, data["user_metric_scores"][user], marker='s', linestyle='-.', label=f'Node {user}')

                        print("Other separate ")
                        print(data["user_metric_scores"])
                        print(other_splits_reversed)


                        if CONFIDENCE:
                            ax.fill_between(other_splits_reversed, data["user_conf_int_low"][user], data["user_conf_int_high"][user], alpha=0.2, label=fr'$\pm$ 95% CI (Node {user})')

                else:
                    # Plot Only One Line for Other Nodes (Average Metric Score)
                    if i % 2 == 0:
                        ax.plot(other_splits_reversed, data["avg_metric_score"], marker='o', linestyle='-.', color=shaded_avg_others_color, label='Avg. Other Nodes')
                    else:
                        ax.plot(other_splits_reversed, data["avg_metric_score"], marker='s', linestyle='-.', color=shaded_avg_others_color, label='Avg. Other Nodes')

                    print("Other AVG ")
                    print(data["avg_metric_score"])
                    print(other_splits_reversed)
                    
                    if CONFIDENCE:
                        ax.fill_between(other_splits_reversed, data["avg_conf_int_low"], data["avg_conf_int_high"], color=shaded_avg_others_color, alpha=0.2, label=fr'$\pm$ 95% CI (Avg. Ohers)')

        # **Configure Bottom X-Axis (Reference Node Splits)**
        ax.set_xticks(ref_splits)
        ax.set_xticklabels([f'{val:.5g}' for val in ref_splits], rotation=0, ha="right")  
        ax.set_xlabel(f"New Node Data Split (%)")
        ax.set_ylabel(METRIC_NAME)  # Use dynamic metric name
        #ax.set_title(f"{METRIC_NAME} vs. Data Split per Node ({len(users)} Nodes)", fontsize=14, pad=55)
        ax.set_title(f"{METRIC_NAME} (New Node + {len(users)} Nodes)", pad=20)
        plt.subplots_adjust(top=1.85)  

        # **Configure Secondary X-Axis (Other Users' Splits - Top Axis)**
        if len(data["other_users"]) > 0:
            ax2 = ax.twiny()
            ax2.set_xlim(ax.get_xlim())  
            ax2.set_xticks(ref_splits)  
            ax2.set_xticklabels([f'{val:.5g}' for val in data["other_splits_labels"]], rotation=30, ha="left")  
            ax2.set_xlabel("Other Nodes' Data Split (%)")

        # **Add vertical grid lines for better visualization**
        for x in ref_splits:
            ax.axvline(x, color='gray', linestyle='--', alpha=0.5)

        # Get the number of labels in the legend
        num_labels = len(ax.get_legend_handles_labels()[1])  # Extract legend labels

        # Set ncol dynamically: 1 column if 4 or fewer labels, 2 columns otherwise
        ncol_value = 2 if num_labels > 6 else 1

        # **Add legend**
        ax.legend(loc='best', ncol=ncol_value)

        # Add text in the bottom-right corner
        plt.gcf().text(0.98, 0.02, main_folder, ha='right', va='bottom')

        # **Generate Output File Path**
        #output_filename = f"{METRIC_FILENAME}_{relative_path.replace('.csv', '_avg_conf.pdf')}"
        confidence_suffix = "_conf" if CONFIDENCE else ""
        separate_suffix = "_separate" if SEPARATE else "_avg"
        swarm_suffix = "_swarm" if SWARM else ""
        central_suffix = "_cent" if CENTRAL else ""
        output_filename = f"[{main_folder}]_{METRIC_FILENAME}{'_'}{'_'.join(pair)}{separate_suffix}{swarm_suffix}{confidence_suffix}{central_suffix}.pdf"
        output_path = os.path.join(output_dir, output_filename)

        # **Save the figure**
        plt.tight_layout()
        plt.savefig(output_path, format="pdf", bbox_inches="tight")
        print(f"\n✅ Plot saved as '{output_path}'.")
        plt.close()  

            



# Iterate through each combination and run the experiment
for combination in COMBINATIONS:
    flag_values = dict(zip(KEYS, combination))
    run_experiment(**flag_values)

