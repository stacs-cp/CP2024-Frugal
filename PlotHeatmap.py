import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os
import matplotlib.gridspec as gridspec

from math import pi
from matplotlib.colors import LinearSegmentedColormap

def plot_heatmap(df, approaches, filename, fig_size =(30,40), size = 22):
    data = df.copy()
   
    data = data[data["approach"].isin(approaches)]
    data['approach'] = pd.Categorical(data['approach'], categories=approaches, ordered=True)

    pivot_df = data.groupby(['dataset_name', 'approach'], observed=True)['min_cost'] \
                   .agg(['mean', 'std']).reset_index()
    pivot_mean = pivot_df.pivot(index='approach', columns='dataset_name', values='mean')
    pivot_std = pivot_df.pivot(index='approach', columns='dataset_name', values='std')
    annot_matrix = pivot_mean.round(1).astype(str) + "\n±" + pivot_std.round(1).astype(str)

    pivot_df_all = data.groupby(['approach'], observed=True)['min_cost'] \
                       .agg(['mean', 'std']).reset_index()

    fig = plt.figure(figsize=fig_size)
    gs = gridspec.GridSpec(1, 2, width_ratios=[20, 2], wspace=0.05)

    normalized_mean = pivot_mean.copy()
    for col in normalized_mean.columns:
        min_val = normalized_mean[col].min()
        max_val = normalized_mean[col].max()

        if pd.notna(min_val) and max_val > min_val:
            normalized_mean[col] = (normalized_mean[col] - min_val) / (max_val - min_val)
        else:
            normalized_mean[col] = 0.5
            
    ax0 = plt.subplot(gs[0])
    sns.heatmap(normalized_mean, ax=ax0, annot=annot_matrix, fmt="", cmap="Blues_r", cbar=False,
                annot_kws={"size": size}, vmin=0, vmax=1)
    
    ax0.set_ylabel("Selection Method", fontsize=size)
    ax0.set_xlabel("Dataset", fontsize=size)
    ax0.tick_params(axis='both', labelsize=size)
    ax0.tick_params(axis='x', labelsize=size, rotation=60)
    ax0.tick_params(axis='y', labelsize=size)

    plt.savefig(filename, format="pdf", bbox_inches='tight')
    
if not os.path.exists(r"CSV_FILES\normalized_results.csv"):
    print(r"CSV_FILES\normalized_results.csv not found. Run PlotExperimentalResults.py to generate it")

normalized_df = pd.read_csv(r"CSV_FILES\normalized_results.csv")

lexical_folder_name = "LEXICOGRAPHIC_SLURM_FILES"
pareto_folder_name = "PARETO_SLURM_FILES"
random_folder_name = "RANDOM_SLURM_FILES"
uncertainty_folder_name = "UNCERTAINTY_SLURM_FILES"

normalized_df["approach"] = normalized_df["approach"].str.replace(
    r" ?\(ONLY TO PEN(?: \+ DTL)?\)", lambda m: " (DTL)" if "+ DTL" in m.group(0) else "", regex=True)

normalized_df = normalized_df[normalized_df["min_gap"] == 1]
normalized_df["min_cost"] = normalized_df["min_cost"] * 100

approach_map = {
    f"{lexical_folder_name} (PredCost -> PredDiff -> Uncertainty)": "LEXICAL (PC → PD → U)",
    f"{lexical_folder_name} (PredCost -> Uncertainty -> PredDiff)": "LEXICAL (PC → U → PD)",
    f"{lexical_folder_name} (PredDiff -> PredCost -> Uncertainty)": "LEXICAL (PD → PC → U)",
    f"{lexical_folder_name} (PredDiff -> Uncertainty -> PredCost)": "LEXICAL (PD → U → PC)",
    f"{lexical_folder_name} (Uncertainty -> PredCost -> PredDiff)": "LEXICAL (U → PC → PD)",
    f"{lexical_folder_name} (Uncertainty -> PredDiff -> PredCost)": "LEXICAL (U → PD → PC)",

    f"{lexical_folder_name} (PredCost -> PredDiff -> Uncertainty) (DTL)": "LEXICAL (PC → PD → U) (DTL)",
    f"{lexical_folder_name} (PredCost -> Uncertainty -> PredDiff) (DTL)": "LEXICAL (PC → U → PD) (DTL)",
    f"{lexical_folder_name} (PredDiff -> PredCost -> Uncertainty) (DTL)": "LEXICAL (PD → PC → U) (DTL)",
    f"{lexical_folder_name} (PredDiff -> Uncertainty -> PredCost) (DTL)": "LEXICAL (PD → U → PC) (DTL)",
    f"{lexical_folder_name} (Uncertainty -> PredCost -> PredDiff) (DTL)": "LEXICAL (U → PC → PD) (DTL)",
    f"{lexical_folder_name} (Uncertainty -> PredDiff -> PredCost) (DTL)": "LEXICAL (U → PD → PC) (DTL)",

    f"{pareto_folder_name} (Uncertainty - PredCost)": "PARETO (U - PC)",
    f"{pareto_folder_name} (Uncertainty - PredDiff)": "PARETO (U - PD)",
    f"{pareto_folder_name} (PredDiff - PredCost)": "PARETO (PD - PC)",
    f"{pareto_folder_name} (Uncertainty - PredDiff - PredCost)": "PARETO (U - PD - PC)",

    f"{pareto_folder_name} (Uncertainty - PredCost) (DTL)": "PARETO (U - PC) (DTL)",
    f"{pareto_folder_name} (Uncertainty - PredDiff) (DTL)": "PARETO (U - PD) (DTL)",
    f"{pareto_folder_name} (PredDiff - PredCost) (DTL)": "PARETO (PD - PC) (DTL)",
    f"{pareto_folder_name} (Uncertainty - PredDiff - PredCost) (DTL)": "PARETO (U - PD - PC) (DTL)",

    f"{random_folder_name}": "RANDOM",
    f"{random_folder_name} (DTL + TO)": "RANDOM (DTL + TO)",
    f"{random_folder_name} (DTL)": "RANDOM (DTL)",
    f"{random_folder_name} (TO)": "RANDOM (TO)",

    f"{uncertainty_folder_name}": "UNCERTAINTY",
    f"{uncertainty_folder_name} (DTL + TO)": "UNCERTAINTY (DTL + TO)",
    f"{uncertainty_folder_name} (DTL)": "UNCERTAINTY (DTL)",
    f"{uncertainty_folder_name} (TO)": "UNCERTAINTY (TO)",
}

normalized_df["approach"] = normalized_df["approach"].replace(approach_map)

plot_heatmap(normalized_df, ["UNCERTAINTY", "RANDOM", "PARETO (U - PC)", "PARETO (U - PD)", "PARETO (PD - PC)", "PARETO (U - PD - PC)",
                            "LEXICAL (U → PC → PD)", "LEXICAL (U → PD → PC)", "LEXICAL (PD → U → PC)", "LEXICAL (PD → PC → U)", "LEXICAL (PC → U → PD)", "LEXICAL (PC → PD → U)",
                            "UNCERTAINTY (TO)", "RANDOM (TO)", "UNCERTAINTY (DTL)", "RANDOM (DTL)", "UNCERTAINTY (DTL + TO)", "RANDOM (DTL + TO)",
                            "PARETO (U - PC) (DTL)", "PARETO (U - PD) (DTL)", "PARETO (PD - PC) (DTL)", "PARETO (U - PD - PC) (DTL)",
                            "LEXICAL (U → PC → PD) (DTL)", "LEXICAL (U → PD → PC) (DTL)", "LEXICAL (PD → U → PC) (DTL)", "LEXICAL (PD → PC → U) (DTL)", "LEXICAL (PC → U → PD) (DTL)", "LEXICAL (PC → PD → U) (DTL)"
                            ], "heatmap_all_configs.pdf", fig_size =(30,40))

plot_heatmap(normalized_df, ["UNCERTAINTY", "UNCERTAINTY (TO)", "UNCERTAINTY (DTL)", "UNCERTAINTY (DTL + TO)",
                             "RANDOM", "RANDOM (TO)", "RANDOM (DTL)", "RANDOM (DTL + TO)",
                             "PARETO (U - PC)", "PARETO (U - PC) (DTL)",
                             "LEXICAL (U → PC → PD)", "LEXICAL (U → PC → PD) (DTL)", 
                            ], "heatmap_representative_configs.pdf", fig_size =(30,20))