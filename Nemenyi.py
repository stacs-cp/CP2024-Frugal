import Orange
import numpy as np
import pandas as pd
import os
from Orange.evaluation import graph_ranks
import matplotlib.pyplot as plt
from scipy.stats import friedmanchisquare

if not os.path.exists(r"CSV_FILES\normalized_results.csv"):
    print(r"CSV_FILES\normalized_results.csv not found. Run PlotExperimentalResults.py to generate it")

normalized_df = pd.read_csv(r"CSV_FILES\normalized_results.csv")

lexical_folder_name = "LEXICOGRAPHIC_SLURM_FILES"
pareto_folder_name = "PARETO_SLURM_FILES"
random_folder_name = "RANDOM_SLURM_FILES"
uncertainty_folder_name = "UNCERTAINTY_SLURM_FILES"

normalized_df["approach"] = normalized_df["approach"].str.replace(
    r" ?\(ONLY TO PEN(?: \+ DTL)?\)", lambda m: " (DTL)" if "+ DTL" in m.group(0) else "", regex=True)

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

# normalized_df = normalized_df[normalized_df["min_gap"] == 1]
normalized_df = normalized_df[normalized_df["min_gap"] == 0.25]
normalized_df["min_cost"] = normalized_df["min_cost"] * 100

pivot = normalized_df.pivot_table(
    index='dataset_name',
    columns='approach',
    values='min_cost',
    aggfunc='mean'
)

selected_approaches = normalized_df["approach"].unique().tolist()
normalized_df["min_cost"] = normalized_df["min_cost"] * 100

stat, p = friedmanchisquare(*[pivot[col] for col in pivot.columns])
print(f"Friedman test statistic = {stat:.4f}, p-value = {p:.4f}")

for value in [0.25, 0.5, 0.75, 1.0]:
    print(f"Plotting CD diagram for prediction power = {value * 100}%")
    normalized_df = normalized_df[(normalized_df["min_gap"] == 0.25)]

    ranks = pivot.rank(axis=1, method='average', ascending=True)
    avg_ranks = ranks.mean()

    graph_ranks(avg_ranks.values, names=avg_ranks.index.tolist(), cd=1.0, width=8) 
    plt.savefig(f"cd_diagram_{value}.pdf", dpi=300, bbox_inches='tight', format='pdf')