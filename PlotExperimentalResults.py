import math
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict
import matplotlib.ticker as ticker
import seaborn as sns
import matplotlib.pyplot as plt
import textwrap
import re

import warnings
warnings.filterwarnings("ignore")


def normalize_data(dataframe):
    t1 = dataframe[["dataset_name", "seed_number", "configuration",
                    "split_number", f"runtime", f"instance_cost"]]
    t1 = t1.sort_values(
        by=["split_number", "seed_number", f"runtime", f"instance_cost"])
    t1 = t1.rename({"seed_number": "seed", "split_number": "split",
                   f"runtime": "gap", f"instance_cost": "cost"}, axis=1)

    tNew = None
    bins = [round(0.01*x, 2) for x in list(range(101))]

    for _, (dataset, seed, split, config) in t1[["dataset_name", "seed", "split", "configuration"]].drop_duplicates().iterrows():
        t2 = t1[(t1.seed == seed) & (t1.split == split)
                & (t1.configuration == config)]

        data = [{"dataset_name": dataset, "approach": config, "min_gap": min_gap, "min_cost": t2[t2.gap >=
                                                                                                 min_gap].cost.values.min() if len(t2[t2.gap >= min_gap].index) > 0 else np.nan} for min_gap in bins]
        data = pd.DataFrame(data)

        data["seed"] = seed
        data["split"] = split
        if tNew is None:
            tNew = data
        else:
            tNew = pd.concat([tNew, data], axis=0)
    tNew = tNew.reset_index(drop=True)

    return tNew


def normalize_data_one_point(df):
    t1_concat = pd.DataFrame()

    for dataset_index, dataset in enumerate(df["dataset_name"].unique()):
        print(f"Normalizing dataset {dataset}")

        dataset_df = df[df["dataset_name"] == dataset]
        for approach_index, approach in enumerate(dataset_df["configuration"].unique()):
            approach_df = dataset_df[dataset_df["configuration"] == approach]
            normalize_df = normalize_data(approach_df)
            t1_concat = pd.concat([t1_concat, normalize_df])

    return t1_concat


def plot_line_chart_all_datasets(df, approach_list, keyword):
    df["min_gap"] = df["min_gap"] * 100
    df["min_cost"] = df["min_cost"] * 100
    n_datasets = len(df["dataset_name"].unique())
    subset = df[df["approach"].isin(approach_list)]

    n_cols = 4
    n_rows = math.ceil(n_datasets / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(
        20, 5 * n_rows))

    axes = axes.flatten()
    sns.set_palette("colorblind")

    for dataset_index, dataset in enumerate(subset["dataset_name"].unique()):
        dataset_df = subset[subset["dataset_name"] == dataset]
        ax = axes[dataset_index]

        sns.lineplot(
            data=dataset_df, x="min_gap", y="min_cost",
            hue="approach", style="approach", ax=ax, errorbar="se",
            dashes=False, linewidth=2
        )
        ax.set_title(f"{dataset}", fontsize=20)
        ax.set_ylabel("Labelling Cost (%)", fontsize=18)
        ax.set_xlabel("AS Performance (%)", fontsize=18)
        ax.tick_params(axis='both', labelsize=18)

        ax.grid(True)

        ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    for i in range(dataset_index + 1, len(axes)):
        axes[i].axis('off')

    handles, labels = axes[0].get_legend_handles_labels()
    unique_labels = list(dict(zip(labels, handles)).items()
                         )

    for label, handle in unique_labels:
        handle.set_linewidth(5.0)

    for ax in fig.axes:
        legend = ax.get_legend()
        if legend is not None:
            legend.set_visible(False)

    fig.legend(
        [h for _, h in unique_labels], [l for l, _ in unique_labels],
        title_fontsize=20, fontsize=18,
        loc="upper center", bbox_to_anchor=(0.5, 1.02), ncol=6, frameon=False
    )

    plt.tight_layout()
    plt.savefig(
        f"line_chart_{keyword}.pdf", bbox_inches="tight")


def plot_approaches(df, keyword, filename):
    filtered_df = df[df["approach"].isin(keyword)]
    unique_approaches = filtered_df["approach"].unique()

    plot_line_chart_all_datasets(
        filtered_df, unique_approaches, filename)


def create_summary_table(filtered_df):
    def extract_selection_metric(approach):
        """Extracts the selection metric from the approach string."""
        match = re.search(r"LEXICAL \((.*?)\)|PARETO \((.*?)\)", approach)
        if match:
            return match.group(1) or match.group(2)
        return None

    def extract_configuration(approach):
        """Extracts the configuration part from the approach string (DTL, TO, etc.)."""
        match = re.findall(r"\b(DTL|TO)\b", approach)
        return " + ".join(match) if match else None

    summary_df = (
        filtered_df[filtered_df["min_gap"] == 1].groupby("approach")
        .agg(
            min_cost=("min_cost", "min"),
            mean_cost=("min_cost", "mean"),
            max_cost=("min_cost", "max"),
            std_cost=("min_cost", "std"),
            median_cost=("min_cost", "median")
        )
        .reset_index()
        .sort_values("mean_cost", ascending=True)
    )

    summary_df["min_cost"] = summary_df["min_cost"].round(2)
    summary_df["mean_cost"] = summary_df["mean_cost"].round(2)
    summary_df["max_cost"] = summary_df["max_cost"].round(2)
    summary_df["std_cost"] = summary_df["std_cost"].round(2)
    summary_df["median_cost"] = summary_df["median_cost"].round(2)

    summary_df["selection_metric"] = summary_df["approach"].apply(
        extract_selection_metric)
    summary_df["configuration"] = summary_df["approach"].apply(
        extract_configuration)

    summary_df["selection_metric"] = summary_df["selection_metric"].fillna(
        "Uncertainty")
    summary_df["configuration"] = summary_df["configuration"].fillna("-")
    summary_df["selection_metric"] = summary_df["selection_metric"].str.replace(
        "->", "-")
    summary_df["approach"] = summary_df["approach"].str.extract(r"(\w+)")

    summary_df.to_csv("summary_table.csv", index=False)


if not Path(r"CSV_FILES\normalized_results.csv").exists():
    print("Creating normalized results file...")

    if not Path(r'CSV_FILES\merged_results.csv').exists():
        print(r"CSV_FILES\merged_results.csv not found. In order to create this file you must run ExperimentalStatuspy")
        exit()

    merged_df = pd.read_csv(r'CSV_FILES\merged_results.csv')

    normalized_df = normalize_data_one_point(merged_df)
    normalized_df.to_csv(r"CSV_FILES\normalized_results.csv", index=False)

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
# Line chart for all datasets
plot_approaches(normalized_df, [
                "PARETO (U - PC) (DTL)", "UNCERTAINTY", "RANDOM"], filename="best_pareto_vs_baselines")

plot_approaches(normalized_df, [
    "LEXICAL (PC → PD → U) (DTL)", "PARETO (U - PC) (DTL)", "UNCERTAINTY", "RANDOM"], filename="best_pareto_lexical_vs_baselines")
