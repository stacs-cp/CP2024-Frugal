import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.ticker import FuncFormatter
sns.set(style="whitegrid")


def plot_ablation_study_aggregated(df, base_variants, filename, dtl_suffix=" (DTL)", bar_width=0.35):
    df["is_dtl"] = df["approach"].apply(
        lambda x: "DTL" if x.endswith(" (DTL)") else "Vanilla")
    df["base_variant"] = df["approach"].str.replace(
        r" \(DTL\)", "", regex=True)

    df["group_variant"] = df["base_variant"]

    df["group_variant"] = df["group_variant"].replace({
        **{k: "LEXICAL" for k in df["group_variant"].unique() if k.startswith("LEXICAL")},
        **{k: "PARETO" for k in df["group_variant"].unique() if k.startswith("PARETO")}
    })

    all_variants = base_variants + [v + dtl_suffix for v in base_variants]
    df = df[df["approach"].isin(all_variants)]

    df.loc[:, "base_variant"] = pd.Categorical(
        df["base_variant"], categories=base_variants, ordered=True)

    plt.figure(figsize=(max(10, len(base_variants) * 0.7), 6))
    sns.set_palette("colorblind")

    ax = sns.boxplot(
        data=df,
        x="base_variant",
        y="min_cost",
        hue="is_dtl",
        order=base_variants,
        hue_order=["DTL", "Vanilla"],
        width=0.6,
        showfliers=False,
    )

    medians = df.groupby(['group_variant', 'is_dtl'])['min_cost'].median()

    vertical_offset = df['min_cost'].median() * 0.05

    for xtick, label in zip(ax.get_xticks(), ax.get_xticklabels()):
        group_label = label.get_text().replace('\n', ' ')

        for i, hue in enumerate(['Vanilla', 'DTL']):
            try:
                median_value = medians[(group_label, hue)]

                hue_offset = (0.15 if hue == 'Vanilla' else -0.15)
                color = "black" if hue == "Vanilla" else "white"
                ax.text(
                    xtick + hue_offset,
                    median_value + vertical_offset,
                    f"{median_value:.1f}",
                    horizontalalignment='center',
                    fontsize=11,
                    color=color,
                )
            except KeyError:
                continue

    wrapped_labels = [
        label.replace("LEXICAL", "LEXICAL\n").replace("PARETO", "PARETO\n")
        for label in base_variants
    ]

    plt.ylabel("Labelling Cost (%)", fontsize=14)
    plt.xlabel("a")
    plt.xticks(ha='center', fontsize=12)
    plt.yticks(fontsize=12)
    plt.ylim(0, 101)
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.legend(title="", loc='upper center', bbox_to_anchor=(
        0.5, 1.1), ncol=2, fontsize=13, frameon=False)
    plt.savefig(filename, format="pdf")


def plot_ablation_study_pareto(df, strategies, filename, base_prefix="PARETO", bar_width=0.20):
    df = df.copy()
    suffixes = ["", " (DTL)"]
    x_labels = ["Pareto (No DTL)", "Pareto (DTL)"]
    x_pos = np.arange(len(x_labels))
    palette = sns.color_palette("colorblind", n_colors=len(strategies))

    plt.figure(figsize=(10, 6))

    total_width = bar_width * len(strategies)
    offsets = np.linspace(-total_width / 2 + bar_width / 2,
                          total_width / 2 - bar_width / 2, len(strategies))

    baseline_no_dtl = df[df["approach"] ==
                         f"{base_prefix} (U - PC)"]["min_cost"].mean()
    baseline_dtl = df[df["approach"] ==
                      f"{base_prefix} (U - PC) (DTL)"]["min_cost"].mean()

    for idx, strategy in enumerate(strategies):
        costs = []
        deltas = []

        for suffix in suffixes:
            full_label = f"{base_prefix} ({strategy}){suffix}"
            val = df[df["approach"] == full_label]["min_cost"].mean()

            costs.append(val)

            if suffix == "":
                delta = val - \
                    baseline_no_dtl if not pd.isna(val) and not pd.isna(
                        baseline_no_dtl) else np.nan
            else:
                delta = val - \
                    baseline_dtl if not pd.isna(val) and not pd.isna(
                        baseline_dtl) else np.nan
            deltas.append(delta)

        bar_positions = x_pos + offsets[idx]
        plt.bar(bar_positions, costs, width=bar_width,
                color=palette[idx], label=strategy)

        for x, y, d in zip(bar_positions, costs, deltas):
            if not pd.isna(y) and not pd.isna(d):
                if delta != 0:
                    plt.text(x, y + 1.5, f"Δ={d:.1f}",
                             ha='center', fontsize=13)

    plt.xticks(x_pos, x_labels, fontsize=13)
    plt.ylabel("Labelling Cost (%)", fontsize=13)
    plt.ylim(0, 100)
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.legend(fontsize=13, title_fontsize=12,
               loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=4, frameon=False)

    plt.tight_layout()
    plt.savefig(filename, format="pdf")


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
normalized_df_copy = normalized_df.copy()

approach_map_2 = {
    "LEXICAL (PC → PD → U)": "LEXICAL",
    "LEXICAL (PC → U → PD)": "LEXICAL",
    "LEXICAL (PD → PC → U)": "LEXICAL",
    "LEXICAL (PD → U → PC)": "LEXICAL",
    "LEXICAL (U → PC → PD)": "LEXICAL",
    "LEXICAL (U → PD → PC)": "LEXICAL",

    "LEXICAL (PC → PD → U) (DTL)": "LEXICAL (DTL)",
    "LEXICAL (PC → U → PD) (DTL)": "LEXICAL (DTL)",
    "LEXICAL (PD → PC → U) (DTL)": "LEXICAL (DTL)",
    "LEXICAL (PD → U → PC) (DTL)": "LEXICAL (DTL)",
    "LEXICAL (U → PC → PD) (DTL)": "LEXICAL (DTL)",
    "LEXICAL (U → PD → PC) (DTL)": "LEXICAL (DTL)",

    "PARETO (U - PC)": "PARETO",
    "PARETO (U - PD)": "PARETO",
    "PARETO (PD - PC)": "PARETO",
    "PARETO (U - PD - PC)": "PARETO",

    "PARETO (U - PC) (DTL)": "PARETO (DTL)",
    "PARETO (U - PD) (DTL)": "PARETO (DTL)",
    "PARETO (PD - PC) (DTL)": "PARETO (DTL)",
    "PARETO (U - PD - PC) (DTL)": "PARETO (DTL)"
}

normalized_df_copy["approach"] = normalized_df_copy["approach"].replace(
    approach_map_2)

# Ablation Study of DTL
plot_ablation_study_aggregated(normalized_df_copy, filename="box_plot_ablation_dtl.pdf", base_variants=[
                               "PARETO", "LEXICAL", "UNCERTAINTY", "RANDOM"])

# Ablation Study of Selection Methods on Pareto
plot_ablation_study_pareto(
    normalized_df,
    strategies=["U - PC", "U - PD", "PD - PC", "U - PD - PC"], filename="pareto_ablation_study.pdf"
)
