from pathlib import Path
import re
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings("ignore")

SLURM_FOLDER_PATH_Q1_AL = Path("Q1_AL_SLURM_FILES")
SLURM_FOLDER_PATH_Q1_RQ = Path("Q1_RQ_SLURM_FILES")
CSV_SAVE_PATH = Path("CSV_FILES")

def add_parentheses_if_not_empty(s):
    return f" ({s})" if s else ""

def create_pl_boxplot(experiment_df, set_type):
    # Define the path for saving plots and ensure the directory exists
    box_plot_path = Path("PLOTS2")
    box_plot_path.mkdir(parents=True, exist_ok=True)

    # Sort the dataframe to ensure consistency in the plots
    experiment_df.sort_values(by=["dataset_name", "seed_number", "timeout_predictor_usage"], inplace=True)

    # Melt the DataFrame to have a suitable form for seaborn's boxplot
    melted_df = pd.melt(experiment_df, 
                        id_vars=["dataset_name", "seed_number", "timeout_predictor_usage", "timeout_limit", 
                                 "query_size", "split_number", "query_number", "passive_learning_instance_cost"], 
                        var_name="approach", value_name="runtime")

    replacements = {
        f'vbs_{set_type}': 'VBS', 
        f'sbs_{set_type}': 'SBS', 
        f'passive_learning_runtime_{set_type}': 'passive learning'
    }
    melted_df['approach'] = melted_df['approach'].replace(replacements)

    melted_df["config"] = melted_df["timeout_predictor_usage"].map({
        "No": "", 
        "Yes": "TO"
    }).fillna("Unknown Config")

    melted_df["config"] = "Passive Learning" + melted_df["config"].apply(add_parentheses_if_not_empty)
    melted_df["config"] = melted_df["config"].astype(str).str.replace(r'_+\s*$', '', regex=True)

    plt.figure(figsize=(30, 15))
    sns.set(style="whitegrid")
    plt.rcParams.update({'axes.labelsize': 40, 'xtick.labelsize': 40, 'ytick.labelsize': 40})

    sns.set_palette("colorblind")
    g = sns.boxplot(data=melted_df[melted_df.approach == "passive learning"], 
                    x="dataset_name", y="runtime", hue="config")

    g.set_xlabel(None)
    g.set_ylabel("Normalized Runtime", fontsize=40)
    g.legend_.set_title(None)
    g.legend(fontsize=40, frameon=False)
    plt.setp(g.get_legend().get_texts(), fontsize='40')
    g.set_ylim(0, 3)
    g.set_xticklabels(g.get_xticklabels(), rotation=45, ha="right")
    plt.tight_layout()

    plt.savefig(box_plot_path / f"boxplot_{set_type}_pl.pdf")


test_pl = pd.read_csv("CSV_FILES/test_pl_sbs_vbs_normalized.csv")

create_pl_boxplot(test_pl, "test")
