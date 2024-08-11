from pathlib import Path
import re
import os
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import cm

import warnings
warnings.filterwarnings("ignore")

CSV_PATH = Path("CSV_FILES")

val_df_q1 = pd.read_csv(f'{CSV_PATH}/val_q1_no_weight.csv')
test_df_q1 = pd.read_csv(f'{CSV_PATH}/test_q1_no_weight.csv')

def normalize_data(dataframe, learning_type, set_type):
    t1 = dataframe[["seed_number", "split_number", f"{learning_type}_runtime_{set_type}", f"{learning_type}_instance_cost"]]
    t1 = t1.sort_values(by=["split_number", "seed_number", f"{learning_type}_runtime_{set_type}", f"{learning_type}_instance_cost"])
    t1 = t1.rename({"seed_number": "seed", "split_number": "split", f"{learning_type}_runtime_{set_type}": "gap", f"{learning_type}_instance_cost": "cost"}, axis=1)

    tNew = None
    bins = [round(0.01*x,2) for x in list(range(101))]

    for _, (seed, split) in t1[["seed", "split"]].drop_duplicates().iterrows():
        t2 = t1[(t1.seed==seed) & (t1.split==split)]

        data = [{"min_gap": min_gap, "min_cost": t2[t2.gap >= min_gap].cost.values.min() if len(t2[t2.gap >= min_gap].index)>0 else np.nan} for min_gap in bins]
        data = pd.DataFrame(data)
        data["seed"] = seed
        data["split"] = split
        data["approach"] = learning_type
        if tNew is None:
            tNew = data
        else:
            tNew = pd.concat([tNew, data], axis=0)
    tNew = tNew.reset_index(drop=True)

    return tNew


def plot_min_data_selected_config_2_options(dataframe, set_type, keyword, keyword_list, filename):
    CONFIG_FILE_PATH = f"PLOTS/"
    os.makedirs(CONFIG_FILE_PATH, exist_ok=True)
    
    experiment_combs = dataframe[['timeout_predictor_usage', 'timeout_limit', 'query_size']].drop_duplicates().reset_index(drop=True)
    
    n_datasets = len(dataframe["dataset_name"].unique())
    n_cols = 3
    n_rows = (n_datasets + 2) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 6 * n_rows)) 

    axes = axes.flatten()
    
    for dataset_index, dataset in enumerate(dataframe["dataset_name"].unique()):
        dataset_df = dataframe[dataframe["dataset_name"] == dataset]
        t1_concat = pd.DataFrame()
        
        for _, comb in experiment_combs.iterrows():
            conditions = (
                (dataset_df["timeout_predictor_usage"] == comb["timeout_predictor_usage"]) &
                (dataset_df["timeout_limit"] == comb["timeout_limit"]) &
                (dataset_df["query_size"] == comb["query_size"])
            )
            t_filtered = dataset_df[conditions]
            
            for query_type in ["active_learning", "random_query"]:
                t_normalized = normalize_data(t_filtered, query_type, set_type)
                config_label = f"{t_normalized['approach'].iloc[0]}_{comb['timeout_predictor_usage']}_Timeout_predictor_{comb['timeout_limit']}"
                config_label = config_label.replace("Yes_Timeout_predictor", "Timeout_predictor").replace("No_Timeout_predictor_", "").replace("100", "Dynamic_Timeout").replace("3600", "")
                t_normalized["config"] = config_label.strip("_ ")
                t1_concat = pd.concat([t1_concat, t_normalized], ignore_index=True)
        
        t1_concat = t1_concat[t1_concat["config"].isin(keyword_list)].sort_values(by="config")

        t1_concat["config"] = t1_concat["config"].replace("active_learning_Timeout_predictor_Dynamic_Timeout", "Uncertainty (TO+DT)")
        t1_concat["config"] = t1_concat["config"].replace("random_query_Timeout_predictor_Dynamic_Timeout", "Random (TO+DT)")

        t1_concat.sort_values(by="config", inplace=True, ascending=False)

        sns.set_palette("colorblind")
        ax = axes[dataset_index]
        sns.lineplot(data=t1_concat, x="min_gap", y="min_cost", hue="config", style="config", ax=ax, errorbar="se",
                     dashes=False, linewidth=3)
        
        ax.set_ylabel(f"{dataset} \n Min Instance Cost", fontsize=25)
        ax.set_xlabel("Runtime Ratio", fontsize=25)
        ax.tick_params(axis='both', labelsize=25)
        ax.grid(True)

    for i in range(dataset_index + 1, len(axes)):
        axes[i].axis('off')

    handles, labels = [], []
    for ax in axes:
        for handle, label in zip(*ax.get_legend_handles_labels()):
            if label not in labels:
                handle.set_linewidth(5.0)
                handles.append(handle)
                labels.append(label)

    for ax in fig.axes:
        ax.get_legend().set_visible(False)

    fig.legend(handles, labels, loc='upper center', ncol=4, fontsize=25, bbox_to_anchor=(0.5, 1.0), frameon=False)
    fig.subplots_adjust(hspace=0.25, right=0.99, bottom=0.15, top=0.90, left=0.11, wspace=0.45)

    plt.savefig(f"{CONFIG_FILE_PATH}/{filename}.pdf")


def plot_min_data_selected_config_3_options(dataframe, set_type, keyword, keyword_list, filename):
    CONFIG_FILE_PATH = f"PLOTS/"
    os.makedirs(CONFIG_FILE_PATH, exist_ok=True)
    
    experiment_combs = dataframe[['timeout_predictor_usage', 'timeout_limit', 'query_size']].drop_duplicates().reset_index(drop=True)

    n_cols = 3
    n_rows = 1
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 6 * n_rows))

    ax = axes.flatten()
    t1_concat_list = [pd.DataFrame(), pd.DataFrame(), pd.DataFrame()]

    for dataset_index, dataset in enumerate(dataframe["dataset_name"].unique()):
        dataset_df = dataframe[dataframe["dataset_name"] == dataset]

        for _, comb in experiment_combs.iterrows():
            conditions = (
                (dataset_df["timeout_predictor_usage"] == comb["timeout_predictor_usage"]) &
                (dataset_df["timeout_limit"] == comb["timeout_limit"]) &
                (dataset_df["query_size"] == comb["query_size"])
            )
            t_filtered = dataset_df[conditions]
            
            for query_type in ["active_learning", "random_query"]:
                t_normalized = normalize_data(t_filtered, query_type, set_type)

                config_label_list = [f"{t_normalized['approach'].iloc[0]}", f"{comb['timeout_predictor_usage']}_Timeout_predictor", f"{comb['timeout_limit']}"]
                for i, config_label in enumerate(config_label_list):
                    config_label = config_label.replace("Yes_Timeout_predictor", "Timeout_predictor").replace("No_Timeout_predictor_", "").replace("100", "Dynamic_Timeout").replace("3600", "")
                    t_normalized["config"] = config_label.strip("_ ")
                    t1_concat_list[i] = pd.concat([t1_concat_list[i], t_normalized], ignore_index=True)

    for i, t1_concat in enumerate(t1_concat_list):
        ax = axes[i]
        t1_concat["config"] = t1_concat["config"].replace(
            {"active_learning": "Uncertainty",
            "random_query": "Random",
            "Timeout_predictor": "With TO",
            "No_Timeout_predictor": "Without TO",
            "Dynamic_Timeout": "With DT",
            "": "Without DT",}
            )

        if any("Without" in s for s in t1_concat["config"].unique()):
            t1_concat.sort_values(by="config", inplace=True, ascending=False)

        sns.set_palette("colorblind")
        sns.lineplot(data=t1_concat, x="min_gap", y="min_cost", hue="config", style="config", ax=ax, errorbar="se",
                    dashes=False, linewidth=3)
        ax.set_ylabel(f"Min Instance Cost", fontsize=25)
        ax.set_xlabel("Runtime Ratio", fontsize=25)
        ax.tick_params(axis='both', labelsize=25)
        ax.grid(True)

    handles, labels = [], []
    for ax in axes:
        for handle, label in zip(*ax.get_legend_handles_labels()):
            if label not in labels:
                handle.set_linewidth(5.0)
                handles.append(handle)
                labels.append(label)

    for ax in fig.axes:
        ax.get_legend().set_visible(False)

    fig.legend([handles[0]], [labels[0]], loc='upper center', ncol=2, fontsize=25, bbox_to_anchor=(0.10, 1.03), frameon=False)
    fig.legend([handles[1]], [labels[1]], loc='upper center', ncol=2, fontsize=25, bbox_to_anchor=(0.26, 1.03), frameon=False)
    fig.legend([handles[2]], [labels[2]], loc='upper center', ncol=2, fontsize=25, bbox_to_anchor=(0.43, 1.03), frameon=False)
    fig.legend([handles[3]], [labels[3]], loc='upper center', ncol=2, fontsize=25, bbox_to_anchor=(0.59, 1.03), frameon=False)
    fig.legend([handles[4]], [labels[4]], loc='upper center', ncol=2, fontsize=25, bbox_to_anchor=(0.76, 1.03), frameon=False)
    fig.legend([handles[5]], [labels[5]], loc='upper center', ncol=2, fontsize=25, bbox_to_anchor=(0.92, 1.03), frameon=False)

    fig.subplots_adjust(hspace=0.3, right=0.98, bottom=0.15, top=0.90, left=0.08, wspace=0.3)

    plt.savefig(f"{CONFIG_FILE_PATH}/{filename}.pdf")


def plot_min_data_selected_config_4_options(dataframe, set_type, keyword, keyword_list, filename):
    CONFIG_FILE_PATH = f"PLOTS/"
    os.makedirs(CONFIG_FILE_PATH, exist_ok=True)
    
    experiment_combs = dataframe[['timeout_predictor_usage', 'timeout_limit', 'query_size']].drop_duplicates().reset_index(drop=True)
    
    n_datasets = len(dataframe["dataset_name"].unique())
    n_cols = 3
    n_rows = (n_datasets + 2) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 6 * n_rows)) 

    axes = axes.flatten()
    
    for dataset_index, dataset in enumerate(dataframe["dataset_name"].unique()):
        dataset_df = dataframe[dataframe["dataset_name"] == dataset]
        t1_concat = pd.DataFrame()
        
        for _, comb in experiment_combs.iterrows():
            conditions = (
                (dataset_df["timeout_predictor_usage"] == comb["timeout_predictor_usage"]) &
                (dataset_df["timeout_limit"] == comb["timeout_limit"]) &
                (dataset_df["query_size"] == comb["query_size"])
            )
            t_filtered = dataset_df[conditions]
            
            for query_type in ["active_learning", "random_query"]:
                t_normalized = normalize_data(t_filtered, query_type, set_type)
                config_label = f"{comb['timeout_predictor_usage']}_Timeout_predictor_{comb['timeout_limit']}"
                config_label = config_label.replace("Yes_Timeout_predictor", "Timeout_predictor").replace("No_Timeout_predictor_", "").replace("100", "Dynamic_Timeout").replace("3600", "")
                t_normalized["config"] = config_label.strip("_ ")
                t1_concat = pd.concat([t1_concat, t_normalized], ignore_index=True)

        t1_concat["config"] = t1_concat["config"].str.replace("Dynamic_Timeout", "DT")
        t1_concat["config"] = t1_concat["config"].str.replace("Timeout_predictor", "TO")
        t1_concat["config"] = t1_concat["config"].str.replace("Timeout_predictor_Dynamic_Timeout", "TO+DT")
        t1_concat["config"] = t1_concat["config"].replace("", "Without TO & Without DT")
        t1_concat["config"] = t1_concat["config"].replace("TO_DT", "TO+DT")

        order = ['Without TO & Without DT', 'TO', 'DT', 'TO+DT']
        t1_concat['config'] = pd.Categorical(t1_concat['config'], categories=order, ordered=True)

        t1_concat.sort_values(by="config", inplace=True, ascending=False)

        colors = [(230/255, 159/255, 0/255), (86/255, 180/255, 233/255), (0/255, 158/255, 115/255), (204/255, 121/255, 167/255)]
        plt.rc('axes', prop_cycle=(plt.cycler('color', colors)))
        ax = axes[dataset_index]
        sns.lineplot(data=t1_concat, x="min_gap", y="min_cost", hue="config", style="config", ax=ax, errorbar="se",
                     dashes=False, linewidth=3)
        
        ax.set_ylabel(f"{dataset} \n Min Instance Cost", fontsize=25)
        ax.set_xlabel("Runtime Ratio", fontsize=25)
        ax.tick_params(axis='both', labelsize=25)
        ax.grid(True)

    for i in range(dataset_index + 1, len(axes)):
        axes[i].axis('off')

    handles, labels = [], []
    for ax in axes:
        for handle, label in zip(*ax.get_legend_handles_labels()):
            if label not in labels:
                handle.set_linewidth(5.0)
                handles.append(handle)
                labels.append(label)

    for ax in fig.axes:
        ax.get_legend().set_visible(False)

    fig.legend(handles, labels, loc='upper center', ncol=4, fontsize=25, bbox_to_anchor=(0.5, 1.0), frameon=False)
    fig.subplots_adjust(hspace=0.25, right=0.99, bottom=0.15, top=0.90, left=0.11, wspace=0.45)

    plt.savefig(f"{CONFIG_FILE_PATH}/{filename}.pdf")


def plot_min_data_selected_config_8_options(dataframe, set_type, keyword, keyword_list, filename):
    CONFIG_FILE_PATH = f"PLOTS/"
    os.makedirs(CONFIG_FILE_PATH, exist_ok=True)
    
    experiment_combs = dataframe[['timeout_predictor_usage', 'timeout_limit', 'query_size']].drop_duplicates().reset_index(drop=True)
    
    n_datasets = len(dataframe["dataset_name"].unique())
    n_cols = 3
    n_rows = (n_datasets + 2) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 6 * n_rows)) 

    axes = axes.flatten()
    
    for dataset_index, dataset in enumerate(dataframe["dataset_name"].unique()):
        dataset_df = dataframe[dataframe["dataset_name"] == dataset]
        t1_concat = pd.DataFrame()
        
        for _, comb in experiment_combs.iterrows():
            conditions = (
                (dataset_df["timeout_predictor_usage"] == comb["timeout_predictor_usage"]) &
                (dataset_df["timeout_limit"] == comb["timeout_limit"]) &
                (dataset_df["query_size"] == comb["query_size"])
            )
            t_filtered = dataset_df[conditions]
            
            for query_type in ["active_learning", "random_query"]:
                t_normalized = normalize_data(t_filtered, query_type, set_type)
                config_label = f"{t_normalized['approach'].iloc[0]}_{comb['timeout_predictor_usage']}_Timeout_predictor_{comb['timeout_limit']}"
                config_label = config_label.replace("Yes_Timeout_predictor", "Timeout_predictor").replace("No_Timeout_predictor_", "").replace("100", "Dynamic_Timeout").replace("3600", "")
                t_normalized["config"] = config_label.strip("_ ")
                t1_concat = pd.concat([t1_concat, t_normalized], ignore_index=True)

        t1_concat = t1_concat[t1_concat["config"].isin(keyword_list)].sort_values(by="config")
        t1_concat["config"] = t1_concat["config"].str.replace("active_learning", "Uncertainty")
        t1_concat["config"] = t1_concat["config"].str.replace("random_query", "Random")
        t1_concat["config"] = t1_concat["config"].str.replace("Dynamic_Timeout", "DT")
        t1_concat["config"] = t1_concat["config"].str.replace("Timeout_predictor", "TO")
        t1_concat["config"] = t1_concat["config"].replace({
            'Uncertainty_TO': 'Uncertainty (TO)',
            'Uncertainty_DT': 'Uncertainty (DT)',
            'Uncertainty': 'Uncertainty (NO TO & NO DT)',
            'Uncertainty_TO_DT': 'Uncertainty (TO+DT)',
            'Random_TO': 'Random (TO)',
            'Random_DT': 'Random (DT)',
            'Random_TO_DT': 'Random (TO+DT)',
            'Random': 'Random (NO TO & NO DT)'
        })

        ax = axes[dataset_index]
        # colors = [(230/255, 159/255, 0/255), (86/255, 180/255, 233/255), (0/255, 158/255, 115/255), (204/255, 121/255, 167/255)]
        # colors += [(213/255, 94/255, 0/255), (0/255, 114/255, 178/255), (240/255, 228/255, 66/255), (0/255, 0/255, 0/255)]
        # plt.rc('axes', prop_cycle=(plt.cycler('color', colors)))
        sns.set_palette("colorblind")
        sns.lineplot(data=t1_concat, x="min_gap", y="min_cost", hue="config", style="config", ax=ax, errorbar="se",
                     dashes=False, linewidth=3)
        
        ax.set_ylabel(f"{dataset} \n Min Instance Cost", fontsize=25)
        ax.set_xlabel("Runtime Ratio", fontsize=25)
        ax.tick_params(axis='both', labelsize=25)
        ax.grid(True)

    for i in range(dataset_index + 1, len(axes)):
        axes[i].axis('off')

    handles, labels = [], []
    for ax in axes:
        for handle, label in zip(*ax.get_legend_handles_labels()):
            if label not in labels:
                handle.set_linewidth(5.0)
                handles.append(handle)
                labels.append(label)

    for ax in fig.axes:
        ax.get_legend().set_visible(False)

    fig.legend(handles, labels, loc='upper center', ncol=3, fontsize=25, bbox_to_anchor=(0.5, 1.0), frameon=False)
    fig.subplots_adjust(hspace=0.25, right=0.99, bottom=0.1, top=0.85, left=0.11, wspace=0.45)

    plt.savefig(f"{CONFIG_FILE_PATH}/{filename}.pdf")


config_list = ['active_learning', 'active_learning_Dynamic_Timeout',
 'active_learning_Timeout_predictor',
 'active_learning_Timeout_predictor_Dynamic_Timeout', 'random_query',
 'random_query_Dynamic_Timeout', 'random_query_Timeout_predictor',
 'random_query_Timeout_predictor_Dynamic_Timeout']

config_list_dt_to = ['active_learning_Timeout_predictor_Dynamic_Timeout', 'random_query_Timeout_predictor_Dynamic_Timeout']

config_list_dt = ['active_learning_Dynamic_Timeout', 'random_query_Dynamic_Timeout',
                  'active_learning_Timeout_predictor_Dynamic_Timeout', 'random_query_Timeout_predictor_Dynamic_Timeout']

config_list_timeout_predictor_subset = ['active_learning', 'active_learning_Timeout_predictor', 'random_query', 'random_query_Timeout_predictor']

config_list_timeout_predictor_full = ['active_learning_Timeout_predictor', 'active_learning_Timeout_predictor_Dynamic_Timeout', 
                                      'random_query_Timeout_predictor', 'random_query_Timeout_predictor_Dynamic_Timeout']

config_list_dt_to = ['active_learning_Timeout_predictor_Dynamic_Timeout', 'random_query_Timeout_predictor_Dynamic_Timeout']

plot_min_data_selected_config_2_options(test_df_q1, "test", "PLOT", config_list_dt_to, "2_configs")
plot_min_data_selected_config_3_options(test_df_q1, "test", "PLOT", config_list, "3_configs")
plot_min_data_selected_config_4_options(test_df_q1, "test", "PLOT", config_list, "4_configs")
plot_min_data_selected_config_8_options(test_df_q1, "test", "PLOT", config_list, "8_configs")