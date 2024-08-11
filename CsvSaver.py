from pathlib import Path
import re
import numpy as np
import pandas as pd
import math

import warnings
warnings.filterwarnings("ignore")

SLURM_FOLDER_PATH_Q1_AL = Path("EXPERIMENT_LOGS/UNCERTAINTY_SLURM_FILES/")
SLURM_FOLDER_PATH_Q1_RQ = Path("EXPERIMENT_LOGS/RANDOM_SLURM_FILES/")
CSV_SAVE_PATH = Path("CSV_FILES")

def get_job_status(folder_path):
    directory_entries = folder_path.iterdir()
    sorting_key = lambda path: [int(s) if s.isdigit() else s for s in path.stem.split('_')]
    sorted_entries = sorted(directory_entries, key=sorting_key)

    finished_scripts = []
    unfinished_scripts = []
    timeouted_scripts = []

    for i, file in enumerate(sorted_entries):
        with open(file) as f:
            lines = f.readlines()
            for line in lines:
                if "Experiment is finished" in line:
                    finished_scripts.append(file)
                    break
                if "CANCELLED" in line:
                    timeouted_scripts.append(file)
                    break


    for i, file in enumerate(sorted_entries):
        if file not in finished_scripts and file not in timeouted_scripts:
            unfinished_scripts.append(file)

    print(f"Total number of scripts: {len(sorted_entries)}")
    print(f"Total number of finisihed scripts: {len(finished_scripts)}")
    print(f"Total number of unfinisihed scripts: {len(unfinished_scripts)}")
    print(f"Total number of timeouted scripts: {len(timeouted_scripts)}")

    return sorted_entries


def create_data_dict(dataset_name, seed_number, timeout_predictor_usage, timeout_limit, query_size, split_number, query_number, column_names, column_values):
    data_dict = {
        'dataset_name': [dataset_name] * len(query_number),
        'seed_number': [seed_number] * len(query_number),
        "timeout_predictor_usage": [timeout_predictor_usage] * len(query_number),
        "timeout_limit": [timeout_limit] * len(query_number),
        "query_size": [query_size] * len(query_number),
        'split_number': [split_number] * len(query_number),
        'query_number': list(range(len(query_number))),
    }
    for column_name, column_value in zip(column_names, column_values):
        data_dict[column_name] = [column_value] * len(query_number) if not isinstance(column_value, list) else column_value
    return data_dict


def get_experimental_results_al(sorted_entries):
    experiment_df_val = pd.DataFrame()
    experiment_df_test = pd.DataFrame()

    for i, file in enumerate(sorted_entries):
        active_learning_instance_cost_list, active_learning_runtime_validation_list, active_learning_runtime_test_list = [], [], []

        with open(file) as f:
            lines = f.readlines()

            for line in lines:
                if "Paramaters are" in line:
                    dataset_name, query_size, timeout_predictor_usage, timeout_limit, seed_number, split_number = re.search(r".*'dataset': '(\S+)'.*'query_size': (\d+).*'timeout_predictor_usage': '(\S+)'.*'timeout_limit': (\d+).*'RANDOM_STATE_SEED': (\d+).*'split_number': (\d+)", line).groups()
                    query_size, timeout_limit, seed_number, split_number = map(int, [query_size, timeout_limit, seed_number, split_number])

                if "Virtual best solver time in validation" in line:
                    vbs_validation = float(re.search(r".*Virtual best solver time in validation = (\d+(?:\.\d+)?)", line).group(1))
                elif "Virtual best solver time in test" in line:
                    vbs_test = float(re.search(r".*Virtual best solver time in test = (\d+(?:\.\d+)?)", line).group(1))
                elif "Single best solver time validation" in line:
                    sbs_validation = float(re.search(r".*Single best solver time validation = [^)]*\(([\d.]+)\)", line).group(1))
                elif "Single best solver time test" in line:
                    sbs_test = float(re.search(r".*Single best solver time test = [^)]*\(([\d.]+)\)", line).group(1))
                elif "Passive Learning Instance Cost" in line:
                    passive_learning_instance_cost = float(re.search(r".*Passive Learning Instance Cost: (\d+(?:\.\d+)?)", line).group(1))
                elif "Passive Learning Runtime on Validation Set" in line:
                    passive_learning_runtime_validation = float(re.search(r".*Passive Learning Runtime on Validation Set: (\d+(?:\.\d+)?)", line).group(1))
                elif "Passive Learning Runtime on Test Set" in line:
                    passive_learning_runtime_test = float(re.search(r".*Passive Learning Runtime on Test Set: (\d+(?:\.\d+)?)", line).group(1))
                elif "Active Learning Instance Cost" in line:
                    active_learning_instance_cost_list.append(float(re.search(r".*Active Learning Instance Cost: (\d+(?:\.\d+)?)", line).group(1)))
                elif "Active Learning Runtime on Validation Set" in line:
                    active_learning_runtime_validation_list.append(float(re.search(r".*Active Learning Runtime on Validation Set: (\d+(?:\.\d+)?)", line).group(1)))
                elif "Active Learning Runtime on Test Set" in line:
                    active_learning_runtime_test_list.append(float(re.search(r".*Active Learning Runtime on Test Set: (\d+(?:\.\d+)?)", line).group(1)))

        column_names_validation = ['vbs_validation', 'sbs_validation', 'passive_learning_runtime_validation', 
                                   'passive_learning_instance_cost', 'active_learning_runtime_validation', 'active_learning_instance_cost']

        column_values_validation = [vbs_validation, sbs_validation, passive_learning_runtime_validation, 
                                    passive_learning_instance_cost, active_learning_runtime_validation_list, active_learning_instance_cost_list]

        column_names_test = ['vbs_test', 'sbs_test', 'passive_learning_runtime_test', 'passive_learning_instance_cost',
                             'active_learning_runtime_test', 'active_learning_instance_cost']
        
        column_values_test = [vbs_test, sbs_test, passive_learning_runtime_test, passive_learning_instance_cost, 
                              active_learning_runtime_test_list, active_learning_instance_cost_list]
        
        data_val = create_data_dict(dataset_name, seed_number, timeout_predictor_usage, timeout_limit, query_size, split_number, active_learning_runtime_validation_list, column_names_validation, column_values_validation)
        data_test = create_data_dict(dataset_name, seed_number, timeout_predictor_usage, timeout_limit, query_size, split_number, active_learning_runtime_test_list, column_names_test, column_values_test)
        
        df_val = pd.DataFrame(data_val)
        df_test = pd.DataFrame(data_test)

        experiment_df_val = pd.concat([experiment_df_val, df_val])
        experiment_df_test = pd.concat([experiment_df_test, df_test])

    return experiment_df_val, experiment_df_test


def get_experimental_results_rq(sorted_entries_rq):
    experiment_rq_val = pd.DataFrame()
    experiment_rq_test = pd.DataFrame()

    for i, file in enumerate(sorted_entries_rq):
        random_query_instance_cost_list, random_query_runtime_validation_list, random_query_runtime_test_list = [], [], []

        with open(file) as f:
            lines = f.readlines()

            for line in lines:
                if "Paramaters are" in line:
                    dataset_name, query_size, timeout_predictor_usage, timeout_limit, seed_number, split_number = re.search(r".*'dataset': '(\S+)'.*'query_size': (\d+).*'timeout_predictor_usage': '(\S+)'.*'timeout_limit': (\d+).*'RANDOM_STATE_SEED': (\d+).*'split_number': (\d+)", line).groups()
                    query_size, timeout_limit, seed_number, split_number = map(int, [query_size, timeout_limit, seed_number, split_number])
                elif "Feature cost of train set" in line:
                    match = re.search(r".*Feature cost of train set is (\d+(?:\.\d+)?)", line)
                    feature_cost = float(match.group(1))
                elif "Active Learning Random Query Instance Cost" in line:
                    random_query_instance_cost_list.append(float(re.search(r".*Active Learning Random Query Instance Cost: (\d+(?:\.\d+)?)", line).group(1)))
                elif "Active Learning Random Query Runtime on Validation Set" in line:
                    random_query_runtime_validation_list.append(float(re.search(r".*Active Learning Random Query Runtime on Validation Set: (\d+(?:\.\d+)?)", line).group(1)))
                elif "Active Learning Random Query Runtime on Test Set" in line:
                    random_query_runtime_test_list.append(float(re.search(r".*Active Learning Random Query Runtime on Test Set: (\d+(?:\.\d+)?)", line).group(1)))
        
        column_names_rq_val = ['feature_cost', 'random_query_runtime_validation', 'random_query_instance_cost']

        column_values_rq_val = [feature_cost, random_query_runtime_validation_list, random_query_instance_cost_list]

        column_names_rq_test = ['feature_cost', 'random_query_runtime_test', 'random_query_instance_cost']
    
        column_values_rq_test = [feature_cost, random_query_runtime_test_list, random_query_instance_cost_list]

        data_rq_val = create_data_dict(dataset_name, seed_number, timeout_predictor_usage, timeout_limit, query_size, split_number, random_query_runtime_validation_list, column_names_rq_val, column_values_rq_val)
        data_rq_test = create_data_dict(dataset_name, seed_number, timeout_predictor_usage, timeout_limit, query_size, split_number, random_query_runtime_test_list, column_names_rq_test, column_values_rq_test)

        df_rq_val = pd.DataFrame(data_rq_val)
        df_rq_test = pd.DataFrame(data_rq_test)

        experiment_rq_val = pd.concat([experiment_rq_val, df_rq_val])
        experiment_rq_test = pd.concat([experiment_rq_test, df_rq_test])

    return experiment_rq_val, experiment_rq_test


def fill_data_with_to_pl(experiment_df, set_type):
    experiment_combinations = experiment_df[["dataset_name", "seed_number", "timeout_limit", "query_size", "split_number"]].drop_duplicates().reset_index(drop=True)

    for _, comb in experiment_combinations.iterrows():
        row = experiment_df[(experiment_df["dataset_name"] == comb["dataset_name"]) & (experiment_df["seed_number"] == comb["seed_number"]) & (experiment_df["timeout_limit"] == comb["timeout_limit"]) & (experiment_df["query_size"] == comb["query_size"]) & (experiment_df["split_number"] == comb["split_number"])]
        if row["timeout_predictor_usage"].values[0] == "Yes":
            row[f"passive_learning_runtime_{set_type}"] = experiment_df[(experiment_df["dataset_name"] == comb["dataset_name"]) & (experiment_df["seed_number"] == comb["seed_number"]) & (experiment_df["timeout_predictor_usage"] == "No") & (experiment_df["timeout_limit"] == comb["timeout_limit"]) & (experiment_df["query_size"] == comb["query_size"]) & (experiment_df["split_number"] == comb["split_number"])][f"passive_learning_runtime_{set_type}"].values[0]


def return_merged_data(experiment_df_al, experiment_df_rq, set_type, add_feature_cost):
    merged_df = pd.merge(experiment_df_al, experiment_df_rq, on=['dataset_name', 'seed_number', 'timeout_predictor_usage', 'timeout_limit', 'query_size', 'split_number', "query_number"], how='outer')
    merged_df.sort_values(by=['dataset_name', 'seed_number', 'timeout_predictor_usage', 'timeout_limit', 'query_size', 'split_number', "query_number"], inplace=True)

    groupby_columns = ['dataset_name', 'seed_number', 'timeout_limit', 'query_size', 'split_number']

    merged_df['passive_learning_instance_cost'] = merged_df.groupby(groupby_columns)['passive_learning_instance_cost'].transform(lambda x: x.fillna(method='ffill'))

    for column in [f"active_learning_runtime_{set_type}", f"random_query_runtime_{set_type}"]:
        merged_df[column] = round(merged_df[f"passive_learning_runtime_{set_type}"]/merged_df[column], 2)
        merged_df[column] = merged_df[column].clip(upper=1)

    for column in ["active_learning_instance_cost", "random_query_instance_cost"]:
        merged_df[column] = round(merged_df[column] / merged_df["passive_learning_instance_cost"], 2)

    merged_df[f"vbs_{set_type}"] = round(merged_df[f"vbs_{set_type}"] / merged_df[f"passive_learning_runtime_{set_type}"] , 2)
    merged_df[f"sbs_{set_type}"] = round(merged_df[f"sbs_{set_type}"] / merged_df[f"passive_learning_runtime_{set_type}"] , 2)

    merged_df["passive_learning_instance_cost"] = 1
    merged_df[f"passive_learning_runtime_{set_type}"] = 1

    return merged_df


sorted_entries_al_q1 = get_job_status(SLURM_FOLDER_PATH_Q1_AL)
sorted_entries_rq_q1 = get_job_status(SLURM_FOLDER_PATH_Q1_RQ)

val_al_q1, test_al_q1 = get_experimental_results_al(sorted_entries_al_q1)
val_rq_q1, test_rq_q1 = get_experimental_results_rq(sorted_entries_rq_q1)

fill_data_with_to_pl(val_al_q1, "validation")
fill_data_with_to_pl(test_al_q1, "test")

q1_df_val = return_merged_data(val_al_q1, val_rq_q1, "validation", False)
q1_df_test = return_merged_data(test_al_q1, test_rq_q1, "test", False)

q1_df_val.to_csv(CSV_SAVE_PATH / "val_q1_no_weight.csv", index=False)
q1_df_test.to_csv(CSV_SAVE_PATH / "test_q1_no_weight.csv", index=False)