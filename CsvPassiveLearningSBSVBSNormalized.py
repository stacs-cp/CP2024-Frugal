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


def get_experimental_results_pl(sorted_entries):
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

        column_names_validation = ['vbs_validation', 'sbs_validation', 'passive_learning_runtime_validation', 
                                   'passive_learning_instance_cost']

        column_values_validation = [vbs_validation, sbs_validation, passive_learning_runtime_validation, 
                                    passive_learning_instance_cost]

        column_names_test = ['vbs_test', 'sbs_test', 'passive_learning_runtime_test', 'passive_learning_instance_cost']
        
        column_values_test = [vbs_test, sbs_test, passive_learning_runtime_test, passive_learning_instance_cost]
        
        data_val = create_data_dict(dataset_name, seed_number, timeout_predictor_usage, timeout_limit, query_size, split_number, [passive_learning_instance_cost], column_names_validation, column_values_validation)
        data_test = create_data_dict(dataset_name, seed_number, timeout_predictor_usage, timeout_limit, query_size, split_number, [passive_learning_instance_cost], column_names_test, column_values_test)
        
        df_val = pd.DataFrame(data_val)
        df_test = pd.DataFrame(data_test)

        experiment_df_val = pd.concat([experiment_df_val, df_val])
        experiment_df_test = pd.concat([experiment_df_test, df_test])

    return experiment_df_val, experiment_df_test

def normalize_dataframe(experiment_df, set_type):
    experiment_df[f'passive_learning_runtime_{set_type}'] = round((experiment_df[f'passive_learning_runtime_{set_type}'] - experiment_df[f"vbs_{set_type}"]) / (experiment_df[f"sbs_{set_type}"] - experiment_df[f"vbs_{set_type}"]), 2)
    experiment_df[f"vbs_{set_type}"] = 0
    experiment_df[f"sbs_{set_type}"] = 1
    experiment_df["passive_learning_instance_cost"] = 1

sorted_entries_pl_q1 = get_job_status(SLURM_FOLDER_PATH_Q1_AL)

val_pl_q1, test_pl_q1 = get_experimental_results_pl(sorted_entries_pl_q1)

normalize_dataframe(val_pl_q1, "validation")
normalize_dataframe(test_pl_q1, "test")

test_pl_q1 = test_pl_q1[test_pl_q1["timeout_limit"] == 3600]
val_pl_q1 = val_pl_q1[val_pl_q1["timeout_limit"] == 3600]

val_pl_q1.to_csv(CSV_SAVE_PATH / "val_pl_sbs_vbs_normalized.csv", index=False)
test_pl_q1.to_csv(CSV_SAVE_PATH / "test_pl_sbs_vbs_normalized.csv", index=False)


