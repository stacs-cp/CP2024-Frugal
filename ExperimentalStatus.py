from pathlib import Path
import pandas as pd
import re
import ast
import time

SLURM_FILES_PATH = Path('EXPERIMENT_LOGS')

def get_experiment_status(folder_path: Path):
    print("Getting experiment status...")
    status_df = pd.DataFrame(columns=['Approach', 'Total', 'Finished', 'Unfinished', 'Timeouted'])

    for folder in folder_path.iterdir():
        print(f"Approach: {folder.stem}")
        finished_scripts = []
        unfinished_scripts = []
        timeouted_scripts = []

        if folder.is_dir():
            directory_entries = folder.iterdir()
            sorting_key = lambda path: [int(s) if s.isdigit() else s for s in path.stem.split('_')]
            sorted_entries = sorted(directory_entries, key=sorting_key)

            for file in folder.iterdir():
                if file.is_file():
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

            status_df = pd.concat([status_df, pd.DataFrame([[folder.stem, len(sorted_entries), len(finished_scripts), len(unfinished_scripts), len(timeouted_scripts)]], columns=['Approach', 'Total', 'Finished', 'Unfinished', 'Timeouted'])], ignore_index=True)

            print(f"Total number of scripts: {len(sorted_entries)}")
            print(f"Total number of finisihed scripts: {len(finished_scripts)}")
            print(f"Total number of unfinisihed scripts: {len(unfinished_scripts)}")
            print(f"Total number of timeouted scripts: {len(timeouted_scripts)}")
    
    return status_df

def extract_parameters(sentence, line, format):
    return float(re.search(rf".*{sentence} {format}", line).group(1))

def create_data_dict(dataset_name, configuration, seed_number, timeout_predictor_usage, timeout_limit, query_size, split_number, query_number, column_names, column_values):
    data_dict = {
        'dataset_name': [dataset_name] * len(query_number),
        'configuration': [configuration] * len(query_number),
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

def get_job_status(folder_path, classification_type):
    print(f"Getting job status for {folder_path.stem}...")
    experiment_df_test = pd.DataFrame()
    directory_entries = folder_path.iterdir()
    sorting_key = lambda path: [int(s) if s.isdigit() else s for s in path.stem.split('_')]
    sorted_entries = sorted(directory_entries, key=sorting_key)

    vbs_test = ""
    sbs_test = ""
    passive_learning_runtime_test = ""
    passive_learning_instance_cost = ""

    keyword = "Active Learning"
    if "RANDOM" in folder_path.stem:
        keyword += " Random Query"
    for i, file in enumerate(sorted_entries):
        active_learning_instance_cost_list, active_learning_runtime_validation_list, active_learning_runtime_test_list = [], [], []
        
        priority = ""
        pareto_features = ""
        penalty_type = ""
        with open(file) as f:
            lines = f.readlines()

            for line in lines:
                if "Paramaters are" in line:
                    if "priority" in line:
                        priority_string = re.search(r"'priority': ({.*?})", line).group(1)
                        priority = ast.literal_eval(priority_string)
                        priority = " -> ".join(priority.keys())
                    if "pareto_features" in line:
                        pareto_features_string = re.search(r"'pareto_features': ({.*?})", line).group(1)
                        pareto_features = ast.literal_eval(pareto_features_string)
                        pareto_features = " - ".join(pareto_features.keys())
                    
                    if "penalty_type" in line:
                        penalty_type = re.search(r"'penalty_type': '(\S+)'", line).group(1)

                    dataset_name, query_size, timeout_predictor_usage, timeout_limit, seed_number, split_number = re.search(r".*'dataset': '(\S+)'.*'query_size': (\d+).*'timeout_predictor_usage': '(\S+)'.*'timeout_limit': (\d+).*'RANDOM_STATE_SEED': (\d+).*'split_number': (\d+)", line).groups()
                    query_size, timeout_limit, seed_number, split_number = map(int, [query_size, timeout_limit, seed_number, split_number])

                elif "Virtual best solver time in test" in line:
                    vbs_test = extract_parameters("Virtual best solver time in test =", line, r"(\d+(?:\.\d+)?)")
                elif "Single best solver time test" in line:
                    sbs_test = extract_parameters("Single best solver time test =", line, r"[^)]*\(([\d.]+)\)")
                elif "Passive Learning Instance Cost" in line:
                    passive_learning_instance_cost = extract_parameters("Passive Learning Instance Cost:", line, r"(\d+(?:\.\d+)?)")
                elif "Passive Learning Runtime on Test Set" in line:
                    passive_learning_runtime_test = extract_parameters("Passive Learning Runtime on Test Set:", line, r"(\d+(?:\.\d+)?)")
                elif "Passive Learning Classifier Runtime on Test Set" in line:
                    passive_learning_runtime_test = extract_parameters("Passive Learning Classifier Runtime on Test Set:", line, r"(\d+(?:\.\d+)?)")
                elif f"{keyword} Instance Cost" in line:
                    active_learning_instance_cost_list.append(extract_parameters(f"{keyword} Instance Cost:", line, r"(\d+(?:\.\d+)?)"))
                elif f"{keyword} Runtime on Validation Set" in line:
                    active_learning_runtime_validation_list.append(extract_parameters(f"{keyword} Runtime on Validation Set:", line, r"(\d+(?:\.\d+)?)"))
                elif f"{keyword} Runtime on Test Set" in line:
                    active_learning_runtime_test_list.append(extract_parameters(f"{keyword} Runtime on Test Set:", line, r"(\d+(?:\.\d+)?)"))

        penalty_flag = {
            "Only-timeout-penalty": "ONLY TO PEN",
            "Dynamic-penalty": "DYN PEN"
        }.get(penalty_type, "")

        dtl_flag = "DTL" if timeout_limit == 100 else ""
        to_flag = "TO" if timeout_predictor_usage == "Yes" else ""

        flags = " + ".join(filter(None, [penalty_flag, dtl_flag, to_flag]))

        config = f"{classification_type}{f' ({priority})' if priority else ''}{f' ({pareto_features})' if pareto_features else ''} ({flags})" if flags else f"{classification_type}{f' ({priority})' if priority else ''}"

        column_names_test = [
            f'vbs',
            f'sbs',
            f'passive_runtime',
            f'passive_instance_cost',
            f'runtime',
            f'instance_cost'
        ]
        
        column_values_test = [vbs_test, sbs_test, passive_learning_runtime_test, passive_learning_instance_cost, 
                            active_learning_runtime_test_list, active_learning_instance_cost_list]
        
        data_test = create_data_dict(dataset_name, config, seed_number, timeout_predictor_usage, timeout_limit, query_size, split_number, active_learning_runtime_test_list, column_names_test, column_values_test)
       
        df_test = pd.DataFrame(data_test)

        experiment_df_test = pd.concat([experiment_df_test, df_test])
    
    return experiment_df_test.reset_index(drop=True)


def fill_random_with_uncertainty(df):
    start_time = time.time()

    unique_combinations = df[["dataset_name", "seed_number", "timeout_predictor_usage", 
                          "timeout_limit", "query_size", "split_number"]].drop_duplicates()
    
    random_mask = df["configuration"].str.startswith("RANDOM")
    uncertainty_mask = df["configuration"].str.startswith("UNCERTAINTY")

    for comb in unique_combinations.itertuples(index=False):

        condition_mask = (
            (df["dataset_name"] == comb.dataset_name) & 
            (df["seed_number"] == comb.seed_number) & 
            (df["timeout_predictor_usage"] == comb.timeout_predictor_usage) & 
            (df["timeout_limit"] == comb.timeout_limit) & 
            (df["query_size"] == comb.query_size) & 
            (df["split_number"] == comb.split_number)
        )

        uncertainty_values = df.loc[uncertainty_mask & condition_mask, 
                                    ["vbs", "sbs", "passive_runtime", "passive_instance_cost"]].iloc[0]

        df.loc[random_mask & condition_mask, ["vbs", "sbs", "passive_runtime", "passive_instance_cost"]] = uncertainty_values.values
    

    df["vbs"] = df["vbs"].astype(float)
    df["sbs"] = df["sbs"].astype(float)
    df["passive_runtime"] = df["passive_runtime"].astype(float)
    df["passive_instance_cost"] = df["passive_instance_cost"].astype(float)
    end_time = time.time()
    print(f"Time taken to fill random query information: {round(end_time - start_time, 2)} seconds")
    
    return df

def normalize_columns(df):
    start_time = time.time()
    df = fill_random_with_uncertainty(df)

    df["runtime"] = round(df["passive_runtime"] / df["runtime"], 2)
    df["runtime"] = df["runtime"].clip(upper=1)
    
    df['instance_cost'] = round(df[f'instance_cost'] / df[f'passive_instance_cost'], 2)  

    df["vbs"] = round(df["vbs"] / df["passive_runtime"], 2)
    df["sbs"] = round(df["sbs"] / df[f"passive_runtime"], 2)

    df[f"passive_instance_cost"] = 1
    df[f"passive_runtime"] = 1

    end_time = time.time()
    print(f"Time taken to normalize columns: {round(end_time - start_time, 2)} seconds")
    return df

def return_merged_data(merged_df):
    merged_df = normalize_columns(merged_df)

    return merged_df

if not Path("experimental_status.csv").exists():
    print("Creating experimental status file...")
    status_df = get_experiment_status(SLURM_FILES_PATH)
    status_df.to_csv(r'CSV_FILES\experimental_status.csv', index=False)

main_df = pd.DataFrame()

for folder in SLURM_FILES_PATH.iterdir():
    if folder.is_dir():
        experiment_df = get_job_status(folder, folder.name)
        main_df = pd.concat([main_df, experiment_df])

main_df = return_merged_data(main_df)

main_df.sort_values(by=["dataset_name", "configuration", "seed_number", "split_number", "query_number"], inplace=True)
main_df.to_csv(r"CSV_FILES\merged_results.csv", index=False)
