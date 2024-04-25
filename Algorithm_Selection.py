import argparse
from functools import reduce
import arff
import collections
import itertools
import json
import logging
import math
import numpy as np
import os
from PassiveRFModel import PassiveRFModel
from ActiveRFModel import ActiveRFModel
import random
import sys
from sklearn.model_selection import KFold
from statistics import mode

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier

from modAL import ActiveLearner
from modAL.uncertainty import classifier_uncertainty
from modAL.uncertainty import uncertainty_sampling

import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(
    description="efficient algorithm selection using active learning")

args = [
    {"name": "--dataset", "required": True, "type": str,
        "help": "Denotes name of the dataset"},
    {"name": "--prefix", "required": False, "default": "", "type": str,
        "help": "Prefix for the output files"},
    {"name": "--penalty", "required": False, "default": 10, "type": int,
        "help": "This parameter signifies the timeout coefficient (PAR2, PAR10)"},
    {"name": "--weight", "required": False, "default": "Weighted", "type": str,
        "help": "This variable indicates whether sample weighting is applied (Weighted, No_Weight)"},
    {"name": "--voting", "required": False, "default": "Hard", "type": str,
        "help": "The voting strategy employed in the model (Hard, Soft)"},
    {"name": "--pre_selection_size", "required": False, "default": 64, "type": int,
        "help": "This refers to the number of algorithms selected during the algorithm pre-selection phase"},
    {"name": "--pre_selection_technique", "required": False, "default": "No_pre_selection", "type": str,
        "help": "This refers to the approach for pre-selecting algorithms (No_pre_selection, VBest_contribution, Minimum_runtime)"},
    {"name": "--feature_elimination_threshold", "required": False, "default": 20, "type": int,
        "help": "This parameter sets the maximum ratio of missing features that will be tolerated."},
    {"name": "--imputer_strategy", "required": False, "default": "median", "type": str,
        "help": "This parameter sets the maximum ratio of missing features that will be tolerated. (mean, median, most_frequent)"},
    {"name": "--feature_selection_technique", "required": False, "default": "No_feature_selection", "type": str,
        "help": "Denotes the technique used for selecting a subset of relevant features from the complete feature space for model training (Backward feature elimination, Forward feature selection, No feature selection)"},
    {"name": "--feature_scaling_technique", "required": False, "default": "StandardScaler", "type": str,
            "help": "Specifies the method used to normalize or standardize features in the feature set (StandardScaler, MinMaxScaler, RobustScaler, No scaling)"},
    {"name": "--initial_train_size", "required": False, "default": 20, "type": int,
            "help": "The size of the initial training dataset in pool-based sampling."},
    {"name": "--query_size", "required": False, "default": 5, "type": int,
            "help": "The size of the initial training dataset in pool-based sampling (percentage-based)."},
    {"name": "--timeout_predictor_usage", "required": False, "default": "Yes", "type": str,
            "help": "Specifies whether timeout estimators are used in the sample selection in the query and in voting for filtering models (Yes, No)."},
    {"name": "--timeout_limit", "required": False, "default": 100, "type": int,
            "help": "Specifies the initial maximum runtime allocated for an algorithm on a single instance."},
    {"name": "--timeout_increase_rate", "required": False, "default": 100, "type": int,
            "help": "Indicates the value at which the maximum runtime limit is incrementally increased for each query step."},           
    {"name": "--seed", "required": False, "default": 7, "type": int,
        "help": "The seed value for random state generation (7, 42, 99, 123, 12345)"},
    {"name": "--n_splits", "required": False, "type": int, "default": 10,
        "help": "The count of groups False in splitting (5, 10)"}, 
    {"name": "--split_number", "required": True, "type": int,
        "help": "The specific split number (0, 1, 2, 3, 4, 5, 6, 7, 8, 9)"},
]

for arg in args:
    parser.add_argument(arg["name"], required=arg.get(
        "required", False), default=arg.get("default"), type=arg["type"], help=arg["help"])
    
args = parser.parse_args()

prefix = "ALGORITHM_SELECTION" + args.prefix

config = {
    'dataset': args.dataset,
    'penalty': args.penalty,
    'weight': args.weight,
    'voting': args.voting,
    'pre_selection_size': args.pre_selection_size,
    'pre_selection_technique': args.pre_selection_technique,
    'feature_elimination_threshold': args.feature_elimination_threshold,
    'imputer_strategy': args.imputer_strategy,
    'feature_selection_technique': args.feature_selection_technique,
    'feature_scaling_technique': args.feature_scaling_technique,
    'initial_train_size': args.initial_train_size,
    'query_size': args.query_size,
    'timeout_predictor_usage': args.timeout_predictor_usage,
    'timeout_limit': args.timeout_limit,
    'timeout_increase_rate': args.timeout_increase_rate,
    'RANDOM_STATE_SEED': args.seed,
    'n_splits': args.n_splits,
    'split_number': args.split_number,
    'max_timeout' : 3600,
    'param_grid': {
            'n_estimators': 100,
            'criterion': 'gini',
            'max_features': 'sqrt',
            'max_depth': 2**31,
            'min_samples_split': 2,
            'bootstrap' : True
        }
}

file_path_config = {
    'DATASET_PATH': "DATASETS/",
    'RESULT_PATH': f"{prefix}/{config['dataset']}/SEED_{config['RANDOM_STATE_SEED']}/",
    'SPLIT_PATH': f"{prefix}/{config['dataset']}/SEED_{config['RANDOM_STATE_SEED']}/Split_{config['split_number']}/",
}

random.seed(config["RANDOM_STATE_SEED"])
np.random.seed(config["RANDOM_STATE_SEED"])

Path(file_path_config["SPLIT_PATH"]).mkdir(parents=True, exist_ok=True)

def setup_logger(name, log_path, level=logging.INFO):
    formatter = logging.Formatter(
        '%(asctime)s | %(levelname)s | %(message)s', datefmt='%Y-%m-%d %H:%M')

    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    return logger

logger_name = f"{file_path_config['SPLIT_PATH']}/result.log"
logger = setup_logger(logger_name, logger_name)

logger.info(f'Paramaters are: {config}')
logger.info('%s is current dataset', config["dataset"])

def read_file(file_path, file_type):
    logger.info('%s is found', file_path)
    if file_type == 'csv':
        data = pd.read_csv(file_path)
    else:
        data = process_arff_file(file_path)
    logger.info('%s is finished', file_path)
    return pd.DataFrame(data)


def renaming_rule(x):
    rules = {
        "PAR10": "runtime",
        "time": "runtime",
        "runtime": "runtime",
        "algorithm": "algorithm"
    }
    for key, value in rules.items():
        if key in x:
            return value
    return x


def process_algo_runs(algo_runs):
    algo_runs.columns = [renaming_rule(col) for col in algo_runs.columns]
    if "runstatus" in algo_runs:
        algo_runs = algo_runs.loc[(
            algo_runs["runstatus"] == "ok")].reset_index(drop=True)
    algo_runs.loc[:, "repetition"].astype(int)
    algo_runs = algo_runs[["instance_id", "algorithm", "runtime"]]
    pivot_table_algo = algo_runs.pivot_table(values='runtime', index=["instance_id"], columns='algorithm',
                                             aggfunc='first', fill_value=config["max_timeout"]).reset_index()
    
    return pivot_table_algo.rename_axis(None, axis=1).reset_index(drop=True)


def process_feature_costs(feature_costs, algo_runs):
    if feature_costs.empty:
        feature_costs['instance_id'] = algo_runs["instance_id"].unique()
        feature_costs['repetition'] = 1
        feature_costs['Sum'] = 0
    else:
        feature_costs.fillna(0)
        feature_costs["Sum"] = feature_costs.iloc[:, 2:].sum(axis=1)
        feature_costs = feature_costs[["instance_id", "repetition", "Sum"]]
    feature_costs.loc[:, "repetition"].astype(int)
    feature_costs = feature_costs[feature_costs["repetition"] == 1]

    return feature_costs.drop(['repetition'], axis=1)


def process_feature_values(feature_values):
    min_count = int(((100 - config["feature_elimination_threshold"]) / 100) *
                    feature_values.iloc[:, 2:].shape[1] + 1 + 2)
    feature_values = feature_values.dropna(axis=0, thresh=min_count)
    feature_values.loc[:, "repetition"].astype(int)
    feature_values = feature_values[feature_values["repetition"] == 1]

    return feature_values.drop(['repetition'], axis=1)


def read_dataset():
    def data_engineering(algo_runs, feature_values, feature_costs):
        algo_runs = process_algo_runs(algo_runs)
        feature_values = process_feature_values(feature_values)
        feature_costs = process_feature_costs(feature_costs, algo_runs)

        return algo_runs, feature_values, feature_costs

    file_types = ['algorithm_runs', 'feature_values', 'feature_costs']
    dataframes = {}

    for file_type in file_types:
        for extension in ['csv', 'arff']:
            file_path = os.path.join(file_path_config["DATASET_PATH"], config["dataset"], f'{file_type}.{extension}')
            if Path(file_path).exists():
                dataframes[file_type] = read_file(file_path, extension)
                if file_type == 'feature_values':
                    dataframes[file_type].replace('?', np.NaN, inplace=True)
                break
        if file_type not in dataframes:
            if file_type == 'feature_costs':
                dataframes[file_type] = pd.DataFrame()

    return data_engineering(dataframes['algorithm_runs'], dataframes['feature_values'], dataframes['feature_costs'])


def process_arff_file(filename):
    with open(filename) as fp:
        try:
            arff_dict = arff.load(fp)
        except arff.BadNominalValue:
            logging.error(
                'Parsing of arff file failed (%s) - maybe conflict of header and data.', config["dataset"])

    col_list = [attr[0] for attr in arff_dict["attributes"]]
    df = pd.DataFrame(arff_dict["data"], columns=col_list)

    return df


def get_columns(df):
    return df.columns[1:]


def fill_missing_features():
    imputer = SimpleImputer(missing_values=np.nan, strategy=config["imputer_strategy"])
    
    train_df[feature_val_columns] = imputer.fit_transform(train_df[feature_val_columns])
    test_df[feature_val_columns] = imputer.transform(test_df[feature_val_columns])
    validation_df[feature_val_columns] = imputer.transform(validation_df[feature_val_columns])

    return train_df, validation_df, test_df


def feature_scaling():
    if config["feature_scaling_technique"] != "No scaling":
        if config["feature_scaling_technique"] == "StandardScaler":
            scaler = StandardScaler()
        elif config["feature_scaling_technique"] == "MinMaxScaler":
            scaler = MinMaxScaler()
        elif config["feature_scaling_technique"] == "RobustScaler":
            scaler = RobustScaler()

        train_df[feature_val_columns] = scaler.fit_transform(train_df[feature_val_columns])
        test_df[feature_val_columns] = scaler.transform(test_df[feature_val_columns])
        validation_df[feature_val_columns] = scaler.transform(validation_df[feature_val_columns])

    return train_df, validation_df, test_df


def algorithm_pre_selection(df):
    if config["pre_selection_technique"] == "VBest_contribution":
        selected_algorithms = df[algo_columns].idxmin(axis=1).value_counts().head(config["pre_selection_size"]).index.tolist()
    elif config["pre_selection_technique"] == "Minimum_runtime":
        algo_run_dict = df[algo_columns].sum().to_dict()
        algo_list = list(
            dict(sorted(algo_run_dict.items(), key=lambda item: item[1])).keys())
        selected_algorithms = algo_list[:config["pre_selection_size"]]
    else:
        selected_algorithms = algo_columns.values

    return selected_algorithms


def save_json_file(json_path_name, data):
    try:
        if not Path(json_path_name).exists() or Path(json_path_name).stat().st_size == 0:
            with open(json_path_name, "w") as outfile:
                json.dump(data, outfile, indent=4)
    except Exception as e:
        logger.error(f"Error while saving JSON file: {e}")


def get_virtual_best_solver_time(df, experiment_name, data_path):
    vbTime = round(df[algo_columns].replace(config["max_timeout"], config["max_timeout"] * config["penalty"]).min(axis=1).sum() + df[feature_costs_columns].sum().values[0], 2)
    
    algo_list = df[algo_columns].idxmin(axis=1).values
    algo_freq = collections.Counter(algo_list)

    logger.info(f"Virtual best solver time in {experiment_name} = {round(vbTime, 2)}")

    json_path_name = os.path.join(
        data_path, f'virtual_best_{experiment_name}.json')

    save_json_file(json_path_name, algo_freq)

    return round(vbTime, 2)


def get_single_best_solver_time(df, experiment_name, data_path):
    timeAlgosTrain = round(df[algo_columns].replace(config["max_timeout"], config["max_timeout"] * config["penalty"]).sum(), 2).to_dict()

    best_solver_name, best_solver_time = min(
        timeAlgosTrain.items(), key=lambda x: x[1])

    logger.info(f"Single best solver list {experiment_name} = {timeAlgosTrain}")
    logger.info(f"Single best solver time {experiment_name} = {best_solver_name} ({round(best_solver_time, 2)})")

    json_path_name = os.path.join(
        data_path, f'single_best_{experiment_name}.json')

    save_json_file(json_path_name, timeAlgosTrain)

    return best_solver_time


def calculate_solver_times(df, runtime_dict, experiment_name, data_path):
    vb_st = get_virtual_best_solver_time(
        df, experiment_name, data_path)
    sb_st = get_single_best_solver_time(df, experiment_name, data_path)
    
    runtime_dict["virtual_best"] = vb_st
    runtime_dict["single_best"] = sb_st


def algorithm_label_transform(df, algo_pairs, encoder):
    y = df[algo_pairs].idxmin(axis=1).values
    y = encoder.transform(y)

    return y

def timeout_label_transform(df, encoder):    
    y = df["Timeout"].values
    y = encoder.transform(y)

    return y


def find_runtime_for_each_row(row, algorithm_predictor_dict):
    pred_list = []

    filtered_algorithm_predictors = algorithm_predictor_dict.values()

    if config["timeout_predictor_usage"] == "Yes":
        non_timeout_algorithms = [algo for algo in algo_columns if row[f"{algo}_Timeout"] != f'{algo}_Timeout']

        if len(non_timeout_algorithms) != 0:
            filtered_algorithm_predictors = [algo_rf for algo_rf in algorithm_predictor_dict.values() if set(algo_rf.get_model_name()).intersection(non_timeout_algorithms)]

    for algo_rf in filtered_algorithm_predictors:
        pred_list.append(algo_rf.get_encoder().inverse_transform(algo_rf.predict(pd.DataFrame(row).transpose()[feature_val_columns]))[0])

    return row[mode(pred_list)]


def hard_voting_mechanism(df, algorithm_predictor_dict, algo_columns, timeout, penalty, timeout_predictor_dict = None):
    df_copy = df.copy()
    df_copy[algo_columns] = df_copy[algo_columns].replace(timeout, timeout * penalty)

    if config["timeout_predictor_usage"] == "Yes":
        for timoeut_algo, timeout_predictor in timeout_predictor_dict.items():
            df_copy[f"{timoeut_algo}_Timeout"] = timeout_predictor.get_encoder().inverse_transform(timeout_predictor.predict(df_copy[feature_val_columns]))
    
    runtime = sum(df_copy.apply(lambda row: find_runtime_for_each_row(row, algorithm_predictor_dict), axis=1))

    return round(runtime, 2)


def passive_learning_algorithm_predictor(train_df, selected_algorithm_pairs):
    algorithm_predictor_dict = {}
    
    for algo in selected_algorithm_pairs:
        algorithm = [*algo]

        algorithm_encoder = preprocessing.LabelEncoder()
        algorithm_encoder.fit(algo)

        train_df_to_discarded = train_df[train_df[algorithm].sum(axis=1) < config["max_timeout"] * len(algorithm)]
        
        passive_algorithm_rf = PassiveRFModel(RandomForestClassifier(**config["param_grid"], random_state=round(1000000 * random.random())), 
                                              "Algorithm_predictor", 
                                              algorithm, 
                                              algorithm_encoder, 
                                              train_df_to_discarded, 
                                              algorithm_label_transform(train_df_to_discarded, algorithm, algorithm_encoder))
        
        passive_algorithm_rf.fit(feature_val_columns, config["weight"], config["max_timeout"], config["penalty"])
        
        algorithm_predictor_dict[algo] = passive_algorithm_rf

    return algorithm_predictor_dict


def passive_learning_timeout_predictor(train_df, selected_algorithms):
    timeout_predictor_list = {}

    for algo in selected_algorithms:
        timeout_list = [algo, f"{algo}_Timeout"]

        timeout_encoder = preprocessing.LabelEncoder()
        timeout_encoder.fit(timeout_list)

        train_timeout_copy = train_df.copy()
        train_timeout_copy["Timeout"] = train_timeout_copy[algo].apply(lambda x: f"{algo}_Timeout" if x >= config["max_timeout"] else algo)

        timeout_rf = PassiveRFModel(RandomForestClassifier(**config["param_grid"], random_state=round(1000000 * random.random())), 
                                    "Timeout_predictor", 
                                    timeout_list, 
                                    timeout_encoder, 
                                    train_timeout_copy, 
                                    timeout_label_transform(train_timeout_copy, timeout_encoder))
        timeout_rf.fit(feature_val_columns)
        
        timeout_predictor_list[algo] = timeout_rf

    return timeout_predictor_list


def passive_learning(train_df, validation_df, test_df):
    passive_learning_algorithms = algorithm_pre_selection(train_df)
    passive_algo_pair_comb_list = list(itertools.combinations(passive_learning_algorithms, 2))  

    passive_algorithm_predictor_dict = passive_learning_algorithm_predictor(train_df, passive_algo_pair_comb_list)

    if config["timeout_predictor_usage"] == "Yes":
        passive_timeout_predictor_dict = passive_learning_timeout_predictor(train_df, passive_learning_algorithms)
        
        if config["voting"] == "Hard":
            validation_runtime = hard_voting_mechanism(validation_df, passive_algorithm_predictor_dict, passive_learning_algorithms, config["max_timeout"], config["penalty"], passive_timeout_predictor_dict) + train_df[feature_costs_columns].sum().sum()
            test_runtime = hard_voting_mechanism(test_df, passive_algorithm_predictor_dict, passive_learning_algorithms, config["max_timeout"], config["penalty"], passive_timeout_predictor_dict) + train_df[feature_costs_columns].sum().sum()
    else:
        if config["voting"] == "Hard":
            validation_runtime = hard_voting_mechanism(validation_df, passive_algorithm_predictor_dict, passive_learning_algorithms, config["max_timeout"], config["penalty"]) + train_df[feature_costs_columns].sum().sum()
            test_runtime = hard_voting_mechanism(test_df, passive_algorithm_predictor_dict, passive_learning_algorithms, config["max_timeout"], config["penalty"]) + train_df[feature_costs_columns].sum().sum()
    instance_cost = round(train_df[algo_columns].sum().sum(), 2)
    
    logger.info(f"Passive Learning Runtime on Validation Set: {round(validation_runtime, 2)}")
    logger.info(f"Passive Learning Runtime on Test Set: {round(test_runtime, 2)}")
    logger.info(f"Passive Learning Instance Cost: {round(instance_cost, 2)}")

    return instance_cost, validation_runtime, test_runtime


def create_instance_cost_table(initial_train_df, timeout_limit):
    melted_df = initial_train_df[algo_columns].reset_index().melt(id_vars=["instance_id"], var_name="Algorithm", value_name="Runtime")
    melted_df["Runstatus"] = np.where(melted_df["Runtime"]  == config["max_timeout"], False, melted_df["Runtime"] <= timeout_limit)
    melted_df["Timeout"] = timeout_limit
    melted_df["Runtime"] = melted_df["Runtime"].clip(upper = timeout_limit)

    return melted_df.sort_values(by=['instance_id', 'Algorithm'])


def initial_train_data_selection(train_df, timeout_limit):
    initial_train_df = train_df.sample(n=config["initial_train_size"], random_state=round(1000000 * random.random()))

    instance_cost_df = create_instance_cost_table(initial_train_df, timeout_limit)

    initial_train_df[algo_columns] = initial_train_df[algo_columns].clip(upper = timeout_limit)
    pool_df = train_df.drop(initial_train_df.index)

    return initial_train_df, pool_df, instance_cost_df


def update_instance_cost_dictionary(instance_cost_table, pool_cost, algorithms, timeout_limit):
    instance_pool_df =  pool_cost[algorithms].reset_index().melt(id_vars=["instance_id"], var_name="Algorithm", value_name="Runtime")
    
    instance_pool_df["Runstatus"] = np.where(instance_pool_df["Runtime"] == config["max_timeout"], False, instance_pool_df["Runtime"] <= timeout_limit)
    instance_pool_df["Timeout"] = timeout_limit
    instance_pool_df["Runtime"] = instance_pool_df["Runtime"].clip(upper = timeout_limit)

    combined_df = pd.concat([instance_pool_df, instance_cost_table])
    combined_df = combined_df.sort_values(by=['instance_id', 'Algorithm', 'Runstatus', 'Runtime'], ascending=[True, True, False, False])
    
    result_df = combined_df.groupby(['instance_id', 'Algorithm'], as_index=False).first()

    return result_df


def create_uncertainty_table(algorithm_predictor_dict, timeout_limit, timeout_predictor_dict = None):
    uncertainty_table = pd.DataFrame(columns=['Model', 'InstanceID', 'Uncertainty', 'Cost', "Algorithm1_RS", "Algorithm2_RS", "Algorithm1_TO", "Algorithm2_TO"])

    for algo_index, active_algo_predictor in algorithm_predictor_dict.items():
        if active_algo_predictor.get_initial_train_data().empty or active_algo_predictor.get_pool_data().empty:
            continue
        
        uncertainty_information = classifier_uncertainty(active_algo_predictor.get_learner(), active_algo_predictor.get_pool_data()[feature_val_columns])

        model_based_uncertainty_table = pd.DataFrame()

        model_based_uncertainty_table["Model"] = [algo_index for _ in range(len(active_algo_predictor.get_pool_data()))]
        model_based_uncertainty_table["InstanceID"] = active_algo_predictor.get_pool_data().index
        model_based_uncertainty_table["Uncertainty"] = uncertainty_information
        model_based_uncertainty_table["Cost"] = active_algo_predictor.get_pool_data()[[*active_algo_predictor.get_model_name()]].clip(upper = timeout_limit).sum(axis=1).values
        model_based_uncertainty_table["Algorithm1_RS"] = np.where(active_algo_predictor.get_pool_data()[algo_index[0]] == config["max_timeout"], False, active_algo_predictor.get_pool_data()[algo_index[0]] <= timeout_limit)
        model_based_uncertainty_table["Algorithm2_RS"] = np.where(active_algo_predictor.get_pool_data()[algo_index[1]] == config["max_timeout"], False, active_algo_predictor.get_pool_data()[algo_index[1]] <= timeout_limit)

        if config["timeout_predictor_usage"] == "Yes":
            model_based_uncertainty_table["Algorithm1_TO"] = timeout_predictor_dict[algo_index[0]].predict(active_algo_predictor.get_pool_data()[feature_val_columns])
            model_based_uncertainty_table["Algorithm2_TO"] = timeout_predictor_dict[algo_index[1]].predict(active_algo_predictor.get_pool_data()[feature_val_columns])
        
        uncertainty_table = pd.concat([uncertainty_table, model_based_uncertainty_table], axis=0)

    return uncertainty_table.sort_values(by='Uncertainty', ascending=False)


def find_cost_free_instances(instance_cost_table, algorithm_predictor_dict, current_timeout_limit):
    for algo_index, active_algo_predictor in algorithm_predictor_dict.items():
        instances_with_two_algos = instance_cost_table[instance_cost_table["Algorithm"].isin(algo_index)].groupby('instance_id').filter(lambda x: len(x) >= 2)

        instances_with_true_status = instances_with_two_algos.groupby('instance_id').filter(lambda x: x['Runstatus'].any())["instance_id"].unique()

        free_indexes = active_algo_predictor.get_pool_data()[active_algo_predictor.get_pool_data().index.isin(instances_with_true_status)].index

        if not free_indexes.empty:
            active_algo_predictor.update_initial_train_data_and_pool(free_indexes, config["weight"], current_timeout_limit, config["penalty"])
        
        if current_timeout_limit == config["max_timeout"]:
            instances_with_false_status = instances_with_two_algos.groupby('instance_id').filter(lambda x: not x['Runstatus'].any())["instance_id"].unique()
            free_double_to_indexes = active_algo_predictor.get_pool_data()[active_algo_predictor.get_pool_data().index.isin(instances_with_false_status)].index
            
            if not free_double_to_indexes.empty:
                active_algo_predictor.delete_from_pool(free_double_to_indexes)


def active_learning(train_df, validation_df, test_df):
    instance_cost_history = []

    runtime_val_history = []
    runtime_test_history = []

    current_timeout_limit = config["timeout_limit"]

    initial_train_df, pool_df, instance_cost_table = initial_train_data_selection(train_df, current_timeout_limit)

    active_learning_algorithms = algorithm_pre_selection(initial_train_df)
    active_algo_pair_comb_list = list(itertools.combinations(active_learning_algorithms, 2))

    active_algorithm_predictor_dict = active_learning_algorithm_predictor(initial_train_df, pool_df, train_df, active_algo_pair_comb_list, current_timeout_limit)
    active_algorithm_predictor_dict_filtered = {algo: predictor for algo, predictor in active_algorithm_predictor_dict.items() if len(predictor.get_initial_train_data()) != 0}

    if config["timeout_predictor_usage"] == "Yes":
        active_timeout_predictor_dict = active_learning_timeout_predictor(train_df, instance_cost_table, active_learning_algorithms, current_timeout_limit)
        
        if config["voting"] == "Hard":
            initial_val_runtime = hard_voting_mechanism(validation_df, active_algorithm_predictor_dict_filtered, active_learning_algorithms ,config["max_timeout"], config["penalty"], active_timeout_predictor_dict) + train_df[feature_costs_columns].sum().sum()
            initial_test_runtime = hard_voting_mechanism(test_df, active_algorithm_predictor_dict_filtered, active_learning_algorithms, config["max_timeout"], config["penalty"], active_timeout_predictor_dict) + train_df[feature_costs_columns].sum().sum()
    else:
        if config["voting"] == "Hard":
            initial_val_runtime = hard_voting_mechanism(validation_df, active_algorithm_predictor_dict_filtered, active_learning_algorithms, config["max_timeout"], config["penalty"]) + train_df[feature_costs_columns].sum().sum()
            initial_test_runtime = hard_voting_mechanism(test_df, active_algorithm_predictor_dict_filtered, active_learning_algorithms, config["max_timeout"], config["penalty"]) + train_df[feature_costs_columns].sum().sum()

    initial_instance_cost = round(instance_cost_table["Runtime"].sum() , 2)

    logger.info(f"Initial Active Learning Runtime on Validation Set: {round(initial_val_runtime, 2)}")
    logger.info(f"Initial Active Learning Runtime on Test Set: {round(initial_test_runtime, 2)}")
    logger.info(f"Initial Active Learning Instance Cost: {round(initial_instance_cost, 2)}")

    runtime_val_history.append(initial_val_runtime)
    runtime_test_history.append(initial_test_runtime)
    instance_cost_history.append(initial_instance_cost)

    bin_size = math.ceil(len(pool_df) * len(active_algo_pair_comb_list) * config["query_size"] / 100)
    logger.info(f"Bin size = {bin_size}")

    i = 1

    while(not all(active_algo_rf.get_pool_data().empty for active_algo_rf in active_algorithm_predictor_dict.values())):
        if config["timeout_predictor_usage"] == "Yes":
            uncertainty_table = create_uncertainty_table(active_algorithm_predictor_dict, current_timeout_limit, active_timeout_predictor_dict)
        else:
            uncertainty_table = create_uncertainty_table(active_algorithm_predictor_dict, current_timeout_limit)
        
        logger.info(f"Query {i} Size of the uncertainty table is = {uncertainty_table.shape[0]}/{len(pool_df) * len(active_algo_pair_comb_list)}")
        
        filtered_table = uncertainty_table.head(bin_size)

        if config["timeout_predictor_usage"] == "Yes":
            filtered_table = uncertainty_table[~((uncertainty_table['Algorithm1_TO'] == 1) & (uncertainty_table['Algorithm2_TO'] == 1))].head(bin_size)

        if filtered_table.empty:
            filtered_table = uncertainty_table.head(bin_size)

        logger.info(f"Query {i} Size of the filtered table is = {filtered_table.shape[0]}")
        
        for model_index in filtered_table["Model"].unique():
            model_table = filtered_table[filtered_table['Model'] == model_index]

            algo_predictor = active_algorithm_predictor_dict[model_index]
            instance_cost_table = update_instance_cost_dictionary(instance_cost_table, algo_predictor.get_pool_data().loc[model_table['InstanceID'].values][[*model_index]], [*model_index], current_timeout_limit)

            model_table_to_discarded = model_table[(model_table["Algorithm1_RS"] == True) | (model_table["Algorithm2_RS"] == True)]
            
            if not model_table_to_discarded.empty:
                algo_predictor.update_initial_train_data_and_pool(model_table_to_discarded['InstanceID'].values, config["weight"], current_timeout_limit, config["penalty"])

            if current_timeout_limit == config["max_timeout"]:
                model_table_to = model_table[(model_table["Algorithm1_RS"] == False) & (model_table["Algorithm2_RS"] == False)]
                
                if not model_table_to.empty:
                    algo_predictor.delete_from_pool(model_table_to['InstanceID'].values)

        find_cost_free_instances(instance_cost_table, active_algorithm_predictor_dict, current_timeout_limit)

        for algo_index, active_algo_predictor in active_algorithm_predictor_dict.items():
            active_algo_predictor.teach(feature_val_columns, config["weight"])

        active_algorithm_predictor_dict_filtered = {algo: predictor for algo, predictor in active_algorithm_predictor_dict.items() if len(predictor.get_initial_train_data()) != 0}
        
        if config["timeout_predictor_usage"] == "Yes":
            active_timeout_predictor_dict = active_learning_timeout_predictor(train_df, instance_cost_table, active_learning_algorithms, current_timeout_limit)
            
            if config["voting"] == "Hard":
                query_val_runtime = hard_voting_mechanism(validation_df, active_algorithm_predictor_dict_filtered, active_learning_algorithms, config["max_timeout"], config["penalty"], active_timeout_predictor_dict) + train_df[feature_costs_columns].sum().sum()
                query_test_runtime = hard_voting_mechanism(test_df, active_algorithm_predictor_dict_filtered, active_learning_algorithms, config["max_timeout"], config["penalty"], active_timeout_predictor_dict) + train_df[feature_costs_columns].sum().sum()
        else:
            if config["voting"] == "Hard":
                query_val_runtime = hard_voting_mechanism(validation_df, active_algorithm_predictor_dict_filtered, active_learning_algorithms, config["max_timeout"], config["penalty"]) + train_df[feature_costs_columns].sum().sum()
                query_test_runtime = hard_voting_mechanism(test_df, active_algorithm_predictor_dict_filtered, active_learning_algorithms, config["max_timeout"], config["penalty"]) + train_df[feature_costs_columns].sum().sum()

        query_instance_cost = round(instance_cost_table["Runtime"].sum() , 2)

        logger.info(f"Query {i} Active Learning Runtime on Validation Set: {query_val_runtime}")
        logger.info(f"Query {i} Active Learning Runtime on Test Set: {query_test_runtime}")
        logger.info(f"Query {i} Active Learning Instance Cost: {query_instance_cost}")
        
        if query_val_runtime >= runtime_val_history[-1]:
            current_timeout_limit = min(config["max_timeout"], current_timeout_limit + config["timeout_increase_rate"])
            logger.info(f"Timeout limit is increased to {current_timeout_limit}")

        runtime_val_history.append(query_val_runtime)
        runtime_test_history.append(query_test_runtime)
        instance_cost_history.append(query_instance_cost)

        i += 1

    return instance_cost_history, runtime_val_history, runtime_test_history


def active_learning_algorithm_predictor(initial_train_df, pool_df, train_df, selected_algorithm_pairs, timeout_limit):
    algorithm_predictor_dict = {}

    for algo in selected_algorithm_pairs:
        algorithm = [*algo]
        algorithm_encoder = preprocessing.LabelEncoder()
        algorithm_encoder.fit(algo)

        train_df_to_discarded = initial_train_df[initial_train_df[algorithm].sum(axis=1) < timeout_limit * len(algorithm)]
        double_to_indexes = initial_train_df[initial_train_df[algorithm].sum(axis=1) == timeout_limit * len(algorithm)].index

        if timeout_limit == config["max_timeout"]:
            pool_df_to_added = pool_df.copy(deep=True)
        else:
            pool_df_to_added = pd.concat([pool_df.copy(deep=True), train_df.loc[double_to_indexes]], axis=0)

        if config["weight"] == "Weighted":
            sample_weight = (train_df_to_discarded.replace(timeout_limit, timeout_limit * config["penalty"])[algorithm[0]] - train_df_to_discarded.replace(timeout_limit, timeout_limit * config["penalty"])[algorithm[1]]).abs().values

            active_algorithm_rf = ActiveRFModel(ActiveLearner(estimator = RandomForestClassifier(**config["param_grid"], random_state=round(1000000 * random.random())), 
                                                            query_strategy = uncertainty_sampling,
                                                            X_training = train_df_to_discarded[feature_val_columns], 
                                                            y_training = algorithm_label_transform(train_df_to_discarded, algorithm, algorithm_encoder), 
                                                            sample_weight = sample_weight), 
                                                            "Algorithm_predictor", algo, algorithm_encoder, train_df_to_discarded, pool_df_to_added, sample_weight)
        else:
            active_algorithm_rf = ActiveRFModel(ActiveLearner(estimator = RandomForestClassifier(**config["param_grid"], random_state=round(1000000 * random.random())), 
                                                            query_strategy = uncertainty_sampling,
                                                            X_training = train_df_to_discarded[feature_val_columns], 
                                                            y_training = algorithm_label_transform(train_df_to_discarded, algorithm, algorithm_encoder)), 
                                                            "Algorithm_predictor", algo, algorithm_encoder, train_df_to_discarded, pool_df_to_added)
        algorithm_predictor_dict[algo] = active_algorithm_rf

    return algorithm_predictor_dict


def active_learning_timeout_predictor(train_df, instance_cost_table, selected_algorithms, timeout_limit):
    timeout_predictor_dict = {}

    for algo in selected_algorithms:
        timeout_list = [algo, f"{algo}_Timeout"]

        timeout_encoder = preprocessing.LabelEncoder()
        timeout_encoder.fit(timeout_list)
        
        instance_cost_filtered = instance_cost_table[((instance_cost_table["Timeout"] == timeout_limit) | (instance_cost_table["Runstatus"] == True)) & (instance_cost_table["Algorithm"] == algo)]
        instance_cost_filtered.set_index("instance_id", inplace = True, drop = True)

        train_timeout_copy = train_df.loc[instance_cost_filtered.index].copy()
        train_timeout_copy[selected_algorithms] = train_timeout_copy[selected_algorithms].clip(upper = timeout_limit)
        
        train_timeout_copy.loc[instance_cost_filtered.index, "Timeout"] = instance_cost_filtered["Runstatus"].apply(lambda x: f"{algo}_Timeout" if x == False else algo)
        
        active_timeout_rf = ActiveRFModel(ActiveLearner(estimator=RandomForestClassifier(**config["param_grid"], random_state=round(1000000 * random.random())), 
                                                        query_strategy=uncertainty_sampling,
                                                        X_training=train_timeout_copy[feature_val_columns], 
                                                        y_training = timeout_label_transform(train_timeout_copy, timeout_encoder)), 
                                                        "Timeout_predictor", timeout_list, timeout_encoder, train_timeout_copy, None)
        
        timeout_predictor_dict[algo] = active_timeout_rf

    return timeout_predictor_dict


def save_results():
    file_name_dict = {"runtime_test_score": runtime_test_dict, 
                      "runtime_validation_score": runtime_validation_dict,
                      "instance_cost": instance_cost_dict}

    for keys in file_name_dict.keys():
        json_path_name = data_path + keys + '.json'
        save_json_file(json_path_name, file_name_dict[keys])
        logger.info('%s is written as json file', keys)


def save_json_file(json_path_name, data):
    try:
        if not Path(json_path_name).exists() or Path(json_path_name).stat().st_size == 0:
            with open(json_path_name, "w") as outfile:
                json.dump(data, outfile, indent=4)
    except Exception as e:
        logger.error(f"Error while saving JSON file: {e}")


algo_runs, feature_values, feature_costs = read_dataset()

algo_columns = get_columns(algo_runs)
feature_val_columns = get_columns(feature_values)
feature_costs_columns = get_columns(feature_costs)

df_list = [algo_runs, feature_values, feature_costs]
df_final = reduce(lambda left, right: pd.merge(
    left, right, on='instance_id'), df_list).reset_index(drop=True)

save_json_file(file_path_config["RESULT_PATH"] +
               "merged_data.json", df_final.to_dict())

kf = KFold(n_splits=config["n_splits"], shuffle=True,
           random_state=round(1000000 * random.random()))
kf_list = list(kf.split(df_final))

train_index, test_index = kf_list[config["split_number"]]
train_index, validation_index = np.split(train_index, [int(0.8 * len(train_index))])

data_path = file_path_config["SPLIT_PATH"] + "DATA_DIR/"
output_path = file_path_config["SPLIT_PATH"] + "OUTPUT_DIR/"

Path(data_path).mkdir(parents=True, exist_ok=True)
Path(output_path).mkdir(parents=True, exist_ok=True)

runtime_train_dict = {}
runtime_validation_dict = {}
runtime_test_dict = {}
instance_cost_dict = {}

train_df, validation_df, test_df = df_final.loc[train_index].reset_index(
    drop=True), df_final.loc[validation_index].reset_index(drop=True), df_final.loc[test_index].reset_index(drop=True)

train_df, validation_df, test_df = fill_missing_features()
train_df, validation_df, test_df = feature_scaling()
train_df.set_index("instance_id", inplace = True, drop = True), validation_df.set_index("instance_id", inplace = True, drop = True), test_df.set_index("instance_id", inplace = True, drop = True)

calculate_solver_times(train_df, runtime_train_dict, "train", data_path)
calculate_solver_times(validation_df, runtime_validation_dict, "validation", data_path)
calculate_solver_times(test_df, runtime_test_dict, "test", data_path)

passive_instance_cost, passive_runtime_val_history, passive_runtime_test_history = passive_learning(train_df, validation_df, test_df)
active_instance_cost_history, active_runtime_val_history, active_runtime_test_history = active_learning(train_df, validation_df, test_df)

instance_cost_dict["passive_learning"] = passive_instance_cost
instance_cost_dict["active_learning"] = active_instance_cost_history

runtime_test_dict["passive_learning"] = passive_runtime_test_history
runtime_test_dict["active_learning"] = active_runtime_test_history

runtime_validation_dict["passive_learning"] = passive_runtime_val_history
runtime_validation_dict["active_learning"] = active_runtime_val_history

save_results()

logger.info('Experiment is finished')