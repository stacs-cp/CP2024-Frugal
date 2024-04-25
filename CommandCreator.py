import math
from itertools import product
from pathlib import Path

dataset_path = Path('DATASETS/')

script_list = ["Algorithm_Selection_Random_Query.py"]
dataset_list = sorted([f.name for f in dataset_path.iterdir() if f.is_dir()])
dataset_list = ["CPMP-2015", "CSP-2010", "MAXSAT12-PMS", "MAXSAT19-UCMS", "QBF-2011", "ASP-POTASSCO"]

seed_list = [7, 42, 99, 123, 12345]
split_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

comb_list = list(product(script_list, dataset_list, seed_list, split_list))
sub_combs = [comb_list[i:i + 500] for i in range(0, len(comb_list), 500)]

file_count = math.ceil((len(comb_list)) / 500)

for i in range(file_count):
    filename = f'commands_rq.txt'

    f = open(filename, "w")

    for combs in comb_list:
        command = f'python {combs[0]} --dataset {combs[1]} --prefix _QS_1_RQ_100_NO_WEIGHT_NO_TIMEOUT_PREDICTOR --weight No_Weight --timeout_predictor_usage No --query_size 1 --seed {combs[2]} --split {combs[3]}\n'
        f.write(command)
        command = f'python {combs[0]} --dataset {combs[1]} --prefix _QS_1_RQ_100_NO_WEIGHT_WITH_TIMEOUT_PREDICTOR --weight No_Weight --timeout_predictor_usage Yes --query_size 1 --seed {combs[2]} --split {combs[3]}\n'
        f.write(command)
        command = f'python {combs[0]} --dataset {combs[1]} --prefix _QS_1_RQ_3600_NO_WEIGHT_NO_TIMEOUT_PREDICTOR --weight No_Weight --timeout_predictor_usage No --query_size 1 --timeout_limit 3600 --seed {combs[2]} --split {combs[3]}\n'
        f.write(command)
        command = f'python {combs[0]} --dataset {combs[1]} --prefix _QS_1_RQ_3600_NO_WEIGHT_WITH_TIMEOUT_PREDICTOR --weight No_Weight --timeout_predictor_usage Yes --query_size 1 --timeout_limit 3600 --seed {combs[2]} --split {combs[3]}\n'
        f.write(command)