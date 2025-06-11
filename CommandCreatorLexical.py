from itertools import product, permutations
from pathlib import Path

# Script and dataset information
script_list = ["Algorithm_Selection_Lexical_Ordering.py"]
dataset_list = ["ASP-POTASSCO", "BNSL-2016", "CPMP-2015", "CSP-2010", "CSP-MZN-2013", 
                "MAXSAT-WPMS-2016", "MAXSAT12-PMS", "MAXSAT15-PMS-INDU", "MAXSAT19-UCMS", 
                "QBF-2011", "QBF-2014", "QBF-2016", "SAT11-HAND", "SAT11-RAND", 
                "SAT12-HAND", "SAT12-INDU"]
priority_list = ["Uncertainty:max", "PredDiff:max", "PredCost:min"]
penalty_type_list = ["Only-timeout-penalty"]
timeout_limit_list = [100, 3600]
seed_list = [7, 42, 99, 123, 12345]
split_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

priority_permutations = list(permutations(priority_list, 3))
comb_list = list(product(script_list, dataset_list, priority_permutations, penalty_type_list,
                        timeout_limit_list, seed_list, split_list))

filename = 'commands_lexical.txt'
with open(filename, "w") as f:
    for combs in comb_list:
        priorities = combs[2]  # priority_permutations at index 2
        timeout_limit = combs[4]
        penalty_type = combs[3]
        
        # Determine the suffix for prefix
        suffix = ""
        if timeout_limit == 100:
            suffix += "_DT"
        if penalty_type == "Only-timeout-penalty":
            suffix += "_OTP"
        elif penalty_type == "Dynamic-penalty":
            suffix += "_DP"
        
        command = (
            f'python {combs[0]} --dataset {combs[1]} --prefix AS_LEXICAL{suffix}_{priorities[0].split(":")[0]}_{priorities[1].split(":")[0]}_{priorities[2].split(":")[0]} --weight No_Weight '
            f'--timeout_predictor_usage No --priority_1 {priorities[0]} --priority_2 {priorities[1]} '
            f'--priority_3 {priorities[2]} --query_size 1 --timeout_limit {timeout_limit} --penalty_type {penalty_type} '
            f'--seed {combs[5]} --split {combs[6]}\n'
        )
        f.write(command)
