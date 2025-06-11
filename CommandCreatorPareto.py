from itertools import product, combinations

# Script and dataset information
script_list = ["Algorithm_Selection_Pareto.py"]
dataset_list = ["ASP-POTASSCO", "BNSL-2016", "CPMP-2015", "CSP-2010", "CSP-MZN-2013", 
                "MAXSAT-WPMS-2016", "MAXSAT12-PMS", "MAXSAT15-PMS-INDU", "MAXSAT19-UCMS", 
                "QBF-2011", "QBF-2014", "QBF-2016", "SAT11-HAND", "SAT11-RAND", 
                "SAT12-HAND", "SAT12-INDU"]
priority_list = ["Uncertainty:max", "PredDiff:max", "PredCost:min"]
penalty_type_list = ["Only-timeout-penalty"]
timeout_limit_list = [100, 3600]
seed_list = [7, 42, 99, 123, 12345]
split_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

priority_combinations = list(combinations(priority_list, 3)) + list(combinations(priority_list, 2))
priority_text_combinations = [",".join(comb) for comb in priority_combinations]

comb_list = list(product(script_list, dataset_list, priority_text_combinations, penalty_type_list,
                        timeout_limit_list, seed_list, split_list))

filename = 'commands_pareto.txt'
with open(filename, "w") as f:
    for combs in comb_list:
        priorities = combs[2].split(",")  # Split the priority text combination
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
        
        # Generate the prefix based on the number of priorities
        priority_prefix = "_".join(p.split(":")[0] for p in priorities)
        
        command = (
            f'python {combs[0]} --dataset {combs[1]} --prefix AS_PARETO{suffix}_{priority_prefix} --weight No_Weight '
            f'--timeout_predictor_usage No --pareto_features {combs[2]} '
            f'--query_size 1 --timeout_limit {timeout_limit} --penalty_type {penalty_type} '
            f'--seed {combs[5]} --split {combs[6]}\n'
        )
        f.write(command)
