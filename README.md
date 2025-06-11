# Frugal Algorithm Selection # 

Frugal algorithm selection is an active learning approach that attempts to reduce the labelling cost by using only a subset of the training data with timeout predictor and dynamic timeout configurations.

## Directory Structure ##
Below is an overview of the main folders in this repository and their contents:

- __`/DATASETS`__: Contains all datasets that were used in the research.
- __`/EXPERIMENT_LOGS`__: Includes Slurm output files of each implemented approach under various configurations.
- __`/EXPERIMENT_OUTPUTS`__: Includes output files of each implemented approach under various configurations.
- __`/PLOTS`__: Holds all plots that are included in the published paper and its appendix.
- __`Appendix.pdf`__: This file is supplementary material accompanying the paper. It provides additional data, analyses, and explanations that support the main text of the research.

## Installation ## 

### Requirements ###
Ensure you have Python 3.12 or higher installed on your machine. The dependencies are listed below and can also be found in the requirements.txt file:

- liac-arff==2.5.0
- matplotlib==3.8.0
- modAL-python==0.4.2.1
- numpy==1.26.1
- pandas==2.1.1
- scikit-learn==1.3.1

## Installing Dependencies ##
To install the required packages with the specified versions, run the following command:

```
pip install -r requirements.txt
```

This ensures you have the exact environment used in our studies, promoting consistency and reproducibility.

# Features: #

## Timeout Predictor: ##
__Timeout Predictor On/Off:__ It can be activated/deactivated with arameter used in the voting mechanism and Run selection

## Dynamic Timeout: ##
__Adjustable Timeout:__ The dynamic timeout increases during runtime based on performance on the validation set and can be adjusted through parameters to suit different experimental setups.

## Query Size: ##
__Customizable Ratio:__ Adjust the query size ratio through parameters to optimize the querying process in learning algorithms.

## Run Selection Approach ## 
__Multiple Run Selection Metrics:__ Supports uncertainty-based (leveraging modAL), random, Pareto optimization, and lexicographic ordering approaches.

# Usage # 

## Example Run ##

### Uncertainty-based Run Selection ###
To run an uncertainty-based Run selection, which includes passive learning approaches, use the following command:

```
python Algorithm_Selection.py --dataset {dataset} --prefix {prefix} --weight {Weighted, No_Weight} --timeout_predictor_usage {Yes, No} --query_size {query_size} --seed {seed} --split {split}
```

### Random Run Selection ###
For running the random Run selection with the same configurable parameters:

```
python Algorithm_Selection_Random_Query.py --dataset {dataset} --prefix {prefix} --weight {Weighted, No_Weight} --timeout_predictor_usage {Yes, No} --query_size {query_size} --seed {seed} --split {split}
```

### Pareto Selection
For running the Pareto selection with the same configurable parameters, only differs in multi-objective optimization features and which direction they need to be optimized: Uncertainty:max,PredDiff:max,PredCost:min

```
python Algorithm_Selection_Pareto.py --dataset {dataset} --prefix {prefix} --weight {Weighted, No_Weight} --timeout_predictor_usage {Yes, No} --pareto_features {feature:direction,feature:direction} --query_size {query_size} --timeout_limit {timeout_limit} --penalty_type {penalty_type} --seed {seed} --split {split}
```

### Lexical Selection
For running the lexicographic ordering selection with the same configurable parameters, only differs in priority-based selection metrics: Uncertainty:max,PredDiff:max,PredCost:min

```
python Algorithm_Selection_Lexical_Ordering.py --dataset {dataset} --prefix {prefix} --weight {Weighted, No_Weight} --timeout_predictor_usage {Yes, No} --priority_1 {feature:direction} --priority_2 {feature:direction} --priority_3 {feature:direction} --query_size {query_size} --timeout_limit {timeout_limit} --penalty_type {penalty_type} --seed {seed} --split {split}
```

# Post-Processing # 
- After running scripts, log files can be used to create CSV files in order to visualize selection metric performance
- ExperimentalStatus.py can be used for getting experimental status table and experiment values in merged_results.csv
- To use plotting scripts, merged_results.csv should be normalized into normalized_results.csv using PlotExperimentalResults.py

# Visualization # 
- PlotExperimentalResults.py for normalize experiment results and plotting converge trend line charts
- AblationStudy.py for ablation study of components
- Nemenyi.py for CD diagram for different prediction power ranking
- PlotHeatmap.py for plotting heatmap with std
- Uncertainty_Measurement_Comparison.py for uncertainty measurements (LC, ES, MS) comparison in binary classification

Further parameter details and their descriptions can be found within the code comments or the supplementary documentation provided in the repository.