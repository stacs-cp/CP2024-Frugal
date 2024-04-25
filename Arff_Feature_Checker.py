from pathlib import Path
import arff
import pandas as pd

dataset_path = Path("DATASETS")

dataset_dict = {}

for file in dataset_path.iterdir():
    dataset_name = file.name
    feature_file = str(file.absolute()) + "/feature_values.arff"
    with open(feature_file) as fp:
        try:
            arff_dict = arff.load(fp)
        except arff.BadNominalValue:
            print.error(
                'Parsing of arff file failed (%s) - maybe conflict of header and data.', dataset_name)
        
        freq_dict = {}
        for val in arff_dict["attributes"]:
            if val[1] in freq_dict:
                freq_dict[val[1]] += 1
            else:
                freq_dict[val[1]] = 1
    
    dataset_dict[dataset_name] = freq_dict
    
df = pd.DataFrame.from_dict(dataset_dict, orient='index').to_csv("feature_check.csv")
print(df)

