import argparse
import yaml
import pandas as pd

from lazypredict.Supervised import LazyClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
import numpy as np
import matplotlib.pyplot as plt
import datetime
import os


def is_nlos(y):
    if y in [1,3,4]:
        return 1
    else:
        return 0

def filter_by_drone_receiver(df: pd.DataFrame, receiver_id: int) -> pd.DataFrame:
    return df[df.from_id==receiver_id]

def run_experiment(yaml_config_filepath: str):
    with open(yaml_config_filepath) as configfile:
        exp_config = yaml.load(configfile.read(), Loader=yaml.Loader)
    print(exp_config)

    list_of_dfs = []
    for miluv_exp_name in exp_config["experiment"]["datasets"]:
        print(miluv_exp_name)
        for dirname in os.listdir(f"data/{miluv_exp_name}"):
            if "ifo" in dirname: # TODO: not robust enough
                df_tmp = pd.read_csv(f"data/{miluv_exp_name}/{dirname}/uwb_cir.csv")

                if "distance_scaling" in exp_config["experiment"]["orderedPreprocessing"]:
                    # TODO WIP
                    pass
                    # df_dist = pd.read_csv(f"data/{miluv_exp_name}/{dirname}/mocap.csv")
                    # differences = np.abs(df['timestamp'] - target)
                    # nearest_index = differences.idxmin()
                    # df_dist = df_dist[df_dist]

                list_of_dfs.append(df_tmp)

    df = pd.concat(list_of_dfs)
    print(df.shape)
    print(df.head())

    if exp_config["experiment"]["orderedPreprocessing"] is not None:
        for preprocessing_method in exp_config["experiment"]["orderedPreprocessing"]:
            if preprocessing_method=="filter_receiver_only_10":
                df = filter_by_drone_receiver(df, receiver_id=10)
            if preprocessing_method=="filter_receiver_only_11":
                df = filter_by_drone_receiver(df, receiver_id=11)
                
    X_data = np.asarray([eval(x) for x in df.cir])
    
    if exp_config["experiment"].get("task") is None:
        raise ValueError("you must specify a task in the .yaml config file")
    elif exp_config["experiment"]["task"]=="NLOS_binary":
        y_data = np.asarray([is_nlos(y) for y in df.to_id])
    else:
        raise ValueError("task from the .yaml config file not recognized")

    if exp_config["experiment"]["orderedPreprocessing"] is not None:
        for preprocessing_method in exp_config["experiment"]["orderedPreprocessing"]:
            if preprocessing_method=="sklearn_normalize":
                X_data = normalize(X_data)                
            
            if preprocessing_method=="fft":
                X_data = np.real(np.fft.fft(X_data))

    X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.2, random_state=0)

    if exp_config["experiment"]["classifier"] == "LazyClassifier":
        
        clf = LazyClassifier()
        models, predictions = clf.fit(X_train, X_test, y_train, y_test)
        print(models)

        models.to_csv(f"{exp_config['experiment']['name']}-{str(datetime.datetime.now())}.csv")

        with open(f"{exp_config['experiment']['name']}-{str(datetime.datetime.now())}.txt", 'w') as outputfile:
            outputfile.write(str(models))
            outputfile.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("yaml_config_filepath")
    args = parser.parse_args()
    print(args.yaml_config_filepath)
    run_experiment(yaml_config_filepath=args.yaml_config_filepath)