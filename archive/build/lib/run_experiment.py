import argparse
import yaml
import pandas as pd

from lazypredict.Supervised import LazyClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from sklearn.preprocessing import MinMaxScaler

from pprint import pprint

import numpy as np
import matplotlib.pyplot as plt
import datetime
import os

uwb_constellation_pos = {
    0: {
        0: [3.273827392578125, 3.46404736328125, 1.8093309326171875],
        1: [3.186386962890625, 0.27394485473632812, 1.5884853515625],
        2: [2.850500244140625, -2.923056884765625, 1.89742041015625],
        3: [-2.497634521484375, -3.5018203125, 1.7730911865234375],
        4: [-2.95793310546875, 0.6128419189453125, 1.65714208984375],
        5: [-2.734676513671875, 3.65854248046875, 1.890254638671875],
    }
}


def is_nlos(y):
    if y in [1, 3, 4]:
        return 1
    else:
        return 0


def filter_by_drone_receiver(df: pd.DataFrame, receiver_id: int) -> pd.DataFrame:
    return df[df.from_id == receiver_id]


def run_experiment(yaml_config_filepath: str):
    with open(yaml_config_filepath) as configfile:
        exp_config = yaml.load(configfile.read(), Loader=yaml.Loader)
    pprint(exp_config)

    for curr_exp_name in exp_config.keys():
        list_of_dfs = []
        for miluv_exp_name in exp_config[curr_exp_name]["datasets"]:
            print(miluv_exp_name)
            for dirname in os.listdir(f"data/{miluv_exp_name}"):
                if "ifo" in dirname:  # TODO: not robust enough
                    df_tmp = pd.read_csv(f"data/{miluv_exp_name}/{dirname}/uwb_cir.csv")
                    dist_from_drone_to_uwb = []
                    if (
                        "distance_scaling"
                        in exp_config[curr_exp_name]["orderedPreprocessing"]
                    ):
                        # TODO WIP
                        df_dist = pd.read_csv(
                            f"data/{miluv_exp_name}/{dirname}/mocap.csv"
                        )

                        for idx, row in df_tmp.iterrows():
                            target = row["timestamp"]
                            # print(target)
                            differences = np.abs(df_dist["timestamp"] - target)
                            nearest_index = differences.idxmin()
                            # print(nearest_index)
                            # print(df_dist.loc[nearest_index])
                            # print(df_dist.loc[nearest_index]["pose.position.x"])
                            # print(df_dist.loc[nearest_index]["pose.position.y"])
                            # print(df_dist.loc[nearest_index]["pose.position.z"])

                            drone_x = df_dist.loc[nearest_index]["pose.position.x"]
                            drone_y = df_dist.loc[nearest_index]["pose.position.y"]
                            drone_z = df_dist.loc[nearest_index]["pose.position.z"]

                            anchor_id = int(row["to_id"])

                            drone_pos = np.asarray([drone_x, drone_y, drone_z])
                            uwb_pos = np.asarray(uwb_constellation_pos[0][anchor_id])

                            abs_dist = np.linalg.norm(drone_pos - uwb_pos)
                            dist_from_drone_to_uwb.append(abs_dist)
                            # print(abs_dist)

                        df_tmp["dist_drone_to_uwb"] = dist_from_drone_to_uwb
                        print(df_tmp["dist_drone_to_uwb"].values)
                    list_of_dfs.append(df_tmp)

        df = pd.concat(list_of_dfs)
        print(df.shape)
        print(df.head())

        if exp_config[curr_exp_name]["orderedPreprocessing"] is not None:
            for preprocessing_method in exp_config[curr_exp_name][
                "orderedPreprocessing"
            ]:
                if preprocessing_method == "filter_receiver_only_10":
                    df = filter_by_drone_receiver(df, receiver_id=10)
                if preprocessing_method == "filter_receiver_only_11":
                    df = filter_by_drone_receiver(df, receiver_id=11)

        X_data = np.asarray([eval(x) for x in df.cir])

        if exp_config[curr_exp_name].get("task") is None:
            raise ValueError("you must specify a task in the .yaml config file")
        elif exp_config[curr_exp_name]["task"] == "NLOS_binary":
            y_data = np.asarray([is_nlos(y) for y in df.to_id])
        else:
            raise ValueError("task from the .yaml config file not recognized")

        if exp_config[curr_exp_name]["orderedPreprocessing"] is not None:
            for preprocessing_method in exp_config[curr_exp_name][
                "orderedPreprocessing"
            ]:
                if (
                    "distance_scaling"
                    in exp_config[curr_exp_name]["orderedPreprocessing"]
                ):
                    print(df.dist_drone_to_uwb.values)
                    assert X_data.shape[0] == df["dist_drone_to_uwb"].values.shape[0]
                    for idx in range(len(X_data)):
                        # print(X_data[idx])
                        X_data[idx] = (
                            df["dist_drone_to_uwb"].values[idx] ** 2
                        ) * X_data[idx]
                        # print(X_data[idx])
                        # print()

                if preprocessing_method == "sklearn_normalize":
                    X_data = normalize(X_data)

                if preprocessing_method == "fft":
                    X_data = np.real(np.fft.fft(X_data))

                if preprocessing_method == "MinMaxScaler":
                    scaler = MinMaxScaler()
                    X_data = scaler.fit_transform(X_data)

        X_train, X_test, y_train, y_test = train_test_split(
            X_data, y_data, test_size=0.2, random_state=0
        )

        if exp_config[curr_exp_name]["classifier"] == "LazyClassifier":
            clf = LazyClassifier()
            models, predictions = clf.fit(X_train, X_test, y_train, y_test)
            print(models)

            models.to_csv(
                f"{exp_config[curr_exp_name]['name']}-{str(datetime.datetime.now())}.csv"
            )

            with open(
                f"{exp_config[curr_exp_name]['name']}-{str(datetime.datetime.now())}.txt",
                "w",
            ) as outputfile:
                outputfile.write(str(models))
                outputfile.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("yaml_config_filepath")
    args = parser.parse_args()
    print(args.yaml_config_filepath)
    run_experiment(yaml_config_filepath=args.yaml_config_filepath)
