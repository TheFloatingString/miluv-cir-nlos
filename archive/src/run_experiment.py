import argparse
import yaml
import pandas as pd

from lazypredict.Supervised import LazyClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC

from pprint import pprint
import pickle

import numpy as np
import matplotlib.pyplot as plt
import datetime
import math
import os
import tqdm


# from tabpfn import TabPFNClassifier


class CIRPoint:
    def __init__(self):
        self.x: float = None
        self.y: float = None
        self.z: float = None
        self.acc: float = None
        self.vel: float = None
        self.is_nlos: bool = None
        self.from_id: int = None
        self.to_id: int = None


ONLY_SVC = False
VERBOSE = False

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


def is_within_valid_angle(from_id: int, to_id: int, drone_pos) -> bool:
    """
    check whether the drone is within a valid projeced angle
    """
    for i in range(6):
        pass  # check
    pass


def is_nlos(y):
    if y in [1, 3, 4]:
        return 1
    else:
        return 0


def filter_by_drone_receiver(df: pd.DataFrame, receiver_id: int) -> pd.DataFrame:
    return df[df.from_id == receiver_id]


def run_classifier(X_train, X_test, y_train, y_test, exp_config, curr_exp_name):
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

    if exp_config[curr_exp_name]["classifier"] == "SVC":
        clf = SVC()
        clf.fit(X_train, y_train)
        print(clf.score(X_test, y_test))
        with open("clf.pkl", "wb") as f:
            pickle.dump(clf, f)

    if exp_config[curr_exp_name]["classifier"] == "tabpfn":
        print("ack")
        clf = TabPFNClassifier()
        print("ack")
        X_train = np.zeros((10, 10))
        y_train = np.zeros(10)
        X_test = np.zeros((10, 10))
        y_test = np.zeros((10, 10))
        clf.fit(X_train, y_train)
        print("ack")
        predictions = clf.predict(X_test)
        print("Accuracy", accuracy_score(y_test, predictions))
        print("hi!")
        # print(models)

        # models.to_csv(
        #     f"{exp_config[curr_exp_name]['name']}-{str(datetime.datetime.now())}.csv"
        # )

        # with open(
        #     f"{exp_config[curr_exp_name]['name']}-{str(datetime.datetime.now())}.txt",
        #     "w",
        # ) as outputfile:
        #     outputfile.write(str(models))
        #     outputfile.close()


def get_ground_truth_dist_between_tags(df_uwb_cir, df_dist) -> list:
    dist_from_drone_to_uwb = []
    for idx, row in df_uwb_cir.iterrows():
        target = row["timestamp"]
        differences = np.abs(df_dist["timestamp"] - target)
        nearest_index = differences.idxmin()

        drone_x = df_dist.loc[nearest_index]["pose.position.x"]
        drone_y = df_dist.loc[nearest_index]["pose.position.y"]
        drone_z = df_dist.loc[nearest_index]["pose.position.z"]

        anchor_id = int(row["to_id"])

        drone_pos = np.asarray([drone_x, drone_y, drone_z])
        uwb_pos = np.asarray(uwb_constellation_pos[0][anchor_id])

        abs_dist = np.linalg.norm(drone_pos - uwb_pos)
        dist_from_drone_to_uwb.append(abs_dist)

        if VERBOSE:
            print(target)
            print(nearest_index)
            print(df_dist.loc[nearest_index])
            print(df_dist.loc[nearest_index]["pose.position.x"])
            print(df_dist.loc[nearest_index]["pose.position.y"])
            print(df_dist.loc[nearest_index]["pose.position.z"])

    return dist_from_drone_to_uwb


def get_ranging_dist_between_tags(df_uwb_cir, df_ranging, tag_blind: bool = True):
    ranging_dist_from_drone_to_uwb = []
    for idx, row in df_uwb_cir.iterrows():
        df_mod = df_ranging.copy()
        target = row["timestamp"]
        from_id = row["from_id"]
        to_id = row["to_id"]
        if not tag_blind:
            df_mod = df_mod[df_mod.from_id == from_id]
            df_mod = df_mod[df_mod.to_id == to_id]

        differences = np.abs(df_mod["timestamp"] - target)
        nearest_index = differences.idxmin()
        ranging_dist_idx = df_mod.loc[nearest_index]["range"]
        # print(idx, df_ranging.shape)
        ranging_dist_from_drone_to_uwb.append(ranging_dist_idx)
    return ranging_dist_from_drone_to_uwb


def in_bool(main_list, key_list):
    return_list = []
    for item in main_list:
        if item in key_list:
            return_list.append(True)
        else:
            return_list.append(False)
    return return_list


def process_miluv_dataset(exp_config, curr_exp_name):
    list_of_dfs = []
    for miluv_exp_name in exp_config[curr_exp_name]["datasets"]:
        print("--------------------------------")
        print(f"data/{miluv_exp_name}")
        for dirname in os.listdir(f"data/{miluv_exp_name}"):
            print(dirname)
            if "ifo" in dirname:  # TODO: not robust enough
                df_tmp = pd.read_csv(f"data/{miluv_exp_name}/{dirname}/uwb_cir.csv")
                df_tmp = df_tmp[
                    in_bool(df_tmp.to_id, [0, 1, 2, 3, 4, 5])
                ]  # TODO: filtering the .to_id
                dist_from_drone_to_uwb = []
                if (
                    "distance_scaling"
                    in exp_config[curr_exp_name]["orderedPreprocessing"]
                ):
                    # TODO WIP
                    df_dist = pd.read_csv(f"data/{miluv_exp_name}/{dirname}/mocap.csv")
                    df_tmp["dist_drone_to_uwb"] = get_ground_truth_dist_between_tags(
                        df_uwb_cir=df_tmp, df_dist=df_dist
                    )
                    print(df_tmp["dist_drone_to_uwb"].values)
                    list_of_dfs.append(df_tmp)
                if (
                    "ranging_scaling"
                    in exp_config[curr_exp_name]["orderedPreprocessing"]
                ):
                    # TODO WIP
                    df_ranging = pd.read_csv(
                        f"data/{miluv_exp_name}/{dirname}/uwb_range.csv"
                    )
                    df_tmp["dist_drone_to_uwb"] = get_ranging_dist_between_tags(
                        df_uwb_cir=df_tmp, df_ranging=df_ranging
                    )
                    print(df_tmp["dist_drone_to_uwb"].values)
                    list_of_dfs.append(df_tmp)
                else:
                    list_of_dfs.append(df_tmp)
    return list_of_dfs


def process_ewine_dataset(exp_config, curr_exp_name):
    list_of_dfs = []
    for ewine_exp_name in exp_config[curr_exp_name]["datasets"]:
        df = pd.read_csv(f"data/ewine/{ewine_exp_name}")

        print(df.head())
        print(df.iloc[:, 15:1031])

        list_of_cir = []
        for idx in tqdm.trange(df.shape[0]):
            tmp = df.iloc[idx, 15:1031].values
            list_of_cir.append(str(tmp.tolist()))

        df = df.iloc[:, 0:2]
        df["cir"] = list_of_cir
        list_of_dfs.append(df)

    return list_of_dfs


def process_ewine_dataset_with_localization(exp_config, curr_exp_name):
    list_of_dfs = []
    for ewine_exp_name in exp_config[curr_exp_name]["datasets"]:
        df = pd.read_csv(f"data/ewine/{ewine_exp_name}")

        print(df.head())
        print(df.iloc[:, 15:1031])

        list_of_cir = []
        list_of_dist = []
        for idx in tqdm.trange(df.shape[0]):
            tmp = df.iloc[idx, 1039 - 1016 : 1039].values
            list_of_cir.append(str(tmp.tolist()))
            dx = df.iloc[idx, 0] - df.iloc[idx, 2]
            dy = df.iloc[idx, 1] - df.iloc[idx, 3]
            dist = math.sqrt(dx**2 + dy**2)
            list_of_dist.append(dist)

        df = df.iloc[:, 0:10]
        df["cir"] = list_of_cir
        df["NLOS"] = df.iloc[:, 5].values
        df["GT_DISTANCE"] = list_of_dist
        # df = df.dropna()

        list_of_dfs.append(df)

    return list_of_dfs


def preprocess_data_for_exp(exp_config, curr_exp_name):
    list_of_dfs = []
    if exp_config[curr_exp_name]["data_source"] == "ewine":
        list_of_dfs = process_ewine_dataset(exp_config, curr_exp_name)
    elif exp_config[curr_exp_name]["data_source"] == "ewine-with-localization":
        list_of_dfs = process_ewine_dataset_with_localization(exp_config, curr_exp_name)
    else:  # default to miluv
        list_of_dfs = process_miluv_dataset(exp_config, curr_exp_name)

    df = pd.concat(list_of_dfs)
    print(df.shape)
    print(df.head())

    if exp_config[curr_exp_name]["orderedPreprocessing"] is not None:
        for preprocessing_method in exp_config[curr_exp_name]["orderedPreprocessing"]:
            if exp_config[curr_exp_name]["data_source"] in [
                "ewine",
                "ewine-with-localization",
            ]:
                pass  # TODO: skipping for now
            else:
                if preprocessing_method == "filter_receiver_only_10":
                    df = filter_by_drone_receiver(df, receiver_id=10)
                if preprocessing_method == "filter_receiver_only_11":
                    df = filter_by_drone_receiver(df, receiver_id=11)

    X_data = np.asarray([eval(x) for x in df.cir])

    if exp_config[curr_exp_name].get("task") is None:
        raise ValueError("you must specify a task in the .yaml config file")
    elif exp_config[curr_exp_name]["task"] == "NLOS_binary":
        if exp_config[curr_exp_name]["data_source"] in [
            "ewine",
            "ewine-with-localization",
        ]:
            y_data = df.NLOS.values
        else:
            y_data = np.asarray([is_nlos(y) for y in df.to_id])
    else:
        raise ValueError("task from the .yaml config file not recognized")

    if exp_config[curr_exp_name]["orderedPreprocessing"] is not None:
        for preprocessing_method in exp_config[curr_exp_name]["orderedPreprocessing"]:
            if (
                "distance_scaling" in exp_config[curr_exp_name]["orderedPreprocessing"]
                or "ranging_scaling"
                in exp_config[curr_exp_name]["orderedPreprocessing"]
            ):
                if exp_config[curr_exp_name]["data_source"] == "ewine":
                    for idx in range(len(X_data)):
                        if (
                            "add_noise_to_ground_truth_distance"
                            in exp_config[curr_exp_name]["orderedPreprocessing"]
                        ):
                            distance = float(df.RANGE[idx]) + np.random.normal(0, 0.5)
                        else:
                            distance = float(df.RANGE[idx])
                        X_data[idx] = (distance**2) * X_data[idx]
                if (
                    exp_config[curr_exp_name]["data_source"]
                    == "ewine-with-localization"
                ):
                    for idx in range(len(X_data)):
                        if (
                            "add_noise_to_ground_truth_distance"
                            in exp_config[curr_exp_name]["orderedPreprocessing"]
                        ):
                            distance = float(df.GT_DISTANCE[idx]) + np.random.normal(
                                0, 0.5
                            )
                        else:
                            distance = float(df.GT_DISTANCE[idx])
                        X_data[idx] = (distance**2) * X_data[idx]
                else:
                    print(df.dist_drone_to_uwb.values)
                    assert X_data.shape[0] == df["dist_drone_to_uwb"].values.shape[0]
                    for idx in range(len(X_data)):
                        if (
                            "add_noise_to_ground_truth_distance"
                            in exp_config[curr_exp_name]["orderedPreprocessing"]
                        ):
                            distance = float(
                                df["dist_drone_to_uwb"].values[idx]
                            ) + np.random.normal(0, 0.5)
                        else:
                            distance = float(df["dist_drone_to_uwb"].values[idx])
                        X_data[idx] = (distance**2) * X_data[idx]
                        if VERBOSE:
                            print(X_data[idx])
                            print()

            if preprocessing_method == "sklearn_normalize":
                X_data = normalize(X_data)

            if preprocessing_method == "fft":
                X_data = np.real(np.fft.fft(X_data))

            if preprocessing_method == "MinMaxScaler":
                scaler = MinMaxScaler()
                X_data = scaler.fit_transform(X_data)
    return X_data, y_data


def run_curr_experiment(exp_config, curr_exp_name):
    X_data, y_data = preprocess_data_for_exp(
        exp_config=exp_config, curr_exp_name=curr_exp_name
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X_data, y_data, test_size=0.2, random_state=0
    )

    print(exp_config[curr_exp_name]["classifier"])
    run_classifier(X_train, X_test, y_train, y_test, exp_config, curr_exp_name)


def run_all_experiments(yaml_config_filepath: str):
    with open(yaml_config_filepath) as configfile:
        exp_config = yaml.load(configfile.read(), Loader=yaml.Loader)
    pprint(exp_config)

    for curr_exp_name in exp_config.keys():
        print(f"exp_name: {curr_exp_name}")
        run_curr_experiment(exp_config, curr_exp_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("yaml_config_filepath")
    args = parser.parse_args()
    print(args.yaml_config_filepath)
    run_all_experiments(yaml_config_filepath=args.yaml_config_filepath)
