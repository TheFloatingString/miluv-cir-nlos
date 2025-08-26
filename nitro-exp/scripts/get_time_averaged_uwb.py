import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import json as jsonlib
from sklearn.model_selection import train_test_split


def create_time_averaged_uwb_row(
    df_cir: pd.DataFrame,
    df_mocap: pd.DataFrame,
    df_ranging: pd.DataFrame,
    t_start: float,
    t_end: float,
):
    # TODO: get time-averaged mocap dist

    # TODO: get time-averaged ranging dist

    # TODO: get time-averaged CIR

    # TODO: get time-averaged FPP1
    fpp1_avgs = []
    for idx, row in df_ranging.iterrows():
        if t_start <= row["timestamp"] < t_end:
            pass


def is_nlos_for_static_case(to_id: int) -> bool:
    if to_id in [1, 3, 4]:
        return True
    else:
        return False


def get_averaged_row(
    df_cir: pd.DataFrame,
    df_mocap: pd.DataFrame,
    df_ranging: pd.DataFrame,
    t_start: float,
    t_end: float,
    from_id: int,
    to_id: int,
):
    # TODO: only return rows that are within the time range
    # TODO: make sure these rows are filtered for the correct from_id and to_id
    df_cir_filtered = df_cir[
        (df_cir["timestamp"] >= t_start)
        & (df_cir["timestamp"] < t_end)
        & (df_cir["from_id"] == from_id)
        & (df_cir["to_id"] == to_id)
    ]
    df_ranging_filtered = df_ranging[
        (df_ranging["timestamp"] >= t_start)
        & (df_ranging["timestamp"] < t_end)
        & (df_ranging["from_id"] == from_id)
        & (df_ranging["to_id"] == to_id)
    ]
    df_mocap_filtered = df_mocap[
        (df_mocap["timestamp"] >= t_start) & (df_mocap["timestamp"] < t_end)
    ]

    # mocap_distance_avg = df_mocap_filtered["distance"].mean()
    ranging_distance_avg = df_ranging_filtered["range"].mean()

    # TODO: get fixed value for the following
    is_nlos: bool = is_nlos_for_static_case(to_id)

    list_of_cir_values = []
    for idx, row in df_cir_filtered.iterrows():
        list_of_cir_values.append(eval(row["cir"]))
    cir_avg = np.mean(list_of_cir_values, axis=0)

    # print(cir_avg.shape)

    list_of_fpp1_values = []
    for idx, row in df_ranging_filtered.iterrows():
        list_of_fpp1_values.append(row["fpp1"])
    fpp1_avg = np.mean(list_of_fpp1_values, axis=0)

    list_of_fpp2_values = []
    for idx, row in df_ranging_filtered.iterrows():
        list_of_fpp2_values.append(row["fpp2"])
    fpp2_avg = np.mean(list_of_fpp2_values, axis=0)

    return_row = {
        "timestamp_start": t_start,
        "timestamp_end": t_end,
        "from_id": int(from_id),
        "to_id": int(to_id),
        "is_nlos": is_nlos,
        "cir": cir_avg.tolist(),
        "fpp1": fpp1_avg,
        "fpp2": fpp2_avg,
        "ranging_distance": float(ranging_distance_avg),
    }
    return return_row


if __name__ == "__main__":
    # TODO: Case 1: static

    # TODO: combine CIR data with UWB range data with 2-second averaging
    # pass
    # TODO: iterate through every possible combination of from_id and to_id
    # TODO: get unique values of `from_id`
    # TODO: get unique values of `to_id`
    df_cir = pd.read_csv(
        "data/source_data/cirObstaclesOneTag_1_static_0/ifo001/uwb_cir.csv"
    )
    df_mocap = pd.read_csv(
        "data/source_data/cirObstaclesOneTag_1_static_0/ifo001/mocap.csv"
    )
    df_ranging = pd.read_csv(
        "data/source_data/cirObstaclesOneTag_1_static_0/ifo001/uwb_range.csv"
    )

    unique_from_ids = df_ranging["from_id"].unique()
    unique_to_ids = df_ranging["to_id"].unique()

    X_all_averaged_rows = []
    y_all_targets = []

    N_SECONDS_PER_WINDOW = 25
    number_of_windows = math.ceil(df_cir["timestamp"].max() / N_SECONDS_PER_WINDOW)

    for from_id in unique_from_ids:
        for to_id in unique_to_ids:
            for i in range(0, number_of_windows):
                t_start = i * N_SECONDS_PER_WINDOW
                t_end = t_start + N_SECONDS_PER_WINDOW
                print(from_id, to_id, t_start, t_end)
                X_all_averaged_rows.append(
                    get_averaged_row(
                        df_cir, df_mocap, df_ranging, t_start, t_end, from_id, to_id
                    )
                )
                y_all_targets.append(to_id)

    X_train, X_test, y_train, y_test = train_test_split(
        X_all_averaged_rows, y_all_targets, test_size=0.2, random_state=42
    )

    # write X and y to jsonl
    with open("X_all_averaged_rows_train.jsonl", "w") as f:
        for row in X_train:
            f.write(jsonlib.dumps(row, default=str) + "\n")
    with open("y_all_targets_train.jsonl", "w") as f:
        for target in y_train:
            f.write(str(target) + "\n")

    with open("X_all_averaged_rows_test.jsonl", "w") as f:
        for row in X_test:
            f.write(jsonlib.dumps(row, default=str) + "\n")
    with open("y_all_targets_test.jsonl", "w") as f:
        for target in y_test:
            f.write(str(target) + "\n")

    plt.subplot(2, 1, 1)
    plt.plot(X_all_averaged_rows[0]["cir"])
    plt.subplot(2, 1, 2)
    plt.plot(X_all_averaged_rows[-1]["cir"])
    plt.show()

    # print(X_all_averaged_rows[0])
    # plt.plot(X_all_averaged_rows[0]["cir"])
    # plt.show()

    # TODO: Case 2: random moving
