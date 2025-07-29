import pandas as pd
import json
import numpy as np
from rich import print as rprint
import tqdm

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


def get_mocap_distance(df_mocap: pd.DataFrame, timestamp: float, tag_id):
    global uwb_constellation_pos
    # given the timestamp, find the row in df_mocap that is the closest
    mocap_row = df_mocap.loc[abs(df_mocap["timestamp"] - timestamp).idxmin()]
    x_diff = mocap_row["pose.position.x"] - uwb_constellation_pos[0][tag_id][0]
    y_diff = mocap_row["pose.position.y"] - uwb_constellation_pos[0][tag_id][1]
    z_diff = mocap_row["pose.position.z"] - uwb_constellation_pos[0][tag_id][2]

    distance = np.linalg.norm(np.array([x_diff, y_diff, z_diff]))

    return distance


def get_ranging_distance(
    df_ranging: pd.DataFrame, timestamp: float, from_id: int, to_id: int
):
    # TODO: why only 10 and not 11 for from_id ?
    df_ranging_filtered = df_ranging[df_ranging["to_id"] == to_id]
    ranging_row = df_ranging_filtered.loc[
        abs(df_ranging_filtered["timestamp"] - timestamp).idxmin()
    ]
    distance = ranging_row["range"]
    return distance


def get_fpp1(df_ranging: pd.DataFrame, timestamp: float, from_id: int, to_id: int):
    df_ranging_filtered = df_ranging[df_ranging["to_id"] == to_id]
    ranging_row = df_ranging_filtered.loc[
        abs(df_ranging_filtered["timestamp"] - timestamp).idxmin()
    ]
    fpp1 = ranging_row["fpp1"]
    return fpp1


def is_nlos_for_static_case(to_id):
    if to_id in [1, 3, 4]:
        return True
    else:
        return False


def create_uwb_row(
    df_cir: pd.DataFrame,
    df_mocap: pd.DataFrame,
    df_ranging: pd.DataFrame,
    df_cir_idx: int,
):
    # print(df_cir.head())
    # print(df_cir.columns)
    timestamp = df_cir.iloc[df_cir_idx]["timestamp"]
    tag_id = df_cir.iloc[df_cir_idx]["to_id"]

    from_id = df_cir.iloc[df_cir_idx]["from_id"]
    to_id = df_cir.iloc[df_cir_idx]["to_id"]

    return_row = {
        "timestamp": timestamp,
        "my_id": int(df_cir.iloc[df_cir_idx]["my_id"]),
        "from_id": int(df_cir.iloc[df_cir_idx]["from_id"]),
        "to_id": int(df_cir.iloc[df_cir_idx]["to_id"]),
        "mocap_distance": float(get_mocap_distance(df_mocap, timestamp, tag_id)),
        "ranging_distance": float(
            get_ranging_distance(df_ranging, timestamp, from_id, to_id)
        ),
        "is_nlos": is_nlos_for_static_case(df_cir.iloc[df_cir_idx]["to_id"]),
        "cir": eval(df_cir.iloc[df_cir_idx]["cir"]),
        "fpp1": get_fpp1(df_ranging, timestamp, from_id, to_id),
    }
    return return_row


def convert_uwb_data_to_jsonl(
    uwb_data_path: str, mocap_data_path: str, ranging_data_path: str
):
    df_cir = pd.read_csv(uwb_data_path)
    df_mocap = pd.read_csv(mocap_data_path)
    df_ranging = pd.read_csv(ranging_data_path)

    list_of_final_rows = []
    for index, row in tqdm.tqdm(df_cir.iterrows()):
        parsed_row = create_uwb_row(df_cir, df_mocap, df_ranging, index)
        list_of_final_rows.append(parsed_row)

    return list_of_final_rows


if __name__ == "__main__":
    # static case
    list_of_formatted_rows = convert_uwb_data_to_jsonl(
        "data/source_data/cirObstaclesOneTag_1_static_0/ifo001/uwb_cir.csv",
        "data/source_data/cirObstaclesOneTag_1_static_0/ifo001/mocap.csv",
        "data/source_data/cirObstaclesOneTag_1_static_0/ifo001/uwb_range.csv",
    )

    for target_id in [0, 1, 2, 3, 4, 5]:
        with open(
            f"data/processed_data/X_data_static_1_target_{target_id}.jsonl", "w"
        ) as f:
            for row in tqdm.tqdm(list_of_formatted_rows):
                if row["to_id"] == target_id:
                    f.write(json.dumps(row) + "\n")

    bool_to_int = {True: 1, False: 0}

    for target_id in [0, 1, 2, 3, 4, 5]:
        with open(
            f"data/processed_data/y_data_static_1_target_{target_id}.jsonl", "w"
        ) as f:
            for row in tqdm.tqdm(list_of_formatted_rows):
                if row["to_id"] == target_id:
                    f.write(json.dumps(bool_to_int[row["is_nlos"]]) + "\n")

    # 1 drone random case
    list_of_formatted_rows = convert_uwb_data_to_jsonl(
        "data/source_data/cirObstacles_1_random3_0/ifo001/uwb_cir.csv",
        "data/source_data/cirObstacles_1_random3_0/ifo001/mocap.csv",
        "data/source_data/cirObstacles_1_random3_0/ifo001/uwb_range.csv",
    )

    for target_id in [0, 1, 2, 3, 4, 5]:
        with open(
            f"data/processed_data/X_data_random_1_target_{target_id}.jsonl", "w"
        ) as f:
            for row in tqdm.tqdm(list_of_formatted_rows):
                if row["to_id"] == target_id:
                    f.write(json.dumps(row) + "\n")

    bool_to_int = {True: 1, False: 0}

    for target_id in [0, 1, 2, 3, 4, 5]:
        with open(
            f"data/processed_data/y_data_random_1_target_{target_id}.jsonl", "w"
        ) as f:
            for row in tqdm.tqdm(list_of_formatted_rows):
                if row["to_id"] == target_id:
                    f.write(json.dumps(bool_to_int[row["is_nlos"]]) + "\n")

    # ewine dataset
    df = pd.read_csv("data/source_data/ewine/uwb_dataset_part1.csv")
    list_of_final_rows = []
    for index, row in tqdm.tqdm(df.iterrows()):
        curr_row = {
            "timestamp": None,
            "is_nlos": row["NLOS"],
            "mocap_distance": row["RANGE"],
            "ranging_distance": row["RANGE"],
            "cir": row[15:].values.tolist(),
        }
        list_of_final_rows.append(curr_row)

    with open("data/processed_data/X_data_ewine.jsonl", "w") as f:
        for row in tqdm.tqdm(list_of_final_rows):
            f.write(json.dumps(row) + "\n")

    with open("data/processed_data/y_data_ewine.jsonl", "w") as f:
        for row in tqdm.tqdm(list_of_final_rows):
            f.write(json.dumps(row["is_nlos"]) + "\n")

    # ewine for localization dataset
    df = pd.read_csv("data/source_data/ewine/tag_room0.csv")
    list_of_final_rows = []
    for index, row in tqdm.tqdm(df.iterrows()):
        gt_distance = np.linalg.norm(
            np.array([row[0], row[1]]) - np.array([row[2], row[3]])
        )
        ranging_distance = row[4]

        cir = row[24 : 24 + 1016].values.tolist()
        cir[-1] = 0.0

        curr_row = {
            "timestamp": None,
            "is_nlos": row[5],
            "mocap_distance": gt_distance,
            "ranging_distance": ranging_distance,
            "cir": cir,
        }
        list_of_final_rows.append(curr_row)

    with open("data/processed_data/X_data_ewine_localization.jsonl", "w") as f:
        for row in tqdm.tqdm(list_of_final_rows):
            f.write(json.dumps(row) + "\n")

    with open("data/processed_data/y_data_ewine_localization.jsonl", "w") as f:
        for row in tqdm.tqdm(list_of_final_rows):
            f.write(json.dumps(row["is_nlos"]) + "\n")
