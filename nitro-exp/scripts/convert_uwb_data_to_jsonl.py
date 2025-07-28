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


def is_nlos_for_static_case(to_id):
    if to_id in [1, 3, 4]:
        return True
    else:
        return False


def create_uwb_row(df_cir: pd.DataFrame, df_mocap: pd.DataFrame, df_cir_idx: int):
    timestamp = df_cir.iloc[df_cir_idx]["timestamp"]
    tag_id = df_cir.iloc[df_cir_idx]["to_id"]
    return_row = {
        "timestamp": timestamp,
        "my_id": int(df_cir.iloc[df_cir_idx]["my_id"]),
        "from_id": int(df_cir.iloc[df_cir_idx]["from_id"]),
        "to_id": int(df_cir.iloc[df_cir_idx]["to_id"]),
        "mocap_distance": float(get_mocap_distance(df_mocap, timestamp, tag_id)),
        "is_nlos": is_nlos_for_static_case(df_cir.iloc[df_cir_idx]["to_id"]),
        "cir": eval(df_cir.iloc[df_cir_idx]["cir"]),
    }
    return return_row


def convert_uwb_data_to_jsonl(
    uwb_data_path: str, mocap_data_path: str, output_path: str
):
    df_cir = pd.read_csv(uwb_data_path)
    df_mocap = pd.read_csv(mocap_data_path)

    list_of_final_rows = []
    for index, row in tqdm.tqdm(df_cir.iterrows()):
        parsed_row = create_uwb_row(df_cir, df_mocap, index)
        list_of_final_rows.append(parsed_row)

    with open(output_path, "w") as f:
        for row in list_of_final_rows:
            f.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    convert_uwb_data_to_jsonl(
        "data/source_data/cirObstaclesOneTag_1_static_0/ifo001/uwb_cir.csv",
        "data/source_data/cirObstaclesOneTag_1_static_0/ifo001/mocap.csv",
        "uwb_data.jsonl",
    )
