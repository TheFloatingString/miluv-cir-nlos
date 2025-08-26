import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

X_data = []
y_data = []

df_uwb_range = pd.read_csv(
    # "/Users/lliang/projects/miluv-cir-nlos/nitro-exp/data/source_data/cirObstaclesOneTag_1_static_0/ifo001/uwb_range.csv"
    "/Users/lliang/projects/miluv-cir-nlos/nitro-exp/data/source_data/cirObstacles_1_random3_0/ifo001/uwb_range.csv"
)
print(df_uwb_range.head())

for idx, row in df_uwb_range.iterrows():
    X_data.append(
        row.loc[
            [
                # "to_id",
                # "from_id",
                # "range",
                "tx1",
                "rx1",
                "tx2",
                "rx2",
                "tx3",
                "rx3",
                # "fpp1",
                # "fpp2",
                "skew1",
                "skew2",
                # "gt_range",
                # "bias",
                # "del_t1",
                # "del_t2",
                # "del_t3",
                # "del_t4",
                # "range_raw",
                # "bias_raw",
                # "tx1_raw",
                # "tx2_raw",
                # "tx3_raw",
                # "rx1_raw",
                # "rx2_raw",
                # "rx3_raw",
                # "std",
            ]
        ].values
    )
    y_data.append(row["to_id"])
    # if row["to_id"] in [1, 3, 4]:
    #     y_data.append(1)
    # elif row["to_id"] in [0, 2, 5]:
    #     y_data.append(0)

X_data = np.array(X_data)
y_data = np.array(y_data)

# X_train, X_test, y_train, y_test = train_test_split(
#     X_data, y_data, test_size=0.2, random_state=42
# )

X_train = []
y_train = []
X_test = []
y_test = []

print(X_data[0])

for idx in range(len(X_data)):
    if y_data[idx] in [1, 3, 4]:
        X_train.append(X_data[idx])
        if y_data[idx] in [1, 3, 4]:
            y_train.append(1)
        else:
            y_train.append(0)
    else:
        X_test.append(X_data[idx])
        if y_data[idx] in [1, 3, 4]:
            y_test.append(1)
        else:
            y_test.append(0)


from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

print(clf.score(X_test, y_test))
