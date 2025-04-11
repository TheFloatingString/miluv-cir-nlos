import pandas as pd
from lazypredict.Supervised import LazyClassifier
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.preprocessing import normalize


df1 = pd.read_csv("uwb_cir_files/cirObstaclesOneTag_1_static_0/ifo001/uwb_cir.csv")
df2 = pd.read_csv("uwb_cir_files/cirObstacles_1_random3_0/ifo001/uwb_cir.csv")
df3 = pd.read_csv("uwb_cir_files/cirObstacles_3_random_0/ifo001/uwb_cir.csv")
df4 = pd.read_csv("uwb_cir_files/cirObstacles_3_random_0/ifo002/uwb_cir.csv")
df5 = pd.read_csv("uwb_cir_files/cirObstacles_3_random_0/ifo003/uwb_cir.csv")


df = pd.concat([df3, df4, df5])

def is_nlos(y):
    if y in [2,4,5]:
        return 1
    else:
        return 0

X_data = [eval(x) for x in df.cir.values]
y_data = np.asarray([is_nlos(y) for y in df.to_id])

X_data = normalize(X_data)

X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.2, random_state=42)

clf = LazyClassifier(verbose=0, ignore_warnings=True, custom_metric=None)
models, predictions = clf.fit(X_train, X_test, y_train, y_test)

print(models)
