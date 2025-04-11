import pandas as pd
from lazypredict.Supervised import LazyClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
import numpy as np

TASK = 2

if TASK==0:
    df = pd.read_csv("data/cirObstacles_1_random3_0/ifo001/uwb_cir.csv")
elif TASK==1:
    df = pd.read_csv("data/cirObstaclesOneTag_1_static_0/ifo001/uwb_cir.csv")
elif TASK==2:
    df1 = pd.read_csv("data/cirObstacles_3_random_0/ifo001/uwb_cir.csv")
    df2 = pd.read_csv("data/cirObstacles_3_random_0/ifo002/uwb_cir.csv")
    df3 = pd.read_csv("data/cirObstacles_3_random_0/ifo003/uwb_cir.csv")
    df = pd.concat([df1, df2, df3])

print(df.head())

def is_nlos(y):
    if y in [1,3,4]:
        return 1
    else:
        return 0

X_data = np.asarray([eval(x) for x in df.cir])
y_data = np.asarray([is_nlos(y) for y in df.to_id])

X_data = normalize(X_data)
X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.2, random_state=0)

clf = LazyClassifier()
models, predictions = clf.fit(X_train, X_test, y_train, y_test)
print(models)
