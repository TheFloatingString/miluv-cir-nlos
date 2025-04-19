import pandas as pd
from lazypredict.Supervised import LazyClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
import numpy as np
import matplotlib.pyplot as plt

import argparse

def main(DATASET_TASK: str, METHODS: str):

    if DATASET_TASK=="1_drone_random_with_obstacles":
        df = pd.read_csv("data/cirObstacles_1_random3_0/ifo001/uwb_cir.csv")
    if DATASET_TASK=="1_drone_random_with_obstacles-from_10_only":
        df = pd.read_csv("data/cirObstacles_1_random3_0/ifo001/uwb_cir.csv")
        df = df[df.from_id==10]
    if DATASET_TASK=="1_drone_random_with_obstacles-from_11_only":
        df = pd.read_csv("data/cirObstacles_1_random3_0/ifo001/uwb_cir.csv")
        df = df[df.from_id==11]
    elif DATASET_TASK=="3_drones_static_with_obstacles":
        df = pd.read_csv("data/cirObstaclesOneTag_1_static_0/ifo001/uwb_cir.csv")
    elif DATASET_TASK=="3_drones_static_with_obstacles-from_10_only":
        df = pd.read_csv("data/cirObstaclesOneTag_1_static_0/ifo001/uwb_cir.csv")
        df = df[df.from_id==10]
    elif DATASET_TASK=="3_drones_static_with_obstacles":
        df = pd.read_csv("data/cirObstaclesOneTag_1_static_0/ifo001/uwb_cir.csv")
    elif DATASET_TASK=="3_drones_random_with_obstacles":
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
    
    def one_hot_id(idx):
        arr = [0.0,0.0,0.0,0.0,0.0,0.0]
        arr[idx] = 1.0
        return arr
    
    X_data = np.asarray([eval(x) for (x, idx) in zip(df.cir, df.to_id)])
    y_data = np.asarray([is_nlos(y) for y in df.to_id])
    
    if METHODS=="fft_domain":
        X_data = np.real(np.fft.fft(X_data))

    if METHODS=="fft_denoise-to_time":
        print(f"Running: {METHODS}")
        n_components = 10000.0
        X_data = np.real(np.fft.fft(X_data))
        # PSD = X_data*np.conj(X_data)
        # MASK = PSD < 30
        # H = MASK * X_data
        X_data = np.fft.ifft(X_data[:,0:100]).real
        
    
        # PSD0 = np.where(PSD<100, 0, PSD)
        # h = np.where(PSD<100, 0, X_data)
        # X_data = np.fft.ifft(h)
        # X_data = np.real(X_data)
        # print(X_data.shape)
    
    X_data = normalize(X_data)
    plt.plot(X_data[0])
    plt.plot(X_data[1])
    plt.plot(X_data[2])
    plt.savefig("tmp.png")
    X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.2, random_state=0)
    
    clf = LazyClassifier()
    models, predictions = clf.fit(X_train, X_test, y_train, y_test)
    print(models)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_task")
    parser.add_argument("methods")
    args = parser.parse_args()
    main(DATASET_TASK=args.dataset_task, METHODS=args.methods)