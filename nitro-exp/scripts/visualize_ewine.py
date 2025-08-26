import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df_ewine = pd.read_csv(
    "/Users/lliang/projects/miluv-cir-nlos/nitro-exp/data/source_data/ewine/uwb_dataset_part1.csv"
)
print(df_ewine.head().iloc[0, 0])
print(df_ewine.head().iloc[0, 15:].values.shape)

nlos = df_ewine[df_ewine["NLOS"] == 1]
nlos = nlos.iloc[:, 15:]
nlos_mean = np.mean(nlos.values, axis=0)
print(nlos.head())

los = df_ewine[df_ewine["NLOS"] == 0]
los = los.iloc[:, 15:]
los_mean = np.mean(los.values, axis=0)
print(los.head())
print(los_mean.shape)

plt.subplot(2, 1, 1)
plt.plot(los_mean)
plt.ylim(0, 10000)
plt.title("LOS")
plt.subplot(2, 1, 2)
print(nlos.shape)
plt.plot(nlos_mean)
plt.plot(nlos.iloc[0, :])
# plt.plot(nlos[0,0:10])
plt.ylim(0, 10000)
plt.title("NLOS")
plt.tight_layout()
plt.show()
