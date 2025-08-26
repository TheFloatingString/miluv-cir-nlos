import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df_cir = pd.read_csv(
    "/Users/lliang/projects/miluv-cir-nlos/nitro-exp/data/source_data/cirObstaclesOneTag_1_static_0/ifo001/uwb_cir.csv"
)

los_cir = df_cir[df_cir["to_id"].isin([0, 2, 5])]
nlos_cir = df_cir[df_cir["to_id"].isin([1, 3, 4])]

mean_los_cir = np.mean([eval(cir) for cir in los_cir["cir"]], axis=0)
mean_nlos_cir = np.mean([eval(cir) for cir in nlos_cir["cir"]], axis=0)

for i in range(6):
    plt.subplot(3, 2, i + 1)
    cir_arr = df_cir[df_cir["to_id"] == i]["cir"]
    mean_cir = np.mean([eval(cir) for cir in cir_arr], axis=0)
    print(mean_cir.shape)
    plt.ylim(0, 5000)
    plt.title(f"to_id={i}")
    plt.plot(mean_cir)
    # plt.subplot(3,2,i+1)
    # plt.title(f"Mean NLOS CIR {i}")
    # plt.plot(mean_nlos_cir[i])
plt.tight_layout()
plt.show()

# plt.subplot(2,1,1)
# plt.title("Mean LOS CIR")
# plt.plot(mean_los_cir)
# plt.subplot(2,1,2)
# plt.title("Mean NLOS CIR")
# plt.plot(mean_nlos_cir)
# plt.tight_layout()
# plt.show()
