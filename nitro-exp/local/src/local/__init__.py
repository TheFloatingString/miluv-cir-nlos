from rich import print as rprint
import numpy as np

def get_cir_only(row: dict) -> list:
    return np.asarray(row["cir"])*(row["mocap_distance"]**2)


class GetCirOnly():
    def __init__(self):
        pass

    def fit(self, X: list) -> None:
        # print("Fitting")
        # print(X)
        pass

    def fit_transform(self, X: list) -> list:
        return self.transform(X)

    def transform(self, X: list) -> list:
        # rprint(X)
        return [get_cir_only(row) for row in X]
