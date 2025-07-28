from rich import print as rprint
import numpy as np



class GetCirOnly():
    def __init__(self):
        pass

    def get_cir_only(self,row: dict) -> list:
        return np.asarray(row["cir"])


    def fit(self, X: list) -> None:
        # print("Fitting")
        # print(X)
        pass

    def fit_transform(self, X: list) -> list:
        return self.transform(X)

    def transform(self, X: list) -> list:
        # rprint(X)
        return [self.get_cir_only(row) for row in X]

class MocapDistanceScaling():
    def __init__(self):
        pass

    def get_mocap_distance(self,row: dict) -> float:
        return np.asarray(row["cir"])*(row["mocap_distance"]**2)

    def fit(self, X: list) -> None:
        pass

    def transform(self, X: list) -> list:
        return [self.get_mocap_distance(row) for row in X]

    def fit_transform(self, X: list) -> list:
        return self.transform(X)


class RangingDistanceScaling():
    def __init__(self):
        pass

    def get_ranging_distance(self,row: dict) -> float:
        return np.asarray(row["cir"])*(row["ranging_distance"]**2)

    def fit(self, X: list) -> None:
        pass

    def transform(self, X: list) -> list:
        return [self.get_ranging_distance(row) for row in X]

    def fit_transform(self, X: list) -> list:
        return self.transform(X)
