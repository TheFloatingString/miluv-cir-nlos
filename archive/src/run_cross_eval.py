from run_experiment import preprocess_data_for_exp
import argparse
import pickle
import yaml
from pprint import pprint
from sklearn.model_selection import train_test_split


def run_cross_eval(X_train, X_test, y_train, y_test, exp_config, curr_exp_name):
    CLF_FILEPATH = exp_config[curr_exp_name]["clf_filepath"]

    with open(CLF_FILEPATH, "rb") as pklfile:
        clf = pickle.load(pklfile)

    print(f"score: {clf.score(X_test, y_test)}")


def run_curr_experiment(exp_config, curr_exp_name):
    X_data, y_data = preprocess_data_for_exp(
        exp_config=exp_config, curr_exp_name=curr_exp_name
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X_data, y_data, test_size=0.2, random_state=0
    )

    print(exp_config[curr_exp_name]["classifier"])
    run_cross_eval(X_train, X_test, y_train, y_test, exp_config, curr_exp_name)


def run_all_experiments(yaml_config_filepath: str):
    with open(yaml_config_filepath) as configfile:
        exp_config = yaml.load(configfile.read(), Loader=yaml.Loader)
    pprint(exp_config)

    for curr_exp_name in exp_config.keys():
        print(f"exp_name: {curr_exp_name}")
        run_curr_experiment(exp_config, curr_exp_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("yaml_config_filepath")
    args = parser.parse_args()
    run_all_experiments(yaml_config_filepath=args.yaml_config_filepath)
