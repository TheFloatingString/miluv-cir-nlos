# run cross eval on train datasets
uv run src/run_experiment.py config/experiments/cross_eval_train/random_only-distance_scaling-svc.yaml
mv clf.pkl random_only-distance_scaling-svc.pkl
uv run src/run_experiment.py config/experiments/cross_eval_train/cirObstaclesOneTag_1_static_0-distance_scaling-svc.yaml
mv clf.pkl cirObstaclesOneTag_1_static_0-distance_scaling-svc.pkl
# move .pkl classifiers to their respective folder
mv *pkl static/May_2025
# run cross eval on test datasets
uv run src/run_cross_eval.py config/experiments/cross_eval_test/eval_random_only_classifier_on_static.yaml
uv run src/run_cross_eval.py config/experiments/cross_eval_test/eval_static_classifier_on_random_only.yaml