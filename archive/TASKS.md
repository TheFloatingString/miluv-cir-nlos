# Tasks

-> .md file that syncs with GitHub issues

+ [X] include the third dataset
+ [X] run on everything
+ [X] package all random vs all static
+ [ ] add 15-degree angle
+ [X] use UWB ranging instead of ground truth metric
+ [ ] add correction code for static case only having 10 and no 11 for `from_id`
+ [ ] check that all datasets have what we want
+ [ ] run TabPFM on all
+ [ ] run with 100-feature and 500-feature clipping
+ [ ] group in triplets ?
+ [X] write script that can extract the X_data and y_data given the config files
+ [ ] build package with uv
+ [ ] create new point entity that contains timestamp, acceleration, position,
  nlos - this would allow for (1) angle checking and (2) visualizing point
  errors
+ [ ] implement existing neural net approach

## Hotfix

+ [ ] Check that ranging must come from specific dataset for multi-dataset tasks

## Analysis

+ [ ] Determine which points fail on the trajectory in 3D
+ [ ] Compare with "passive-only" ranging and "passive + ranging"
