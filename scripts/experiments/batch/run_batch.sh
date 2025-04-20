#!/bin/bash

directory="./config/experiments"

for filepath in "$directory"/*; do
    echo "Running experiment with configuration file $filepath"
    python -m src.run_experiment $filepath
done

