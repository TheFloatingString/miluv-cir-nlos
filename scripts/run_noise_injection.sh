#!/bin/bash

directory="./config/experiments/noise_injection"

for filepath in "$directory"/*; do
    echo "Running experiment with configuration file $filepath"
    uv run -m src.run_experiment $filepath
done

