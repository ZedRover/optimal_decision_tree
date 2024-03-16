#!/bin/bash

# Define the datasets and depths
datasets=("seeds" "glass" "body" "cncret" "car_eval_enc.csv" "Banknote-authentication" "Contraceptive-method-choice" "Ozone-level-detection-eight" "Ozone-level-detection-one" "Spambase" "Statlog-project-German-credit-24d" "Statlog-project-landsat-satellite" "Thyroid-disease-ann-thyroid" "Wall-following-robot-2" "page-blocks.csv")
depths=(2 3)

# Iterate over each dataset
for dataset in "${datasets[@]}"; do
    # Iterate over each depth
    for depth in "${depths[@]}"; do
        # Run the Julia command in the background for each combination of dataset and depth
        julia test/test.jl $depth "CF+MILP+SG" 1 "par" $dataset &
    done
done

# Wait for all background processes to finish
wait
