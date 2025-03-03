#!/bin/bash

# Arguments of the different runs
CONFIG_FILES=("config_files/bgl_no_iid.yaml")
NUM_CLIENTS_LIST=(1 5 10)
RUN_NUMBERS=(0 1 2)
METHODS_LIST=("KnowEvents" "LengthDetection" "ECVC" "2-Gram" "3-Gram" "Edit" "KnowEvents+LengthDetection" "KnowEvents+LengthDetection+ECVC" "KnowEvents+LengthDetection+Edit" "2-Gram+LengthDetection")


echo "Start runs"
for CONFIG_FILE in "${CONFIG_FILES[@]}"; do
    for NUM_CLIENTS in "${NUM_CLIENTS_LIST[@]}"; do
        for RUN_NUMBER in "${RUN_NUMBERS[@]}"; do
            for METHODS in "${METHODS_LIST[@]}"; do
                echo "sh ./run_multiple_clients.sh $RUN_NUMBER $CONFIG_FILE $NUM_CLIENTS $METHODS"
                sh ./run_multiple_clients.sh $RUN_NUMBER $CONFIG_FILE $NUM_CLIENTS $METHODS
            done
        done
    done
done

echo "Gather all data"
python script_gather_results.py

echo "End all runs"