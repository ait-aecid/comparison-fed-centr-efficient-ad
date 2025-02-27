#!/bin/bash

# Arguments of the different runs
CONFIG_FILES=("config_files/hdfs_iid.yaml" "config_files/bgl_iid.yaml")
NUM_CLIENTS_LIST=(1 5 10)
RUN_NUMBERS=(0 1 2)
METHODS_LIST=("LengthDetection")


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
echo "End all runs"