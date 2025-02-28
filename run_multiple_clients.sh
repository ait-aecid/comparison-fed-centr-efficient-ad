#!/bin/bash

# Default arguments
DEFAULT_RUN_NUMBER=2
DEFAULT_CONFIG_PATH="config_files/hdfs_iid.yaml"
DEFAULT_NUM_CLIENTS=1
DEFAULT_METHODS="2-Gram LengthDetection"


# Getting final arguments
RUN_NUMBER=${1:-$DEFAULT_RUN_NUMBER}
CONFIG_PATH=${2:-$DEFAULT_CONFIG_PATH}
NUM_CLIENTS=${3:-$DEFAULT_NUM_CLIENTS}
METHODS=${4:-$DEFAULT_METHODS}
METHODS=$(echo "$METHODS" | sed 's/+/ /g')  # Allow to pass multiple methods as agurment. Example: ECVC+KnowEvents -> ECVC KnowEvents


echo "Variables:"
echo "      - RUN_NUMBER: $RUN_NUMBER"
echo "      - CONFIG_PATH: $CONFIG_PATH"
echo "      - NUM_CLIENTS: $NUM_CLIENTS"
echo "      - METHODS: $METHODS"
sleep 2


# Run main server
echo "Start Server"
echo "python server_app.py --config $CONFIG_PATH --method $METHODS --run_number $RUN_NUMBER --amount_clients $NUM_CLIENTS"
python server_app.py --config $CONFIG_PATH --method $METHODS --run_number $RUN_NUMBER  --amount_clients $NUM_CLIENTS&
sleep 5


# Run all the clients
echo "Start Clients"
ITER=$(($NUM_CLIENTS - 1))
for i in $(seq 0 $ITER); do 
    echo Starting client number: $i
    echo "python client_app.py --config $CONFIG_PATH --method $METHODS --run_number $RUN_NUMBER --num_client $i --amount_clients $NUM_CLIENTS"
    python client_app.py --config $CONFIG_PATH --method $METHODS --run_number $RUN_NUMBER --num_client $i  --amount_clients $NUM_CLIENTS&
done


# Wait until finish
wait
echo "All jobs are finish"