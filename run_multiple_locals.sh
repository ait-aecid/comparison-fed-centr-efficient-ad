#!/bin/bash

# Default arguments
RUN_NUMBER=2
CONFIG_PATH="config_files/bgl_iid.yaml"
NUM_CLIENTS=5


METHODS_LIST=("KnowEvents" "LengthDetection" "ECVC" "Edit" "3-Gram" "2-Gram")
# Run main server
echo "Start Server"

# Run all the clients
echo "Start Clients"
for METHODS in "${METHODS_LIST[@]}"; do
    ITER=$((5 - 1))
    for i in $(seq 0 $ITER); do 
        echo "python server_app.py --config $CONFIG_PATH --method $METHODS --run_number $RUN_NUMBER --amount_clients 1"
        python server_app.py --config $CONFIG_PATH --method $METHODS --run_number $RUN_NUMBER  --amount_clients 1&
        sleep 15
        echo Starting client number: $i
        echo "python client_app.py --config $CONFIG_PATH --method $METHODS --run_number $RUN_NUMBER --num_client $i --amount_clients $NUM_CLIENTS"
        python client_app.py --config $CONFIG_PATH --method $METHODS --run_number $RUN_NUMBER --num_client $i  --amount_clients $NUM_CLIENTS&
        wait
        sleep 1
    done
done

# Wait until finish
echo "All jobs are finish"