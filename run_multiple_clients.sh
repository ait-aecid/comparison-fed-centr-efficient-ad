
# Arguments run
RUN_NUMBER=2
CONFIG_PATH="config_files/bgl_iid.yaml"
NUM_CLIENTS=5
METHODS="2-Gram LengthDetection"


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