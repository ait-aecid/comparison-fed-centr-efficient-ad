
# Arguments run
RUN_NUMBER=0
CONFIG_PATH="config_files/hdfs_no_iid.yaml"
NUM_CLIENTS=10
METHODS="KnowEvents"

ITER=$(($NUM_CLIENTS - 1))
for i in $(seq 0 $ITER); do 
    echo Starting client number: $i
    echo "python client_app.py --config $CONFIG_PATH --method $METHODS --run_number $RUN_NUMBER --num_client $i"
    python client_app.py --config $CONFIG_PATH --method $METHODS --run_number $RUN_NUMBER --num_client $i &
done
