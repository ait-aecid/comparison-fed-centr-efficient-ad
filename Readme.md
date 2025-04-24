# Federated Learning for Efficient Anomaly Detection in Log Data Comparison

This is the official code from ....

Most of the code is subjetct to GNU license. Files **ml_tools.py** and **ml_models.py** are based from this [repository](https://github.com/d0ng1ee/logdeep?tab=MIT-1-ov-file)
 and under MIT license (license added in each file).

### Install dependencies

To install dependencies do:

```
pip install -r requirements.txt 
```
and 
```
pip install -U "flwr[simulation]"
```
we run this repository with **Python 3.12.3**.

## Datasets
The datasets that were use are: HDFS, BGL.

## Outputs
The expected file outputs are:

*   Each script will return a **distribution_plot.png** with the client data distribution.
*   Light Weight methods will return a **result.csv** with the final results.

## Scripts Light Weight methods
Current scripts in the repository:
*   **client_app.py**: start a client server.
*   **server_app.py**: start the main server.

### Example Single method:
To run the code:
- Running server:
```
python server_app.py --config config_files/hdfs_iid.yaml --method 3-Gram --run_number 0 --amount_clients 3
```
- Running client:
```
python client_app.py --config config_files/hdfs_iid.yaml --method 3-Gram --run_number 0 --num_client 0 --amount_clients 3
```

### Example Combine methods:
To run the code:
- Running server:
```
python server_app.py --config config_files/bgl_no_iid.yaml --method KnowEvents LengthDetection --run_number 0 --amount_clients 3
```
- Running client:
```
python client_app.py --config config_files/bgl_no_iid.yaml --method KnowEvents LengthDetection --run_number 0 --num_client 0 --amount_clients 3
```

### Run multiple methods at once
Change the arguments inside the script to run multiple clients at once:
```
sh run_multiple_clients.sh
```
Or run the commnad bellow to run multiple runs at once:
```
sh run_multiple_runs.sh
```

### Run with the flower simulation function

Warning: the model will be train and the result will be save, but the script will finish with a flower exception.
```
python simulation_app.py  --config config_files/hdfs_iid.yaml --amount_clients 2 --run_number 0 --method KnowEvents
```

## Scripts ML methods
Current scripts in the repository:
*   **ml_centralize.py**: Run the ml methods as centralize.
```
python ml_centralize.py 
```
*   **ml_federated_simulation.py**: Run the ml methods as federated.
```
python ml_federated_simulation.py  --config config_files/hdfs_iid.yaml --num_clients 2 --device cpu
```
## Docker support
To run the code inside a docker container use the file **Dockerfile**. More info in [docker documentation](https://docs.docker.com/).

Build the image with the next command:
```bash
docker build -t fed_comparison .
```
And run the container with:
```bash
docker run --name fed_comparison fed_comparison
```

## Federated learning diagram
The strategy was created following the [flower documentation](https://flower.ai/docs/framework/how-to-implement-strategies.html).


![diagram](img/diagram.jpg)