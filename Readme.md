# Comparison ...

This is the official code from ....

Most of the code is subjetct to GNU license. Files **ml_tools.py** and **ml_models.py** are based from this [repository](https://github.com/d0ng1ee/logdeep?tab=MIT-1-ov-file)
 and under MIT license (license added in each file).

## Datasets
The datasets that were use are:

*   **HDFS**: .....
*   **BGL**: .....

## Scripts Simple methods
Current scripts in the repository:
*   **client_app.py**: start a client server.
*   **server_app.py**: start the main server.

### Example Single method:
To run the code:
- Running server:
```
python server_app.py --config config_files/hdfs_iid.yaml --method 3-Gram --run_number 0
```
- Running client:
```
python client_app.py --config config_files/hdfs_iid.yaml --method 3-Gram --run_number 0 --num_client 0
```

### Example Combine methods:
To run the code:
- Running server:
```
python server_app.py --config config_files/bgl_no_iid.yaml --method KnowEvents LengthDetection --run_number 0
```
- Running client:
```
python client_app.py --config config_files/bgl_no_iid.yaml --method KnowEvents LengthDetection --run_number 0 --num_client 0
```

### Run multiple methods at once
Change the arguments inside the script:
```
sh run_multiple_clients.sh
```

## Scripts ML methods
Current scripts in the repository:
*   **ml_centralize.py**: Run the ml methods as centralize.
*   **ml_federated_simulation.py**: Run the ml methods as federated.

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