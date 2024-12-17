# Comparison ...

This is the official code from ....

## Datasets
The datasets that were use are:

*   **HDFS**: .....
*   **BGL**: .....

## Scripts
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

## Federated learning diagram
The strategy was created following the [flower documentation](https://flower.ai/docs/framework/how-to-implement-strategies.html).


![diagram](img/diagram.jpg)