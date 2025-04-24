import numpy as np
import torch
import yaml
from collections import OrderedDict
from typing import List
import argparse

import flwr as fl
from flwr.client import ClientApp
from flwr.common import Context
from flwr.server import ServerApp, ServerConfig, ServerAppComponents
from flwr.server.strategy import FedAvg
from flwr.simulation import run_simulation
from flwr.common import ndarrays_to_parameters, NDArrays, Scalar
from typing import Dict, List, Optional, Tuple
from flwr.server import ServerApp, ServerConfig, ServerAppComponents
from flwr.server.strategy import FedAvg

from dataloader import load_data
import op.ml_tools as ml_tools
from models.ml_models import deeplog, loganomaly
from flower.ml_flower_tools import DeepLogClient
import flower.ml_flower_tools as ml_flower_tools

def client_fn(context: Context):
        
    partition_id = context.node_config["partition-id"]
    data = load_data(config=config_params["Dataset"], num_client=partition_id, amount_clients=config_params["Dataset"]["amount_clients"])
    if config_params["Deeplog"]['validation_rate'] != 0:
        val_split = int(len(data.train)*(1-config_params["Deeplog"]['validation_rate']))
        train_data=data.train[:val_split]
        val_data=data.train[val_split:]
    else:
        train_data = data.train
        val_data = data.train

    model = config_params['model'](input_size=config_params['Deeplog']['input_size'], 
                        hidden_size=config_params['Deeplog']['hidden_size'], 
                        num_layers=config_params['Deeplog']['num_layers'], 
                        num_keys=config_params['Deeplog']['num_classes'])

    return DeepLogClient(partition_id, model, train_data, val_data, config_params, DEVICE).to_client()

def evaluate(server_round: int, parameters: NDArrays, config: Dict[str, Scalar]) -> Optional[Tuple[float, Dict[str, Scalar]]]:
    global_model = config_params['model'](input_size=config_params['Deeplog']['input_size'], 
                        hidden_size=config_params['Deeplog']['hidden_size'], 
                        num_layers=config_params['Deeplog']['num_layers'], 
                        num_keys=config_params['Deeplog']['num_classes']).to(DEVICE)
    data = load_data(
        config=config_params["Dataset"],
        num_client=0,
        amount_clients=config_params["Dataset"]["amount_clients"],
    )
    ml_flower_tools.set_parameters(global_model, parameters)  # Update model with the latest parameters
    P, R, F1, FP, FN, TP, TN, prediction_time = ml_tools.predict_unsupervised(global_model, data,
                        window_size=config_params['Deeplog']['window_size'], 
                        input_size=config_params['Deeplog']['input_size'], 
                        num_candidates=config_params['Deeplog']['num_candidates'],
                        num_classes=config_params['Deeplog']['num_classes'], 
                        device=DEVICE)
    print(f"Server-side evaluation F1-score {F1} / FP {FP} / FN {FN} / Precision {P} / Recall {R} / Prediction_time {prediction_time}")
    return 0, {"F1": F1}
    
def fit_metrics_aggregation_fn(metrics_list):
    metrics_dicts = [metrics for _, metrics in metrics_list]
    max_training_time = max(metrics["training_time"] for metrics in metrics_dicts)
    return {"max_training_time": max_training_time}



def server_fn(context: Context) -> ServerAppComponents:
    # Define the federated learning strategy
    strategy = FedAvg(
        fraction_fit=1,  # Sample 100% of available clients for training
        fraction_evaluate=1,  # Sample 50% of available clients for evaluation
        min_fit_clients=num_clients,  # Never sample less than 10 clients for training
        min_evaluate_clients=num_clients,  # Never sample less than 5 clients for evaluation
        min_available_clients=num_clients,  # Wait until all 10 clients are available
        initial_parameters=ndarrays_to_parameters(initial_params),
        evaluate_fn=evaluate,    
        fit_metrics_aggregation_fn=fit_metrics_aggregation_fn
    )
    
    # Configure the server
    config = ServerConfig(num_rounds=num_rounds)  # Set the number of federated learning rounds

    # Return the ServerAppComponents
    return ServerAppComponents(strategy=strategy, config=config)


parser = argparse.ArgumentParser(description="Server script")
parser.add_argument("--config", required=True, type=str, help="Path to configuration file")
parser.add_argument("--num_clients", required=True, type=int)
parser.add_argument("--model", required=True, type=str, help="deeplog or loganomaly")
parser.add_argument("--device", required=True, type=str, help="cpu or the number for GPU device [0,1,2..]")

args = parser.parse_args()

if __name__ == "__main__":

    if args.device == 'cpu':
        DEVICE = torch.device("cpu") 
    else:
        DEVICE = torch.device(f"cuda:{int(args.device)}") 

    config_path = args.config #set the config data path 'config_files/hdfs_iid.yaml' 
    with open(config_path, "r") as f:
            config_params = yaml.safe_load(f)

    config_params['Dataset']['amount_clients'] = args.num_clients
    num_clients = config_params['Dataset']['amount_clients']
    num_rounds = config_params['General']['number_rounds']
    
    if args.model == 'deeplog':
        config_params['model'] = deeplog
    elif args.model == 'loganomaly':
       config_params['model'] = loganomaly

    #Initialise first global model
    net=config_params['model'](input_size=config_params['Deeplog']['input_size'], 
                            hidden_size=config_params['Deeplog']['hidden_size'], 
                            num_layers=config_params['Deeplog']['num_layers'], 
                            num_keys=config_params['Deeplog']['num_classes'])

    initial_params = ml_flower_tools.get_parameters(net)
    #global_validation_data = ml_flower_tools.create_global_validation_data(config_params) #validation for hyperparameters optimisation

    # Flower client
    client = ClientApp(client_fn=client_fn)

    # Flower Server
    server = ServerApp(server_fn=server_fn)

    # Specify the resources each of your clients need
    # If set to none, by default, each client will be allocated 2x CPU and 0x GPUs
    backend_config = {"client_resources": None}
    if DEVICE.type == "cuda":
        backend_config = {"client_resources": {"num_cpus": 1, "num_gpus": 1.0}}

    # Run simulation
    run_simulation(
        server_app=server,
        client_app=client,
        num_supernodes=num_clients,
        backend_config=backend_config,
    )

