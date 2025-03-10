
from flower.flw_strategy import CustomStrategy
from models import model_init, _list
from client_app import FlowerClient
from server_app import _select_thres

from flwr.server.superlink.fleet.vce.backend.backend import BackendConfig
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.simulation import run_simulation
from flwr.client import ClientApp
from flwr.common import Context

import argparse
import torch
import yaml


def client_fn(context: Context):
    partition_id = context.node_config["partition-id"]
    return FlowerClient(
        config=config["Dataset"],
        num_client=partition_id,
        num_run=args.run_number,
        amount_clients=args.amount_clients,
        model=model_init(args.method)["Method"]
    ).to_client()



def server_fn(context: Context) -> ServerAppComponents:
    thres = _select_thres(config["Dataset"]["dataset_path"])
    model = model_init(args.method, thres=thres)
    strategy = CustomStrategy(
        config=config["Dataset"],
        model=model["Method"],
        num_run=args.run_number,
        update_strategy=model["Update"],
        amount_clients=args.amount_clients,
    )
    config_server = ServerConfig(num_rounds=1)  

    return ServerAppComponents(strategy=strategy, config=config_server)



parser = argparse.ArgumentParser(description="Server script")
parser.add_argument(
    "--config", required=True, type=str, help="Path to configuration file"
)
parser.add_argument("--amount_clients", required=True, type=int)
parser.add_argument(
    "--device", default="cpu", type=str, help="cpu or the number for GPU device [0,1,2..]"
)
parser.add_argument("--run_number", default=0, help="Run number (Default: 0)", type=int)
parser.add_argument(
    "--method", help=f"Select one of this {list(_list.keys())}", required=True, nargs="+"
)

args = parser.parse_args()

if __name__ == "__main__":
    args = parser.parse_args()
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    num_clients = args.amount_clients

    if args.device == 'cpu':
        DEVICE = torch.device("cpu") 
    else:
        DEVICE = torch.device(f"cuda:{int(args.device)}") 

    client = ClientApp(client_fn=client_fn)

    # Flower Server
    server = ServerApp(server_fn=server_fn)

    # Specify the resources each of your clients need
    # If set to none, by default, each client will be allocated 2x CPU and 0x GPUs
    backend_config = {"client_resources": {"num_cpus": 0.25}}
    if DEVICE.type == "cuda":
        backend_config = {"client_resources": {"num_cpus": 1, "num_gpus": 1.0}}

    # Run simulation
    run_simulation(
        server_app=server,
        client_app=client,
        num_supernodes=num_clients,
        backend_config=BackendConfig(backend_config),
    )