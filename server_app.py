import flwr as fl

from flw_strategy import CustomStrategy
from models import  model_init, _list
from op.aux import Color

import argparse
import yaml



parser = argparse.ArgumentParser(description="Server script")
parser.add_argument("--config", required=True, help="Configuration file")
parser.add_argument(
    "--method", help=f"Select one of this {list(_list.keys())}", required=True, nargs="+"
)
parser.add_argument("--run_number", default=0, help="Run number (Default: 0)", type=int)
parser.add_argument("--amount_clients", required=True, type=int)


if __name__ == "__main__":
    args = parser.parse_args()
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    ip = f"{config['General']["server_ip"]}:{config['General']['port']}"
    print(Color.purple(f"Start server {ip}"))

    fl.server.start_server(
        server_address=ip,
        config=fl.server.ServerConfig(
            num_rounds=config["General"]["number_rounds"],
        ),
        strategy=CustomStrategy(
            config=config["Dataset"],
            model=model_init(args.method)["Method"],
            num_run=args.run_number,
            update_strategy=model_init(args.method)["Update"],
            amount_clients=args.amount_clients,
        )
    )

