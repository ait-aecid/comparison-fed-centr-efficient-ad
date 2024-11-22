import flwr as fl

from op.flw_strategy import CustomStrategy
from models import  _list
from op.aux import Color

import argparse
import yaml



parser = argparse.ArgumentParser(description="Server script")
parser.add_argument("--config", required=True, help="Configuration file")
parser.add_argument(
    "--method", help=f"Select one of this {list(_list.keys())}", required=True
)
parser.add_argument("--run_number", default=0, help="Run number (Default: 0)", type=int)


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
            model=_list[args.method]["Method"],
            num_run=args.run_number,
            update_strategy=_list[args.method]["Update"]
        )
    )

