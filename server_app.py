import flwr as fl

from models._thresholds import ThresHDFS, Thresholds, ThresBGL

from flw_strategy import CustomStrategy
from models import  model_init, _list
from op.aux import Color

import argparse
import yaml


def _select_thres(dataset_path: str) -> Thresholds:
    if "hdfs" in dataset_path.lower():
        return ThresHDFS()
    elif "bgl" in dataset_path.lower():
        return ThresBGL()
    else:
        return Thresholds()


parser = argparse.ArgumentParser(description="Server script")
parser.add_argument("--config", required=True, help="Configuration file")
parser.add_argument(
    "--method", help=f"Select one of this {list(_list.keys())}", required=True, nargs="+"
)
parser.add_argument("--run_number", default=0, help="Run number (Default: 0)", type=int)
parser.add_argument("--amount_clients", required=True, type=int)
parser.add_argument(
    "--force_set_threshold", 
    action="store_true", 
    help="Always set threshold "
)


if __name__ == "__main__":
    args = parser.parse_args()
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    ip = f"{config['General']["server_ip"]}:{config['General']['port']}"
    print(Color.purple(f"Start server {ip}"))
    if args.force_set_threshold:
        thres = Thresholds()
    else:
        thres = _select_thres(config["Dataset"]["dataset_path"])
    model = model_init(args.method, thres=thres)

    fl.server.start_server(
        server_address=ip,
        config=fl.server.ServerConfig(
            num_rounds=config["General"]["number_rounds"],
        ),
        strategy=CustomStrategy(
            config=config["Dataset"],
            model=model["Method"],
            num_run=args.run_number,
            update_strategy=model["Update"],
            amount_clients=args.amount_clients,
        )
    )

