from models._imodel import Model
from models import _list

from dataloader import load_data
from op.aux import Color

from flwr.common import NDArrays
import flwr as fl

import typing as t
import argparse
import yaml


class FlowerClient(fl.client.NumPyClient):
    def __init__(
        self, 
        config: t.Dict[str, t.Any], 
        num_run: int, 
        num_client: int,
        model: Model,
    ) -> None:
        self.data = load_data(
            config=config, num_client=num_client, num_run=num_run
        )
        print(self.data)
        self.model = model()
        self.n = len(self.data.test_abnormal) + len(self.data.test_normal)

    def fit(self, parameters, config) -> None:
        print(Color.blue("Starting Local Training"))
        self.model.set_weights(parameters[0].tolist())
        results = self.model.fit(self.data.train)
        weights = NDArrays([self.model.get_weights()])
        print(Color.blue("Local Training Complete"))

        return weights, len(self.data.train), {"Loss": results}



parser = argparse.ArgumentParser(description="Server script")
parser.add_argument("--config", required=True, help="Configuration file")
parser.add_argument("--num_client", required=True, type=int)
parser.add_argument(
    "--method", help=f"Select one of this {list(_list.keys())}", required=True
)
parser.add_argument("--run_number", default=0, help="Run number (Default: 0)", type=int)


if __name__ == "__main__":
    args = parser.parse_args()
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    num_client = args.num_client
    ip = f"{config['General']["client_ip"]}:{config['General']['port']}"

    print(Color.purple("Starting server:"), ip)
    fl.client.start_client(
        server_address=ip,
        client=FlowerClient(
            config=config["Dataset"], 
            num_run=args.run_number, 
            num_client=num_client,
            model=_list[args.method]["Method"]
        ).to_client()
    )


