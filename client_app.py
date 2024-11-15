from models.known_events import KnownEvents
from dataloader import load_data
from op.aux import Color

from flwr.common import NDArrays
import flwr as fl

import typing as t
import argparse
import yaml


class FlowerClient(fl.client.NumPyClient):
    def __init__(
        self, config: t.Dict[str, t.Any], num_run: int, num_client: int,
    ) -> None:
        self.data = load_data(
            config=config, num_client=num_client, num_run=num_run
        )
        print(self.data)
        self.model = KnownEvents()
        self.n = len(self.data.test_abnormal) + len(self.data.test_normal)

    def fit(self, parameters, config) -> None:
        print(Color.blue("Starting Local Training"))
        self.model.set_weights(parameters[0].tolist())
        results = self.model.fit(self.data.train)
        weights = NDArrays([self.model.get_weights()])
        print(Color.blue("Local Training Complete"))

        return weights, len(self.data.train), {"Loss": results}



parser = argparse.ArgumentParser(description="Server script")
parser.add_argument("--config", required=True)
parser.add_argument("--num_client", required=True, type=int)


if __name__ == "__main__":
    args = parser.parse_args()
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    num_run = 0
    num_client = args.num_client
    ip = f"{config['General']["client_ip"]}:{config['General']['port']}"

    print(Color.purple("Starting server:"), ip)
    fl.client.start_client(
        server_address=ip,
        client=FlowerClient(
            config=config["Dataset"], 
            num_run=num_run, 
            num_client=num_client
        ).to_client()
    )


