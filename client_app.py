from models.known_events import KnownEvents
from metrics import apply_metrics
from dataloader import load_data
from op.aux import Color

import flwr as fl

import typing as t


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
        self.model.set_weights(parameters)
        results = self.model.fit(self.data.train)
        print(Color.blue("Local Training Complete"))
        return self.model.get_weights(), len(self.data.train), results

    def evaluate(self, parameters, config):
        print(Color.blue("Starting Evaluation"))
        self.model.set_weights(parameters)

        results = apply_metrics(
            pred_normal=self.model.predict(self.data.test_normal),
            pred_abnormal=self.model.predict(self.data.test_abnormal)
        )
        print(Color.blue("Evaluation Complete"))

        return 0, self.n, results.as_dict()


args = {
    "dataset_path": "datasets/BGL",
    "amount_clients": 3,
    "seed_number": 2,
    "train_per": 0.1,
}
num_run = 0
num_client = 0
ip = "127.0.0.1:8080"

if __name__ == "__main__":
    print(Color.purple("Starting server:"), ip)
    fl.client.start_client(
        server_address=ip,
        client=FlowerClient(
            config=args, num_run=num_run, num_client=num_client
        )
    )


