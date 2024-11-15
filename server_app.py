

from flwr.common import Parameters
import flwr as fl

from models import known_events as ke

from metrics import apply_metrics
from dataloader import load_data
from op.aux import Color

import argparse
import yaml


class CustomStrategy(fl.server.strategy.Strategy):
    def __init__(self, config) -> None:
        self.num_clients = int(config["amount_clients"])
        self.data = load_data(config=config, num_client=0, num_run=0)
        print(self.data)

    def initialize_parameters(self, client_manager):
        self.model = ke.KnownEvents()
        return Parameters(
            tensors=list(self.model.get_weights()),
            tensor_type="Custom"
        )

    def configure_fit(self, server_round, parameters, client_manager):
        instructions = []
        for client in client_manager.sample(num_clients=self.num_clients):
            instructions.append((client, fl.common.FitIns(parameters, {})))

        return instructions
    
    def aggregate_fit(self, server_round, results, failures):
        for fail in failures:
            print(Color.red(fail))

        weights = []
        for _, result in results:
            weights.append(set(
                fl.common.parameters_to_ndarrays(result.parameters)[0].tolist())
            )
 
        updated_weights = Parameters(
            tensors=list(ke.update_strategy(self.model, clients_weights=weights)),
            tensor_type="Custom"
        )
    
        return updated_weights, {}


    def aggregate_evaluate(self, server_round, results, failures): pass 
    def configure_evaluate(self, server_round, parameters, client_manager): pass 
    
    def evaluate(self, server_round, parameters):
        self.model.set_weights(set(parameters.tensors))

        print(Color.blue(f"Evaluation round {server_round}:"))
        print(Color.yellow(
            apply_metrics(
                pred_normal=self.model.predict(self.data.test_normal),
                pred_abnormal=self.model.predict(self.data.test_abnormal),
            )
        )) 



parser = argparse.ArgumentParser(description="Server script")
parser.add_argument("--config", required=True)


if __name__ == "__main__":
    args = parser.parse_args()
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    ip = f"{config['General']["server_ip"]}:{config['General']['port']}"
    print(Color.purple(f"Start server {ip}"))

    fl.server.start_server(
        server_address=ip,
        config=fl.server.ServerConfig(
            num_rounds=config["General"]["number_rounds"]
        ),
        strategy=CustomStrategy(config=config["Dataset"])
    )

