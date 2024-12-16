from dataloader import load_data, DataWrapper
from metrics import apply_metrics
from models._imodel import Model
from op.aux import Color


import flwr as fl


import time
import typing as t


# %% Customizable methods in the strategy
def add_time_client(
    server_round: int, metrics: t.Dict[str, t.Any]
) -> t.Dict[str, float]:    

    time_metrics = list(filter(lambda x : "Time" in x, metrics.keys()))
    return {
         f"Round {server_round} {k}": metrics[k] for k in time_metrics 
    }


def update_weights(
    server_round: int, 
    model: Model,
    weights: t.List[t.List[t.Any]],
    update_strategy: t.Callable[[Model, t.List[t.List[t.Any]]], t.List[t.Any]]
) -> t.Tuple[t.List[t.Any], t.Dict[str, float]]:
    
    start = time.time()
    updated_weights = fl.common.ndarrays_to_parameters(
        [update_strategy(model, clients_weights=weights)]
    )
    end = time.time() - start
    
    return updated_weights, {f"Round {server_round} update": end}


def do_metrics(
    server_round: int, 
    data: DataWrapper,
    model: Model,
    times : t.Dict[str, float]
) -> None: 

    print(Color.blue(f"Evaluation round {server_round}:"))
    print(Color.blue("  - Setting up threshold"))

    start = time.time()
    model.set_threshold(
        X_normal=data.test_normal, X_abnormal=data.test_abnormal
    )
    end = time.time() - start
    times[f"Round {server_round} threshold selection"] = end

    start = time.time()
    print(Color.blue("  - Doing evaluation with threshold"))
    pred_normal = model.predict(data.test_normal)
    pred_abnormal = model.predict(data.test_abnormal)
    end = time.time() - start
    times[f"Round {server_round} evaluation"] = end

    print(Color.yellow(
        apply_metrics(
            pred_normal=pred_normal,
            pred_abnormal=pred_abnormal,
            times=times,
        )
    ))


# %% Strategy

class CustomStrategy(fl.server.strategy.Strategy):
    def __init__(
        self,
        config: t.Dict[str, t.Any],
        model: Model,
        num_run: int,
        update_strategy: t.Callable[[Model, t.List[t.List[t.Any]]], t.List[t.Any]]
    ) -> None:
        self.num_clients = int(config["amount_clients"])
        self.data = load_data(config=config, num_client=0, num_run=num_run)
        self.model, self.update_strategy = model, update_strategy
        self.times = {}
        print(self.data)

    def initialize_parameters(self, client_manager):
        return fl.common.ndarrays_to_parameters([self.model.get_weights()])

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
            self.times.update(add_time_client(server_round, metrics=result.metrics))
            weights.append(
                fl.common.parameters_to_ndarrays(result.parameters)[0].tolist()
            )
        updated_weights, times = update_weights(
            server_round=server_round, 
            model=self.model, 
            weights=weights,
            update_strategy=self.update_strategy
        )
        self.times.update(times)

        return updated_weights, {}

    def evaluate(self, server_round, parameters):
        weights = fl.common.parameters_to_ndarrays(parameters)[0].tolist()
        self.model.set_weights(weights)
        do_metrics(
            server_round=server_round, 
            data=self.data, 
            model=self.model,
            times=self.times
        )

    def aggregate_evaluate(self, server_round, results, failures): pass
    def configure_evaluate(self, server_round, parameters, client_manager): pass