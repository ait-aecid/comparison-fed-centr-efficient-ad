import torch
from typing import Dict, List, Optional, Tuple

import flwr as fl
from flwr.common import Context
from flwr.common import ndarrays_to_parameters, NDArrays, Scalar

from dataloader import load_data
import op.ml_tools as ml_tools
from models.ml_models import deeplog

# import wandb

def get_parameters(model):
    return [val.cpu().numpy() for _, val in model.state_dict().items()]

def set_parameters(model, parameters):
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = {k: torch.tensor(v) for k, v in params_dict}
    model.load_state_dict(state_dict, strict=True)

def create_global_validation_data(config_params):
    global_validation_data = []
    for client in range(config_params["Dataset"]["amount_clients"]):
        data = load_data(config=config_params["Dataset"], num_client=client)
        val_split = int(len(data.train)*(1-config_params["Deeplog"]['validation_rate']))
        global_validation_data.extend(data.train[val_split:])

    return global_validation_data


class DeepLogClient(fl.client.NumPyClient):
    def __init__(self, partition_id, model, train_data, val_data, config_params, DEVICE):
        self.partition_id = partition_id
        self.model = model
        self.train_data = train_data
        self.val_data = val_data
        self.config_params = config_params
        self.device = DEVICE

    def get_parameters(self, config):
        print(f"[Client {self.partition_id}] get_parameters")
        return get_parameters(self.model)

    def fit(self, parameters, config):
        print(f"[Client {self.partition_id}] fit, config: {config}")
        set_parameters(self.model, parameters)
        train_loss, elapsed_time = ml_tools.train(self.model, self.train_data, 
                                 window_size=self.config_params['Deeplog']['window_size'], 
                                 batch_size=self.config_params['Deeplog']['batch_size'], 
                                 local_epochs=self.config_params['Deeplog']['max_epoch'], 
                                 learnin_rate=self.config_params['Deeplog']['lr'], 
                                 num_classes=self.config_params['Deeplog']['num_classes'], 
                                 device=self.device)
        return self.get_parameters(self.model), len(self.train_data), {"loss": train_loss, "training_time": elapsed_time}

    def evaluate(self, parameters, config):
        set_parameters(self.model, parameters)
        val_loss = ml_tools.validation_loss(self.model, self.val_data,self.config_params['Deeplog']['window_size'], self.config_params['Deeplog']['num_classes'], self.device)
        FP, TN, FP_rate, val_loss = ml_tools.validation_unsupervised(self.model, self.val_data,
                                window_size=self.config_params['Deeplog']['window_size'], 
                                input_size=self.config_params['Deeplog']['input_size'], 
                                num_candidates=self.config_params['Deeplog']['num_candidates'],
                                num_classes=self.config_params['Deeplog']['num_classes'], 
                                device=self.device)
        id = self.partition_id
        # wandb.log({f"FP_val_client{id}": FP, f"TN_val_client{id}": TN, f"FP_rate_val_client{id}": FP_rate, f"val_loss_client{id}": val_loss})
        return val_loss, len(self.val_data), {"FP_rate": FP_rate, "FP": FP, "TN": TN}


