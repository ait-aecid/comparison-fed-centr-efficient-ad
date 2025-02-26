from dataloader import load_data

import ml_tools
from ml_models import deeplog
import yaml

config_path = 'config_files/hdfs_iid.yaml' # set the config data path
with open(config_path, "r") as f:
    config = yaml.safe_load(f)

config["Dataset"]["amount_clients"] = 1

data = load_data(config=config["Dataset"], num_client=0)
val_split = int(len(data.train)*(1-config["Deeplog"]['validation_rate']))

model = deeplog(input_size=config['Deeplog']['input_size'], 
                hidden_size=config['Deeplog']['hidden_size'], 
                num_layers=config['Deeplog']['num_layers'], 
                num_keys=config['Deeplog']['num_classes'])

train_loss, elapsed_time = ml_tools.train(model, data.train[:val_split], 
                   window_size=config['Deeplog']['window_size'], 
                   batch_size=config['Deeplog']['batch_size'], 
                   local_epochs=config['Deeplog']['max_epoch'], 
                   learnin_rate=config['Deeplog']['lr'], 
                   device='cpu')
# validation
if config["Deeplog"]['validation_rate']!=0:
    FP, FP_rate, val_loss = ml_tools.validation_unsupervised(model, data.train[val_split:],
                        window_size=config['Deeplog']['window_size'], 
                        input_size=config['Deeplog']['input_size'], 
                        num_candidates=config['Deeplog']['num_candidates'], 
                        device='cpu')

# evaluation on the test set
P, R, F1, FP, FN = ml_tools.predict_unsupervised(model, data,
                     window_size=config['Deeplog']['window_size'], 
                     input_size=config['Deeplog']['input_size'], 
                     num_candidates=config['Deeplog']['num_candidates'], 
                     device='cpu')