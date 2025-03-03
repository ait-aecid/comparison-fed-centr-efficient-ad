import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from tqdm import tqdm
import time
from collections import OrderedDict
from typing import List

class CustomDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return torch.tensor(self.data[index]).unsqueeze(-1).float(), torch.tensor(self.labels[index])
    


class deeplog(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_keys):
        super(deeplog, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size,
                            hidden_size,
                            num_layers,
                            batch_first=True)
        self.fc = nn.Linear(hidden_size, num_keys)

    def forward(self, features, device):
        input0 = features[0]
        h0 = torch.zeros(self.num_layers, input0.size(0),
                         self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, input0.size(0),
                         self.hidden_size).to(device)
        out, _ = self.lstm(input0, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out
    

def sliding_window(sequences, window_size):
    result_logs = []
    labels = []

    for sequence in sequences:
        # Convert the sequence to a tuple of integers (subtract 1 to make it 0-indexed)
        sequence = tuple(map(lambda n: n - 1, map(int, sequence)))

        for i in range(len(sequence) - window_size):
            sequential_pattern = list(sequence[i:i + window_size])
            sequential_pattern = np.array(sequential_pattern)
            result_logs.append(sequential_pattern)
            labels.append(sequence[i + window_size])

    return result_logs, labels

def train(model, train_data, window_size, batch_size, local_epochs, learnin_rate, device):

    train_data, train_labels = sliding_window(sequences=train_data, window_size=window_size)
    dataset = CustomDataset(train_data, train_labels)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model.to(device)

    accumulation_step = 1
    start_time = time.time()
    for epoch in range(local_epochs):
        print("Starting epoch: ", epoch)
        model.train()
        optimizer = torch.optim.Adam(model.parameters(),lr=learnin_rate,betas=(0.9, 0.999))
        optimizer.zero_grad()
        criterion = nn.CrossEntropyLoss()
        tbar = tqdm(data_loader, desc="\r")
        total_losses = 0
        for i, (log, label) in enumerate(tbar):
            features = [log.to(device)]
            output = model(features=features, device=device)
            loss = criterion(output, label.to(device))
            total_losses += float(loss)
            loss /= accumulation_step
            loss.backward()
            if (i + 1) % accumulation_step == 0:
                optimizer.step()
                optimizer.zero_grad()
            tbar.set_description("Train loss: %.5f" % (total_losses / (i + 1)))
    
    elapsed_time = time.time() - start_time
    print('elapsed_time: {}'.format(elapsed_time))
    
    return total_losses / len(data_loader), float(elapsed_time)

def generate(sequences, window_size):
    seq = {}
    length = 0
    
    for sequence in sequences:
        # Convert string elements to integers and subtract 1 from each
        ln = [int(n) - 1 for n in sequence]
        
        # Pad the sequence with -1 if it's shorter than window_size + 1
        ln = ln + [-1] * (window_size + 1 - len(ln))
        
        # Create a tuple from the sequence and update the hdfs dictionary
        sequence_tuple = tuple(ln)
        seq[sequence_tuple] = seq.get(sequence_tuple, 0) + 1
        length += 1
    
    print(f'Number of unique sequences: {len(seq)}')
    return seq, length

def predict_unsupervised(model, data, window_size, input_size, num_candidates, device):
    model = model.to(device)
    model.eval()
    test_normal_loader, test_normal_length = generate(data.test_normal, window_size)
    test_abnormal_loader, test_abnormal_length = generate(data.test_abnormal, window_size)
    TP = 0
    FP = 0
    # Test the model
    start_time = time.time()
    with torch.no_grad():
        for line in tqdm(test_normal_loader.keys()):
            for i in range(len(line) - window_size):
                seq0 = line[i:i + window_size]
                label = line[i + window_size]
                seq0 = torch.tensor(seq0, dtype=torch.float).view(
                    -1, window_size, input_size).to(device)
                label = torch.tensor(label).view(-1).to(device)
                output = model(features=[seq0], device=device)
                predicted = torch.argsort(output,
                                            1)[0][-num_candidates:]
                if label not in predicted:
                    FP += test_normal_loader[line]
                    break
    with torch.no_grad():
        for line in tqdm(test_abnormal_loader.keys()):
            for i in range(len(line) - window_size):
                seq0 = line[i:i + window_size]
                label = line[i + window_size]
                seq0 = torch.tensor(seq0, dtype=torch.float).view(
                    -1, window_size, input_size).to(device)
                label = torch.tensor(label).view(-1).to(device)
                output = model(features=[seq0], device=device)
                predicted = torch.argsort(output,
                                            1)[0][-num_candidates:]
                if label not in predicted:
                    TP += test_abnormal_loader[line]
                    break

    # Compute precision, recall and F1-measure
    FN = test_abnormal_length - TP
    P = 100 * TP / (TP + FP)
    R = 100 * TP / (TP + FN)
    F1 = 2 * P * R / (P + R)
    TN = test_normal_length - FP
    print(
        'false positive (FP): {}, false negative (FN): {}, Precision: {:.3f}%, Recall: {:.3f}%, F1-measure: {:.3f}%'
        .format(FP, FN, P, R, F1))
    print('Finished Predicting')
    elapsed_time = time.time() - start_time
    print('elapsed_time: {}'.format(elapsed_time))

    return P, R, F1, FP, FN, TP, TN, elapsed_time
    
def validation_loss(model, validation_data, device):

    val_data, val_labels = sliding_window(sequences=validation_data, window_size=10)
    val_dataset = CustomDataset(val_data, val_labels)
    val_loader = DataLoader(val_dataset, shuffle=True)
    
    model.to(device)
    model.eval()
    total_losses = 0
    criterion = nn.CrossEntropyLoss()
    tbar = tqdm(val_loader, desc="\r")
    num_batch = len(val_loader)
    for i, (log, label) in enumerate(tbar):
        with torch.no_grad():
            output = model(features=[log.to(device)], device=device)
            loss = criterion(output, label.to(device))
            total_losses += float(loss)
    val_loss = total_losses / num_batch
    print("Validation loss:", val_loss)

    return val_loss

def validation_unsupervised(model, validation_data, window_size, input_size, num_candidates, device='cpu'):
    model = model.to(device)
    model.eval()
    test_normal_loader, test_normal_length = generate(validation_data, window_size)
    TP = 0
    FP = 0
    # Test the model
    start_time = time.time()
    with torch.no_grad():
        for line in tqdm(test_normal_loader.keys()):
            for i in range(len(line) - window_size):
                seq0 = line[i:i + window_size]
                label = line[i + window_size]
                seq0 = torch.tensor(seq0, dtype=torch.float).view(
                    -1, window_size, input_size).to(device)
                label = torch.tensor(label).view(-1).to(device)
                output = model(features=[seq0], device=device)
                predicted = torch.argsort(output,
                                            1)[0][-num_candidates:]
                if label not in predicted:
                    FP += test_normal_loader[line]
                    break

    # Compute precision, recall and F1-measure
    FP_rate = 100*FP/test_normal_length
    print(
        'false positive (FP): {}, false positive rate (FP_rate): {:.3f}%'
        .format(FP, FP_rate))
    print('Finished Predicting')
    elapsed_time = time.time() - start_time
    print('elapsed_time: {}'.format(elapsed_time))

    val_loss = validation_loss(model, validation_data, device)

    return FP, FP_rate, val_loss

