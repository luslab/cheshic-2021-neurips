#!/usr/bin/env python3
# -*- coding: utf-8 -*-

########### IMPORT LIBS ##########

import logging
import argparse
import math

import numpy as np
import pandas as pd
import scanpy as sc

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

########### SETUP LIBS ##########

logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')

########### VARIABLES ##########

example_count = 22463
peak_count = 116490
label_count = 21
test_perc = 0.05

input_dimensions = 200
batch_size = 20
learning_rate = 0.001
momentum = 0.9
epochs = 100
loss_print_freq = 500
eval_freq = 1000

########### CLASSES ##########

class GraphDataset(Dataset):
    def __init__(self, label_path, embed_path):
        self.dataset = sc.read(label_path)

        # Load and sort data
        df_embed = pd.read_csv(embed_path, header=None, sep='\t', index_col=0)
        df_embed_1 = df_embed[df_embed.index.str.contains("cell")]
        df_embed_1['name'] = df_embed_1.index
        df_embed_1[['type', 'id']] = df_embed_1['name'].str.split('-', expand=True)
        df_embed_1['id'] = df_embed_1['id'].astype(int)
        df_embed_1 = df_embed_1.sort_values('id')
        self.embed = df_embed_1.drop(columns=['name', 'type', 'id'])

    def __len__(self):
        return len(self.dataset.obs.index)

    def __getitem__(self, idx):
        label = self.dataset.obs.iloc[idx]['label_id']
        data = np.asarray(self.embed.values[idx]).astype(np.float32)
        data = np.squeeze(data)
        return data, label

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(input_dimensions, 1000)
        self.fc2 = nn.Linear(1000, 1000)
        self.fc3 = nn.Linear(1000, 200)
        self.fc4 = nn.Linear(200, label_count)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x

########### PARSE ARGUMENTS ##########
parser = argparse.ArgumentParser()

## REQUIRED PARAMETERS
parser.add_argument('--atac_data')
parser.add_argument('--graph_data')
args = parser.parse_args()

# ########### MAIN ##########

# Log GPU status
is_cuda = torch.cuda.is_available()
logging.info("Cuda available: " + str(is_cuda))
if is_cuda:
    current_device = torch.cuda.current_device()
    #torch.cuda.device(current_device)
    device_count = torch.cuda.device_count()
    logging.info("Cuda device count: " + str(device_count))
    device_name = torch.cuda.get_device_name(current_device)
    logging.info("Cuda device name: " + str(device_name))

logging.info('Initialising')

logging.info('Setup')

# Init parameters
test_num = math.floor(example_count * test_perc)
train_num = example_count - test_num

logging.info('Train examples: ' + str(train_num))
logging.info('Test examples: ' + str(test_num))

# Load and split dataset
logging.info('Loading and splitting dataset')
dataset = GraphDataset(args.atac_data, args.graph_data)
train_set, test_set = torch.utils.data.random_split(dataset, [train_num, test_num])

train_dataloader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_set, batch_size=1, shuffle=True)

## DEBUG ##
train_features, train_labels = next(iter(train_dataloader))
print(f"Feature batch shape: {train_features.size()}")
print(f"Labels batch shape: {train_labels.size()}")

# Create model
logging.info('Creating model')
model = Net()
print(model)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)

# Train model
logging.info('Training model')

for epoch in range(epochs):
    running_loss = 0.0
    train_losses = []
    test_losses = []
    test_accuracy = []
    for i, data in enumerate(train_dataloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % loss_print_freq == loss_print_freq - 1:
            single_loss = running_loss / loss_print_freq
            train_losses.append(single_loss)
            logging.info('[epoch-%d, %5d] loss: %.3f' % (epoch + 1, i + 1, single_loss))
            running_loss = 0.0

        # eval
        if i % eval_freq == eval_freq - 1:
            model.eval()
            test_loss = 0.0
            accuracy = 0.0

            with torch.no_grad():
                for inputs, labels in test_dataloader:
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    ps = torch.exp(outputs)
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    test_loss += loss.item()
                    accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

            test_count = len(test_dataloader)

            single_test_loss = test_loss / test_count
            single_test_accuracy = accuracy / test_count
            test_losses.append(single_test_loss)
            test_accuracy.append(single_test_accuracy)
            model.train()

            logging.info('EVAL - [epoch-%d, %5d] test_loss: %.3f test_accuracy: %.3f' % (epoch + 1, i + 1, single_test_loss, single_test_accuracy * 100))

logging.info('Finished Training')