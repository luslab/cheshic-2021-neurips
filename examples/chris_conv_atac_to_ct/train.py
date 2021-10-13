########### IMPORT LIBS ##########

import logging
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

logging.basicConfig(level=logging.INFO)

########### VARIABLES ##########

example_count = 22463
label_count = 21
test_perc = 0.05

batch_size = 64
learning_rate = 0.001
momentum = 0.9
epochs = 1
loss_print_freq = 10

########### CLASSES ##########

class ATACDataset(Dataset):
    def __init__(self, data_path):
        self.dataset = sc.read(data_path)
        self.X = self.dataset.X.todense()

    def __len__(self):
        return len(self.dataset.obs.index)

    def __getitem__(self, idx):
        label = self.dataset.obs.iloc[idx]['label_id']
        data = np.asarray(self.X[idx])
        data = data[np.newaxis, :, :]
        return data, label

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, (1, 100))
        #self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, (1, 100))
        self.fc1 = nn.Linear(16 * 1 * 100, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, label_count)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # If the size is a square, you can specify with a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = torch.flatten(x, 1) # flatten all dimensions except the batch dimension
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

########### MAIN ##########

logging.info('Training...')

logging.info('Setup')

# Init file paths
file_path_training = 'preprocessed.h5ad'

# Init parameters
test_num = math.floor(example_count * test_perc)
train_num = example_count - test_num

logging.info('Train examples: ' + str(train_num))
logging.info('Test examples: ' + str(test_num))

# Load and split dataset
logging.info('Loading and splitting dataset')
dataset = ATACDataset(file_path_training)
train_set, test_set = torch.utils.data.random_split(dataset, [train_num, test_num])

train_dataloader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_set, batch_size=batch_size, shuffle=True)

## DEBUG ##
# train_features, train_labels = next(iter(train_dataloader))
# print(f"Feature batch shape: {train_features.size()}")
# print(f"Labels batch shape: {train_labels.size()}")

# Create model
logging.info('Creating model')
net = Net()
print(net)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=momentum)

# Train model
logging.info('Training model')

for epoch in range(epochs):
    logging.info('Epoch: ' + str(epoch))

    running_loss = 0.0
    for i, data in enumerate(train_dataloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % loss_print_freq == 1999:    # print every n mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

logging.info('Finished Training')