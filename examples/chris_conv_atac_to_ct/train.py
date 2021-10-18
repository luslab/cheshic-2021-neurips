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
peak_count = 116490
label_count = 21
test_perc = 0.05

batch_size = 20
learning_rate = 0.001
momentum = 0.9
epochs = 10
loss_print_freq = 100
eval_freq = 1000

########### CLASSES ##########

class RNA_funct( ):




class Load_Dataset(Dataset):
    
    def __init__(self):
        # main peak vs cell matrix to be fed to transformer
        self.dataset = sc.read(data_path)
        self.X = self.dataset.X.todense()

        # extra features from ATAC to include in paralel
        self.PCs = self.dataset.varm['PCs'] 
        self.Genes = self.dataset. ##### TO DO 
        self.QCs = self.dataset.obs[['nCount_peaks', 'atac_fragments', 'reads_in_peaks_frac', 'blacklist_fraction', 'nucleosome_signal']].to_numpy()


        # get ground truth

        # load scRNAseq
        # XXXXXXXXX

        # RNA_funct output for another truth
        # XXXXXXXXX

        self.truth = self.dataset.obs.['label_id']


    def __len__(self):
        return len(self.dataset.obs.index)


    def __getitem__(self, idx):
        label = self.truth.iloc[idx]

        data_transformer = np.asarray(self.X[idx])
        data_extra = 

        data = np.squeeze(data)
        return data, label



# transformer
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed1 = nn.Linear(peak_count, 20)
        self.fc1 = nn.Linear(20, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, label_count)

    def forward(self, x):
        x = F.relu(self.embed1(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x




# extra ATAC
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed1 = nn.Linear(peak_count, 20)
        self.fc1 = nn.Linear(20, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, label_count)

    def forward(self, x):
        x = F.relu(self.embed1(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x




# Final
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed1 = nn.Linear(peak_count, 20)
        self.fc1 = nn.Linear(20, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, label_count)

    def forward(self, x):
        x = F.relu(self.embed1(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x






########### MAIN ##########

# Log GPU status
is_cuda = torch.cuda.is_available()
logging.info("Cuda available: " + str(is_cuda))
if is_cuda:
    current_device = torch.cuda.current_device()
    #torch.cuda.device(current_device)
    device_count = torch.cuda.device_count()
    logging.info("Cuda device count: " + device_count)
    device_name = torch.cuda.get_device_name(current_device)
    logging.info("Cuda device name: " + device_name)

logging.info('Initialising')

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