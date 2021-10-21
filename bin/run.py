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

########### IMPORT MODULES ##########

from models import Net
from helpers import Load_Dataset
from RNA_functions import celltype_function
from trainer import train

########### PARSE ARGUMENTS ##########
parser = argparse.ArgumentParser()

## REQUIRED PARAMETERS
parser.add_argument('--training_data') ### NEED TO ADD Y_TRAIN HERE
args = parser.parse_args()

########### MAIN ##########

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


dataset = DATALOADER_FUNCTION_TBC(X_data, Y_data, r_func=celltype_function())
model = Net()

########### VARIABLES ##########

test_perc = 0.05
batch_size = 20
learning_rate = 0.001
momentum = 0.9
epochs = 10
loss_print_freq = 100
eval_freq = 500
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)

########### LET'S GO ##########

train(dataset, model, batch_size, optimizer, learning_rate, criterion, epochs, 
      test_pct, loss_print_freq, eval_freq)

