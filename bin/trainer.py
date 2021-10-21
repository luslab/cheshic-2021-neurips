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

########### FUNCTIONS ##########

def train(dataset, model, batch_size, optimizer, learning_rate, criterion, epochs, 
          test_pct=0.05, loss_print_freq=50, eval_freq=100):
    logging.info('Initialising')
    logging.info('Setup')

    test_num = math.floor(dataset.n_cells * test_pct)
    train_num = dataset.n_cells - test_num

    logging.info('Train examples: ' + str(train_num))
    logging.info('Test examples: ' + str(test_num))

    # Load and split dataset
    logging.info('Loading and splitting dataset')
    train_set, test_set = torch.utils.data.random_split(dataset, [train_num, test_num])

    train_dataloader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_set, batch_size=1, shuffle=True)

    # ## DEBUG ##
    # train_features, train_labels = next(iter(train_dataloader))
    # print(f"Feature batch shape: {train_features.size()}")
    # print(f"Labels batch shape: {train_labels.size()}")

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

                print(accuracy)
                print(test_count)

                single_test_loss = test_loss / test_count
                single_test_accuracy = accuracy / test_count
                test_losses.append(single_test_loss)
                test_accuracy.append(single_test_accuracy)
                model.train()

                logging.info('EVAL - [epoch-%d, %5d] test_loss: %.3f test_accuracy: %.3f' % (epoch + 1, i + 1, single_test_loss, single_test_accuracy * 100))

    logging.info('Finished Training')