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

########### CLASSES ##########

class Load_Dataset(Dataset):
    """
    This takes your ATAC data (as a directory location) and RNA data and a user defined function 
    to transform the RNA in a meaningful way.
    
    It stores your data, performs your transformation and sets them up in an iterable way for pytorch.
    """
    def __init__(self, ATAC_path, RNA_path, r_func):
        # main peak vs cell matrix to be fed to transformer
        self.dataset = sc.read(ATAC_path)
        self.X = self.dataset.X.todense()
        self.n_cells = self.X.shape[0]
        self.n_peaks = self.X.shape[1]
        
        # get ground truth
        self.RNA = sc.read(RNA_path)    
        self.Y = r_func(self.RNA)
        self.n_labels = len(np.unique(self.Y))
        self.r_func = r_func

    def __len__(self):
        return len(self.dataset.obs.index)

    def __getitem__(self, idx):
        label = np.squeeze(np.asarray(self.Y[idx]))
        data = np.asarray(self.X[idx])
        data = np.squeeze(data)
        return data, label


class ATACDataset(Dataset):
    def __init__(self, data_path):
        self.dataset = sc.read(data_path)
        self.X = self.dataset.X.todense()
        self.n_cells = self.X.shape[0]
        self.n_peaks = self.X.shape[1]

    def __len__(self):
        return len(self.dataset.obs.index)

    def __getitem__(self, idx):
        label = self.dataset.obs.iloc[idx]['label_id']
        data = np.asarray(self.X[idx])
        #data = data[np.newaxis, :, :]
        data = np.squeeze(data)
        return data, label

########### FUNCTIONS ##########

def save(model, file_prefix='unnamed_model', working_dir=".",):
    """ save the given model with given name in given directory"""
    file_path = working_dir + '/model_params/' + file_prefix
    torch.save(model.state_dict(), file_path)

def load(model, path, eval=True, cuda=True):
    """
    load model weights
    :param model: model with weights to be updated
    :param path: path to weights
    :param eval: set model to eval() mode
    :param cuda: enable GPU/cuda
    :return: Nothing
    """
    if cuda:
        model.load_state_dict(torch.load(path))
    else:
        model.load_state_dict(torch.load(path, map_location='cpu'))
    if eval:
        model.eval()
