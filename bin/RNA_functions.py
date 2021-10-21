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

########### FUNCTIONS ##########

def celltype_function(RNA_adata):
    """
    this code is taken from Chris, just to demonstrate wrapping into a function
    """
    # Find cell types and get index labels
    ct_grouped = RNA_adata.obs.groupby("cell_type").size()
    df_ct_grouped = pd.DataFrame(ct_grouped, columns=["count"])
    df_ct_grouped = df_ct_grouped.reset_index()
    df_ct_grouped['label_id'] = df_ct_grouped.index

    # Merge label ids with obs
    RNA_adata.obs = RNA_adata.obs.reset_index().merge(df_ct_grouped, on='cell_type', how='inner').set_index('index')
    return np.array(RNA_adata.obs.label_id)