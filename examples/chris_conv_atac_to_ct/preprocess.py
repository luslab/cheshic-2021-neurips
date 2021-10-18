########### IMPORT LIBS ##########

import logging

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc

########### SETUP LIBS ##########

logging.basicConfig(level=logging.INFO)

########### MAIN ##########

logging.info('Preprocessing...')

# Init file paths
file_path_training = '/Users/cheshic/dev/test_data/neurips-2021/multiome/multiome_atac_processed_training.h5ad'

# Load data
sc_raw_training = sc.read(file_path_training)

logging.info('Generating training labels...')

# Find cell types and get index labels
ct_grouped = sc_raw_training.obs.groupby("cell_type").size()
df_ct_grouped = pd.DataFrame(ct_grouped, columns=["count"])
df_ct_grouped = df_ct_grouped.reset_index()
df_ct_grouped['label_id'] = df_ct_grouped.index

# Merge label ids with obs
sc_raw_training.obs = sc_raw_training.obs.reset_index().merge(df_ct_grouped, on='cell_type', how='inner').set_index('index')

# compute PCs
sc.tl.pca(sc_raw_training, svd_solver='arpack')


# top500 most variable genes 



logging.info('Saving data...')

sc_raw_training.write(filename='preprocessed.h5ad')

