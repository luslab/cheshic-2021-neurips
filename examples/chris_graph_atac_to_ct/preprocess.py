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


# Convert to iterable format
matrix = sc_raw_training.X.tocoo()
logging.info("Data shape: " + str(matrix.shape))

# Open output file
writer = open("data/preprocessed.tsv", "w")

count = 0

# Convert to tsv edge file
logging.info('Writing edges...')
for i,j,v in zip(matrix.row, matrix.col, matrix.data):

    line = str(i) + "\tcell_peak\t" + str(j) + "\n"
    writer.write(line)

    # count = count + 1
    # if count > 10000:
    #     break

writer.close()

logging.info('Preprocessing complete')
