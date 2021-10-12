########### IMPORT LIBS ##########

import anndata as ad
import numpy as np
import scipy

########### SETUP LIBS ##########

logging.basicConfig(level=logging.INFO)

## VIASH START
# Anything within this block will be removed by `viash` and will be
# replaced with the parameters as specified in your config.vsh.yaml.

########### INITIAL TEST PARAMS ##########
par = {
    'input_train_mod1': 'starter_kit/sample_data/openproblems_bmmc_multiome_starter/openproblems_bmmc_multiome_starter.train_mod1.h5ad',
    'input_train_mod2': 'starter_kit/sample_data/openproblems_bmmc_multiome_starter/openproblems_bmmc_multiome_starter.train_mod2.h5ad',
    'input_train_sol': 'starter_kit/sample_data/openproblems_bmmc_multiome_starter/openproblems_bmmc_multiome_starter.train_sol.h5ad',
    'input_test_mod1': 'starter_kit/sample_data/openproblems_bmmc_multiome_starter/openproblems_bmmc_multiome_starter.test_mod1.h5ad',
    'input_test_mod2': 'starter_kit/sample_data/openproblems_bmmc_multiome_starter/openproblems_bmmc_multiome_starter.test_mod2.h5ad',
    'output': 'output.h5ad'
}


## VIASH END

method_id = "crick_solution_1"

logging.info('Reading `h5ad` files...')
input_train_mod1 = ad.read_h5ad(par['input_train_mod1'])
input_train_mod2 = ad.read_h5ad(par['input_train_mod2'])
input_train_sol = ad.read_h5ad(par['input_train_sol'])
input_test_mod1 = ad.read_h5ad(par['input_test_mod1'])
input_test_mod2 = ad.read_h5ad(par['input_test_mod2'])