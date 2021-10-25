#!/usr/bin/env python3
# -*- coding: utf-8 -*-

########### IMPORT LIBS ##########

import logging
import argparse
from pathlib import Path

import torch

from torchbiggraph.config import parse_config, add_to_sys_path
from torchbiggraph.converters.importers import TSVEdgelistReader, convert_input_data
from torchbiggraph.util import (
    SubprocessInitializer,
    set_logging_verbosity,
    setup_logging,
)
from torchbiggraph.train import train

########### SETUP LIBS ##########

logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')

########### VARIABLES ##########

DATA_DIR = "data"
MODEL_DIR = "model"
GRAPH_PATH = "data/preprocessed.tsv"

########### CONFIG ##########

raw_config = dict(
    entity_path=DATA_DIR,
    edge_paths=[
        # graph data in HDF5 format will be saved here
        DATA_DIR + '/edges_partitioned',
    ],

    # trained embeddings as well as temporary files go here
    checkpoint_path=MODEL_DIR,

    # Graph structure
    entities={"all": {"num_partitions": 1}},
    relations=[
        {
            "name": "cell_peak",
            "lhs": "all",
            "rhs": "all",
            "operator": "complex_diagonal",
        }
    ],
    dynamic_relations=False,
    global_emb=False,

    comparator="dot",
    dimension=200,
    num_epochs=1000,
    num_uniform_negs=50,
    loss_fn="softmax",
    lr=0.1,
    eval_fraction=0.05,
)

########### MAIN ##########

def train_graph():
    parser = argparse.ArgumentParser()

    parser.add_argument('--graph_path')
    parser.add_argument('--model_dir')
    parser.add_argument('--data_dir')
    args = parser.parse_args()

    DATA_DIR = args.data_dir
    MODEL_DIR = args.model_dir
    GRAPH_PATH = args.graph_path


    logging.info('Initialising...')

    # Setup logging
    setup_logging()

    # Load config
    config = parse_config(raw_config)
 
    # Setup subprocess
    set_logging_verbosity(config.verbose)
    subprocess_init = SubprocessInitializer()
    subprocess_init.register(setup_logging, config.verbose)

    input_edge_paths = [Path(GRAPH_PATH)]

    convert_input_data(
        config.entities,
        config.relations,
        config.entity_path,
        config.edge_paths,
        input_edge_paths,
        TSVEdgelistReader(lhs_col=0, rhs_col=2, rel_col=1),
        dynamic_relations=config.dynamic_relations,
    )

    train(config, subprocess_init=subprocess_init)

    logging.info('Finished Training')

if __name__ == "__main__":
    train_graph()