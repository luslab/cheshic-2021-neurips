
#!/usr/bin/env python3

DATA_DIR = "cluster/data"
MODEL_DIR = "cluster/model"
GRAPH_PATH = "cluster/data/preprocessed.tsv"

def get_torchbiggraph_config():

    config = dict(
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
    num_epochs=100,
    num_uniform_negs=50,
    loss_fn="softmax",
    lr=0.1,
    eval_fraction=0.05,
)


    return config