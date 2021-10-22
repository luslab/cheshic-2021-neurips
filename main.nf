#!/usr/bin/env nextflow

nextflow.enable.dsl = 2

//ch_training_data = file(params.training_data)
ch_graph_data = file(params.graph_data)
ch_model_dir = file(params.model_dir)
ch_data_dir = file(params.data_dir)

def modules = params.modules.clone()

// include { TRAIN_MODEL } from "./modules/train_model/main" addParams( options: modules['train_model'])

include { TRAIN_ATAC_GRAPH } from "./modules/train_atac_graph/main" addParams( options: modules['train_model'])

workflow {
    TRAIN_ATAC_GRAPH( ch_graph_data, ch_data_dir, ch_model_dir )
}