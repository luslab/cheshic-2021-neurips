#!/usr/bin/env nextflow

nextflow.enable.dsl = 2

ch_training_data = file(params.training_data, checkIfExists: true)

def modules = params.modules.clone()

include { TRAIN_MODEL } from "./modules/train_model/main" addParams( options: modules['train_model'])

workflow {
    TRAIN_MODEL( ch_training_data )
}