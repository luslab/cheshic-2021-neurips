include { init_options; save_files; get_software_name } from '../functions'

params.options = [:]
def options    = init_options(params.options)

process TRAIN_BASIC_MODEL {
    publishDir "${params.outdir}",
        mode: params.publish_dir_mode,
        saveAs: { filename -> save_files(filename:filename, options:params.options, publish_dir:get_software_name(task.process), publish_id:'') }

    container "luslab/neurips:nvidia"
    
    input:
    path training_data

    script:
    """
    train_model.py --training_data $training_data
    """
}