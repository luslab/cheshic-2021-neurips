include { init_options; save_files; get_software_name } from '../functions'

params.options = [:]
def options    = init_options(params.options)

process TRAIN_ATAC_GRAPH {
    publishDir "${params.outdir}",
        mode: params.publish_dir_mode,
        saveAs: { filename -> save_files(filename:filename, options:params.options, publish_dir:get_software_name(task.process), publish_id:'') }

    container "luslab/neurips-pbg:cpu"
    
    input:
    path graph_path
    path data_dir
    path model_dir

    script:
    """
    train_atac_graph.py --graph_path $graph_path --data_dir $data_dir --model_dir $model_dir
    """
}