/*
 * -----------------------------------------------------
 *  Utility functions used in DSL2 module files
 * -----------------------------------------------------
 */

/*
 * Extract name of software tool from process name using $task.process
 */
def get_software_name(task_process) {
    return task_process.tokenize(':')[-1].tokenize('_')[0].toLowerCase()
}

/*
 * Function to initialise default values and to generate a Groovy Map of available options for nf-core modules
 */
def init_options(Map args) {
    def Map options = [:]
    options.args          = args.args ?: ''
    options.args2         = args.args2 ?: ''
    options.publish_by_id = args.publish_by_id ?: false
    options.publish_dir   = args.publish_dir ?: ''
    options.publish_files = args.publish_files
    options.suffix        = args.suffix ?: ''
    options.ext           = args.ext ?: ''
    return options << args
}

/*
 * Tidy up and join elements of a list to return a path string
 */
def get_path_from_list(path_list) {
    def paths = path_list.findAll { item -> !item?.trim().isEmpty() }  // Remove empty entries
    paths = paths.collect { it.trim().replaceAll("^[/]+|[/]+\$", '') } // Trim whitespace and trailing slashes
    return paths.join('/')
}

/*
 * Function to save/publish module results
 */
def save_files(Map args) {
    if (!args.filename.endsWith('.version.txt')) {
        def ioptions = init_options(args.options)
        def path_list = [ ioptions.publish_dir ?: args.publish_dir ]
        if (ioptions.publish_by_id) {
            path_list.add(args.publish_id)
        }
        if (ioptions.publish_files instanceof Map) {
            for (ext in ioptions.publish_files) {
                if (args.filename.endsWith(ext.key)) {
                    def ext_list = path_list.collect()
                    ext_list.add(ext.value)
                    return "${get_path_from_list(ext_list)}/$args.filename"
                }
            }
        } else if (ioptions.publish_files == null) {
            return "${get_path_from_list(path_list)}/$args.filename"
        }
    }
}
