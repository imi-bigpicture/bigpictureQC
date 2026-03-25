"""
This module can collect a set of patches from image files and save it as an array.
"""

import digitalpathology.generator.batch.batchsource as dptbatchsource
import digitalpathology.utils.dataconfig as dptdataconfig
import digitalpathology.utils.filesynchronizer as dptfilesynchronizer
import digitalpathology.utils.imagefile as dptimagefile
import digitalpathology.utils.loggers as dptloggers
import digitalpathology.utils.trace as dpttrace

import argparse
import yaml
import logging
import datetime
import time
import os
import sys

#----------------------------------------------------------------------------------------------------

def extract_patch_collection(input_path,
                             mask_path,
                             output_path,
                             work_dir_path,
                             config_file_path,
                             patch_shapes,
                             mask_spacing,
                             patch_count,
                             label_ratios,
                             label_mode,
                             image_channels,
                             spacing_tolerance,
                             category_distribution,
                             purpose_id,
                             dimension_order,
                             label_check_level,
                             label_check_spacing,
                             copy_override,
                             path_override,
                             random_seed,
                             repetition_count,
                             cpu_enforce,
                             compress,
                             overwrite):
    """
    Extract a collection of patches from the given directory of batch source.

    Args:
        input_path (str): Input image file or directory path or data configuration file path.
        mask_path (str, None: Input mask file or directory filter expression.
        output_path (str): Output array file path.
        work_dir_path (str, None): Work directory path. Output is written here first before being copied to the target path.
        config_file_path (str, None): Configuration file path. If set all parameters that are in the config file are ignored.
        patch_shapes (dict, None): Image pixel spacing to shape of the loaded patches without the channels.
        mask_spacing (float, None): Mask spacing to use for selecting patch center coordinates.
        patch_count (int, None): Number of patches to load.
        label_ratios (dict, None): Mask label value to label ratio distribution.
        label_mode (str, None): Label generation mode, 'central', 'synthesize', or 'load'.
        image_channels (list, None): Desired channels that are extracted for each patch.
        spacing_tolerance (float, None): Pixel spacing tolerance (percentage).
        category_distribution (dict, None): Image category sampling distribution mapping from image category to ratio.
        purpose_id (str, None): Purpose to use from data configuration and from the parameter configuration files.
        dimension_order (str): Dimension order, 'BHWC' or 'BCHW'.
        label_check_level (int, None): Level to read the masks for extracting label information.
        label_check_spacing (float, None): Pixel spacing to read the masks for extracting label information (micrometer).
        copy_override (dict, None): Path override map, used with data configuration file as a source for copying.
        path_override (dict, None): Path override map, used with data configuration file.
        random_seed (int, None): Random seed for reproducible experiments.
        repetition_count (int): Repetition count. The whole patch extraction process is repeated.
        cpu_enforce (int, None): CPU count enforce.
        compress (bool): Compress the output .NPZ file.
        overwrite (bool): Overwrite existing target.
    """

    # Start time measurement.
    #
    start_time = time.time()

    # Get a logger.
    #
    logger = logging.getLogger(name=__name__)

    try:
        # Load the configuration parameters.
        #
        if config_file_path is not None:
            with open(file=config_file_path, mode='r') as config_file:
                config_parameters = yaml.load(stream=config_file, Loader=yaml.SafeLoader)
        else:
            config_parameters = None

        # Build BatchSource for the batch generator.
        #
        if os.path.splitext(input_path)[1].lower() in ['.json', '.yaml']:
            # Load the BatchSource object.
            #
            logger.info('Loading batch source: {path}'.format(path=input_path))

            batch_source = dptbatchsource.BatchSource(source_items=None)
            batch_source.load(file_path=input_path)

        else:
            # Create the BatchSource object.
            #
            logger.info('Building batch source from: {path}'.format(path=input_path))

            purpose_distribution = {purpose_id: 1} if purpose_id is not None else None
            batch_source = dptdataconfig.build_batch_source(image_path=input_path,
                                                            mask_path=mask_path,
                                                            stat_path=None,
                                                            labels=list(label_ratios.keys()),
                                                            read_level=label_check_level,
                                                            read_spacing=label_check_spacing,
                                                            purpose_distribution=purpose_distribution,
                                                            mask_spacing_tolerance=spacing_tolerance,
                                                            random_item_order=False)

        # Copy the items if configured.
        #
        if copy_override is not None:
            # Copy the batch source.
            #
            logger.info('Copying source items...')

            # Use the categories from the given parameter configuration file. Otherwise from the argument. Finally, just copy everything if none is given.
            #
            if config_parameters is not None:
                category_list = list(config_parameters['data']['categories'].keys())
            elif category_distribution is not None:
                category_list = list(category_distribution.keys())
            else:
                category_list = None

            copied_items, skipped_items = dptimagefile.copy_batch_source(batch_source=batch_source,
                                                                         source_replacements=copy_override,
                                                                         target_replacements=path_override,
                                                                         purposes=purpose_id,
                                                                         categories=category_list,
                                                                         allow_missing_stat=True,
                                                                         overwrite=overwrite)

            # Report item counts.
            #
            logger.info('Copied items: {count}'.format(count=copied_items))
            logger.info('Skipped items: {count}'.format(count=skipped_items))

        # Update the batch source to the final paths.
        #
        if path_override:
            batch_source.update(path_replacements=path_override)

        # Create work directory.
        #
        if work_dir_path:
            os.makedirs(work_dir_path, exist_ok=True)

        # Initialize file synchronizer.
        #
        file_sync = dptfilesynchronizer.FileSynchronizer(work_directory=work_dir_path)

        output_path_list = []
        actual_repetition_count = config_parameters['training']['iterations'][purpose_id]['repetition count'] if config_parameters is not None else repetition_count
        for repetition in range(actual_repetition_count):
            target_path = output_path.format(index=repetition)
            output_path_list.append(target_path)

            file_sync.add(target_path=target_path)

        # Save the batches.
        #
        if config_file_path is not None:
            # Configure the batch generator with parameter file.
            #
            dptdataconfig.save_config_of_patches(output_path=output_path_list,
                                                 batch_source=batch_source,
                                                 parameters=config_parameters,
                                                 dimension_order=dimension_order,
                                                 purpose=purpose_id,
                                                 cpu_count_enforce=cpu_enforce,
                                                 random_seed=random_seed,
                                                 compress=compress,
                                                 overwrite=overwrite,
                                                 file_sync=file_sync)

        else:
            # Configure the batch generator with command line arguments.
            #
            dptdataconfig.save_batch_of_patches(output_path=output_path_list,
                                                batch_source=batch_source,
                                                patch_shapes=patch_shapes,
                                                mask_spacing=mask_spacing,
                                                patch_count=patch_count,
                                                label_dist=label_ratios,
                                                label_mode=label_mode,
                                                image_channels=image_channels,
                                                dimension_order=dimension_order,
                                                purpose=purpose_id,
                                                category_dist=category_distribution,
                                                spacing_tolerance=spacing_tolerance,
                                                random_seed=random_seed,
                                                compress=compress,
                                                overwrite=overwrite,
                                                file_sync=file_sync)
        # Log execution time.
        #
        execution_time = time.time() - start_time
        logger.info('Done in {delta}'.format(delta=datetime.timedelta(seconds=execution_time)))

    except Exception as exception:
        # Collect and summarize traceback information.
        #
        _, _, exception_traceback = sys.exc_info()
        trace_string = dpttrace.format_traceback(traceback_object=exception_traceback)

        # Log the exception.
        #
        logger.info('Exception raised: "{ex}"'.format(ex=exception))
        logger.error('Exception: "{ex}"; trace: "{trace}"'.format(ex=exception, trace=trace_string))

        # Re-raise the exception.
        #
        raise

#----------------------------------------------------------------------------------------------------

def collect_arguments():
    """
    Collect command line arguments.

    Returns:
        (str, str,  str, str, str, dict, float, int, dict, str, list, float, int, dict, str, str, int, float, dict, dict, str, int,  int, bool, bool): The parsed command line arguments:
            input image file or directory expression of data configuration file path, input mask file or directory expression, output array path, work directory path, configuration file
            path, image spacing to patch shape map, patch count, label ratios, label mode, channel list, pixel spacing tolerance, repetition count, category distribution, purpose to use
            from a distributed data configuration file, dimension order, label content checking level, label content checking pixel spacing, copy configuration path override, data
            configuration path override, log file path, random seed, enforced CPU count, compress flag, and the overwrite flag.
    """

    # Configure argument parser.
    #
    argument_parser = argparse.ArgumentParser(description='Train network for whole slide image classification.',
                                              epilog='The input can be a data configuration file, an image filter expression or a single image path. In case of an input filter expression '
                                                     'image and mask file matching is based on file names. Use {image} replacement for image file base names in mask file name specification '
                                                     'for filtering and matching and {index} replacement for repetition index in output file name specification. If the configuration file is'
                                                     'used all command line arguments are overwritten with the values found in it.')

    argument_parser.add_argument('-i', '--input',  required=True,  type=str,               help='config, or image file or directory path to process')
    argument_parser.add_argument('-m', '--mask',   required=False, type=str, default=None, help='mask file or directory path to use')
    argument_parser.add_argument('-f', '--output', required=True,  type=str,               help='output array path')
    argument_parser.add_argument('-q', '--work',   required=False, type=str, default=None, help='work directory')

    check_group = argument_parser.add_mutually_exclusive_group(required=True)
    check_group.add_argument('-c', '--config', required=False, type=str,  default=None, help='configuration file path')
    check_group.add_argument('-a', '--shapes', required=False, type=str,  default=None, help='image spacing to patch shape map')

    argument_parser.add_argument('-n', '--mask_spacing', required=False, type=float, default=0.0,         help='mask pixel spacing (micrometer)')
    argument_parser.add_argument('-z', '--count',        required=False, type=int,   default=0,           help='number of patches ot extract')
    argument_parser.add_argument('-r', '--ratios',       required=False, type=str,   default='{}',        help='label value to label ratio map')
    argument_parser.add_argument('-d', '--mode',         required=False, type=str,   default='central',   help='label generation mode: \'central\', \'load\' or \'synthesize\'')
    argument_parser.add_argument('-x', '--channels',     required=False, type=str,   default='(0, 1, 2)', help='list of input channel indices')
    argument_parser.add_argument('-t', '--tolerance',    required=False, type=float, default=0.25,        help='pixel spacing tolerance (percentage)')
    argument_parser.add_argument('-v', '--repetition',   required=False, type=int,   default=1,           help='repetition count')
    argument_parser.add_argument('-e', '--categories',   required=False, type=str,   default=None,        help='category distribution for data configuration file, uniform by default')
    argument_parser.add_argument('-p', '--purpose',      required=False, type=str,   default=None,        help='purpose to use from a distributed data configuration file')

    argument_parser.add_argument('-l', '--dimension_order', required=False, type=str, default='BHWC', choices=['BHWC', 'BCHW'], help='dimension order')

    check_group = argument_parser.add_mutually_exclusive_group(required=False)
    check_group.add_argument('-k', '--check_level',   required=False, type=int,   default=None, help='mask label content checking level for directory processing')
    check_group.add_argument('-g', '--check_spacing', required=False, type=float, default=None, help='mask label content checking pixel spacing (micrometer) for directory processing')

    argument_parser.add_argument('-y', '--copy_from',  required=False, type=str, default=None, help='data copy source for data config file processing')
    argument_parser.add_argument('-o', '--override',   required=False, type=str, default=None, help='data source overrides')
    argument_parser.add_argument('-b', '--log',        required=False, type=str, default=None, help='lof file path')
    argument_parser.add_argument('-s', '--seed',       required=False, type=int, default=None, help='random seed')
    argument_parser.add_argument('-u', '--cpu',        required=False, type=int, default=None, help='enforced CPU count')
    argument_parser.add_argument('-j', '--compress',   action='store_true',                    help='compress the output NPZ file.')
    argument_parser.add_argument('-w', '--overwrite',  action='store_true',                    help='overwrite existing results')

    # Parse arguments.
    #
    arguments = vars(argument_parser.parse_args())

    # Collect arguments.
    #
    parsed_input_path = arguments['input']
    parsed_mask_path = arguments['mask']
    parsed_output_path = arguments['output']
    parsed_work_path = arguments['work']
    parsed_config_path = arguments['config']
    parsed_shapes_str = arguments['shapes']
    parsed_mask_spacing = arguments['mask_spacing']
    parsed_count = arguments['count']
    parsed_ratios_str = arguments['ratios']
    parsed_mode = arguments['mode']
    parsed_channels_str = arguments['channels']
    parsed_spacing_tolerance = arguments['tolerance']
    parsed_repetition_count = arguments['repetition']
    parsed_categories_str = arguments['categories']
    parsed_purpose = arguments['purpose']
    parsed_dimension_order = arguments['dimension_order']
    parsed_check_level = arguments['check_level']
    parsed_check_spacing = arguments['check_spacing']
    parsed_copy_override_str = arguments['copy_from']
    parsed_data_override_str = arguments['override']
    parsed_log_path = arguments['log']
    parsed_random_seed = arguments['seed']
    parsed_cpu_count = arguments['cpu']
    parsed_compress = arguments['compress']
    parsed_overwrite = arguments['overwrite']

    # Evaluate expressions
    #
    parsed_shapes = eval(parsed_shapes_str) if parsed_shapes_str else None
    parsed_ratios = eval(parsed_ratios_str)
    parsed_channels = eval(parsed_channels_str)
    parsed_categories = eval(parsed_categories_str) if parsed_categories_str else None
    parsed_copy_override = eval(parsed_copy_override_str) if parsed_copy_override_str else None
    parsed_data_override = eval(parsed_data_override_str) if parsed_data_override_str else None

    # Print parameters.
    #
    print(argument_parser.description)
    print('Input path: {path}'.format(path=parsed_input_path))
    print('Mask path: {path}'.format(path=parsed_mask_path))
    print('Output path: {path}'.format(path=parsed_output_path))
    print('Work directory path: {path}'.format(path=parsed_work_path))

    if parsed_config_path is not None:
        print('Config file path: {path}'.format(path=parsed_config_path))
    else:
        print('Patch shapes: {shapes}'.format(shapes=parsed_shapes))
        print('Mask pixel spacing: {spacing} um'.format(spacing=parsed_mask_spacing))
        print('Patch count: {count}'.format(count=parsed_count))
        print('Label ratios: {ratios}'.format(ratios=parsed_ratios))
        print('Label mode: {mode}'.format(mode=parsed_mode))
        print('Input channels: {channels}'.format(channels=parsed_channels))
        print('Pixel spacing tolerance: {tolerance}'.format(tolerance=parsed_spacing_tolerance))
        print('Repetition count: {repetitions}'.format(repetitions=parsed_repetition_count))
        print('Category distribution in data config: {categories}'.format(categories=parsed_categories))

    print('Purpose in data config: {purposes}'.format(purposes=parsed_purpose))
    print('Dimension order: {order}'.format(order=parsed_dimension_order))

    if parsed_check_level is not None:
        print('Label content checking level: {level}'.format(level=parsed_check_level))
    else:
        print('Label content checking pixel spacing: {spacing} um'.format(spacing=parsed_check_spacing))

    print('Copy from override in data config: {override}'.format(override=parsed_copy_override))
    print('Path override in data config: {override}'.format(override=parsed_data_override))
    print('Log path: {path}'.format(path=parsed_log_path))
    print('Random seed: {seed}'.format(seed=parsed_random_seed))
    print('Enforced CPU count: {cpus}'.format(cpus=parsed_cpu_count))
    print('Compress results: {flag}'.format(flag=parsed_compress))
    print('Overwrite existing results: {flag}'.format(flag=parsed_overwrite))

    # Return parsed values.
    #
    return (parsed_input_path,
            parsed_mask_path,
            parsed_output_path,
            parsed_work_path,
            parsed_config_path,
            parsed_shapes,
            parsed_mask_spacing,
            parsed_count,
            parsed_ratios,
            parsed_mode,
            parsed_channels,
            parsed_spacing_tolerance,
            parsed_repetition_count,
            parsed_categories,
            parsed_purpose,
            parsed_dimension_order,
            parsed_check_level,
            parsed_check_spacing,
            parsed_copy_override,
            parsed_data_override,
            parsed_log_path,
            parsed_random_seed,
            parsed_cpu_count,
            parsed_compress,
            parsed_overwrite)

#----------------------------------------------------------------------------------------------------

def main():
    """Main function."""

    # Parse parameters.
    #
    (input_path_exp,
     mask_path_exp,
     output_file_path,
     work_directory_path,
     config_file_path,
     shapes,
     mask_pixel_spacing,
     count,
     ratios,
     mode,
     channels,
     tolerance,
     repetitions,
     categories,
     purpose_id,
     dimension_order,
     check_level,
     check_spacing,
     copy_override,
     data_override,
     log_file_path,
     seed,
     cpu_count,
     compress_flag,
     overwrite_flag) = collect_arguments()

    print('')
    print('Executing...')

    # Init the logger to print to the console.
    #
    dptloggers.init_file_logger(log_path=log_file_path, debug=False)

    # Extract patch collection to patch array file.
    #
    extract_patch_collection(input_path=input_path_exp,
                             mask_path=mask_path_exp,
                             output_path=output_file_path,
                             work_dir_path=work_directory_path,
                             config_file_path=config_file_path,
                             patch_shapes=shapes,
                             mask_spacing=mask_pixel_spacing,
                             patch_count=count,
                             label_ratios=ratios,
                             label_mode=mode,
                             image_channels=channels,
                             spacing_tolerance=tolerance,
                             category_distribution=categories,
                             purpose_id=purpose_id,
                             dimension_order=dimension_order,
                             label_check_level=check_level,
                             label_check_spacing=check_spacing,
                             copy_override=copy_override,
                             path_override=data_override,
                             random_seed=seed,
                             repetition_count=repetitions,
                             cpu_enforce=cpu_count,
                             compress=compress_flag,
                             overwrite=overwrite_flag)

    print('Exiting')

#----------------------------------------------------------------------------------------------------


if __name__ == '__main__':

    main()
