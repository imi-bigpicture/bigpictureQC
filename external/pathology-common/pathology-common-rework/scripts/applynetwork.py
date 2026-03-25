"""
This module can load a network model and apply classification on a whole slide image.
"""

import digitalpathology.image.processing.inference as dptinference
import digitalpathology.generator.batch.batchsource as dptbatchsource
import digitalpathology.utils.foldercontent as dptfoldercontent
import digitalpathology.utils.loggers as dptloggers

import argparse
import os
import sys

#----------------------------------------------------------------------------------------------------

def assemble_jobs(image_path, image_filter_regexp, mask_path, model_path, purpose_filter, category_filter, path_override, output_path, no_mask_from_config):
    """
    Assemble (source image path, source mask path, target output path) job triplets for network application.

    Args:
        image_path (str): Path of the image to classify or data configuration YAML/JSON file.
        image_filter_regexp (str, None): Image filter regular expression.
        mask_path (str): Path of the mask image to use.
        model_path (str): Path of the model file to load.
        purpose_filter (list, None): Purpose filter list. Used if the input is a data configuration JSON/YAML in distributed mode.
        category_filter (list, None): Category filter list. Used if the input is a data configuration JSON/YAML in distributed mode.
        path_override (map, None): Path overrides for  data configuration JSON/YAML.
        output_path (str): Path of the result image.
        no_mask_from_config (bool): Do not use the mask entries from data config file.

    Returns:
        list: List of job tuples.

    Raises:
        InvalidDataFileExtensionError: The format cannot be derived from the file extension.
        InvalidDataSourceTypeError: Invalid JSON or YAML file.
        PurposeListAndRatioMismatchError: The configured purpose distribution does not match the available purposes.
    """

    # Find out operation mode. If all paths are file paths the mode is 'file'.
    #
    data_config = os.path.splitext(image_path)[1].lower() in ['.json', '.yaml']
    model_base = os.path.splitext(os.path.basename(model_path))[0]
    result_job_list = []
    if os.path.isfile(image_path) and not data_config:
        # Return a single triplet if the paths were existing files.
        #
        image_base = os.path.splitext(os.path.basename(image_path))[0]
        target_output_path = output_path.format(image=image_base, model=model_base)

        # Add job item to the list.
        #
        job_item = (image_path, mask_path, target_output_path)
        result_job_list.append(job_item)
    else:
        # Check if the input is an expression or a data configuration file.
        #
        if data_config:
            # Collect files from the config file.
            #
            batch_source = dptbatchsource.BatchSource(source_items=None)
            batch_source.load(file_path=image_path)
            if path_override:
                batch_source.update(path_replacements=path_override)

            target_purposes = [purpose_id for purpose_id in batch_source.purposes() if purpose_id in purpose_filter] if purpose_filter else None
            available_categories = batch_source.categories(purpose_id=target_purposes)
            target_categories = [category_id for category_id in available_categories if category_id in category_filter] if category_filter else None

            # Go through the filtered purposes and collect all images.
            #
            for source_item in batch_source.items(purpose_id=target_purposes, category_id=target_categories, replace=True):
                image_item_path = source_item.image
                mask_item_path = source_item.mask
                image_key = os.path.splitext(os.path.basename(image_item_path))[0]
                mask_key = os.path.splitext(os.path.basename(mask_item_path))[0]

                current_mask_path = mask_item_path if not no_mask_from_config else None
                target_output_path = output_path.format(image=image_key, mask=mask_key, model=model_base)

                # Only add items with complete image-mask pairs if the mask path is set.
                #
                if no_mask_from_config or current_mask_path is not None:
                    # Add job item to the list.
                    #
                    job_item = (image_item_path, current_mask_path, target_output_path)
                    result_job_list.append(job_item)

            # Print image count for checking.
            #
            print('Found image count: {match_count}'.format(match_count=len(result_job_list)))

        else:
            # Replace the image matching string in the mask path to an asterisk to be able to collect all possible mask files.
            #
            mask_wildcard_path = mask_path.format(image='*') if mask_path else ''

            # Collect all source images and masks and match their file name for processing.
            #
            image_file_path_list = dptfoldercontent.folder_content(folder_path=image_path, filter_exp=image_filter_regexp, recursive=False)
            mask_file_path_list = dptfoldercontent.folder_content(folder_path=mask_wildcard_path, filter_exp=None, recursive=False)

            # Build file name to path maps.
            #
            image_file_path_map = {os.path.splitext(os.path.basename(image_path_item))[0]: image_path_item for image_path_item in image_file_path_list}
            mask_file_path_map = {os.path.splitext(os.path.basename(mask_path_item))[0]: mask_path_item for mask_path_item in mask_file_path_list}

            # Construct image match expression.
            #
            if mask_path:
                mask_match_base = '{image}' if os.path.isdir(mask_path) else os.path.splitext(os.path.basename(mask_path))[0]
                mask_match = mask_match_base if 0 <= mask_match_base.find('{image}') else '{image}'
            else:
                mask_match = ''

            # Assemble list.
            #
            for image_key in image_file_path_map:
                mask_key = mask_match.format(image=image_key)
                mask_replace = mask_key if mask_key in mask_file_path_map else ''
                target_output_path = output_path.format(image=image_key, mask=mask_replace, model=model_base)

                current_image_path = image_file_path_map[image_key]
                current_mask_path = mask_file_path_map.get(mask_key, None)

                # Only add items with complete image-mask pairs if the mask path is set.
                #
                if mask_path is None or current_mask_path is not None:
                    # Add job item to the list.
                    #
                    job_item = (current_image_path, current_mask_path, target_output_path)
                    result_job_list.append(job_item)

            # Print match count for checking.
            #
            print('Matching image count: {match_count}'.format(match_count=len(result_job_list)))

    # Return the result list.
    #
    return result_job_list

#----------------------------------------------------------------------------------------------------

def collect_arguments():
    """
    Collect command line arguments.

    Returns:
        (str, str, float, str, str, float, str, int, int, int, list, str, int, float, float, float, float, float, int, list, list, dict, str, str, str, list, list, bool, bool, bool, bool,
            bool, bool, bool, bool, bool): The parsed command line arguments: input image path, input file filter regular expression, input image pixel spacing, mask image path, output image,
            output image pixel spacing, model file path, processing patch size, output class, number of output classes, input channel indices, padding mode, padding mode constant, network
            confidence, region diagonal threshold, dilation distance, hole diagonal threshold, pixel spacing tolerance, interpolation order, purpose list for data config files, category list
            for data config files, path override map for data config files, target directory path for data copy, work directory path, normalizer type, normalizer input interval, normalizer
            output interval, full connectivity matrix flag, unrestrict network flag, test time augmentation, soft classification mode flag, quantization flag, no masks from data config files
            flag, keep intermediate files flag, keep copied files flag, and the overwrite flag.
    """

    # Prepare argument value choices.
    #
    pm_choices = ['constant', 'reflect']
    nr_choices = ['general', 'rgb', 'rgb_to_0-1']

    # Configure argument parser.
    #
    argument_parser = argparse.ArgumentParser(description='Apply segmentation on mr-image using the given network model.',
                                              epilog='It can either work with file paths, file filter expressions or a data configuration JSON/YAML file. If filter expressions are set '
                                                     'the matching is based on file names. Use {image} and {mask} as replacement for image and mask file base names in output and mask '
                                                     'file name specification and image file to mask file matching. Additionally, {model} can be used for model for replacement for model '
                                                     'file base name in output file specifications. With distributed JSON/YAML files the purpose can be filtered with the optional purpose '
                                                     'option.')

    argument_parser.add_argument('-i',  '--input',                     required=True,  type=str,                                             help='image file or directory path to process')
    argument_parser.add_argument('-if', '--input_filter',              required=False, type=str,   default=None,                             help='image file filter regular expression')
    argument_parser.add_argument('-l',  '--input_spacing',             required=True,  type=float,                                           help='input image pixel spacing (micrometer)')
    argument_parser.add_argument('-m',  '--mask',                      required=False, type=str,   default=None,                             help='mask file or directory path to use')
    argument_parser.add_argument('-o',  '--output',                    required=True,  type=str,                                             help='output file or directory path')
    argument_parser.add_argument('-g',  '--output_spacing',            required=False, type=float, default=None,                             help='target output pixel spacing (micrometer)')
    argument_parser.add_argument('-n',  '--model',                     required=True,  type=str,                                             help='network model')
    argument_parser.add_argument('-p',  '--patch_size',                required=False, type=int,   default=1024,                             help='processing patch size')
    argument_parser.add_argument('-c',  '--output_class',              required=False, type=int,   default=-1,                               help='output class')
    argument_parser.add_argument('-u',  '--num_classes',               required=False, type=int,   default=-1,                               help='number of output classes of the network')
    argument_parser.add_argument('-a',  '--channels',                  required=False, type=str,   default='(0, 1, 2)',                      help='list of input channel indices')
    argument_parser.add_argument('-pm', '--padding_mode',              required=False, type=str,   default='constant', choices=pm_choices,   help='padding mode')
    argument_parser.add_argument('-pc', '--padding_constant',          required=False, type=int,   default=255,                              help='padding constant to use with \'constant mode\'')
    argument_parser.add_argument('-f',  '--confidence',                required=False, type=float, default=0.0,                              help='network confidence for thresholding')
    argument_parser.add_argument('-b',  '--region_diagonal_threshold', required=False, type=float, default=0.0,                              help='region size filter (micrometer)')
    argument_parser.add_argument('-dd', '--dilation_distance',         required=False, type=float, default=0.0,                              help='dilation distance (micrometer)')
    argument_parser.add_argument('-rt', '--hole_diagonal_threshold',   required=False, type=float, default=0.0,                              help='region size filter (micrometer)')
    argument_parser.add_argument('-ht', '--tolerance',                 required=False, type=float, default=0.25,                             help='pixel spacing tolerance (percentage)')
    argument_parser.add_argument('-z',  '--order',                     required=False, type=int,   default=0,                                help='interpolation order')
    argument_parser.add_argument('-r',  '--purposes',                  required=False, type=str,   default=None,                             help='purpose identifiers of data configuration file')
    argument_parser.add_argument('-e',  '--categories',                required=False, type=str,   default=None,                             help='category identifiers of data configuration file')
    argument_parser.add_argument('-d',  '--input_override',            required=False, type=str,   default=None,                             help='data config source overrides')
    argument_parser.add_argument('-cp', '--copy_directory',            required=False, type=str,   default=None,                             help='data copy target directory path')
    argument_parser.add_argument('-wp', '--work_directory',            required=False, type=str,   default=None,                             help='work directory path')
    argument_parser.add_argument('-nr', '--normalizer',                required=False, type=str,   default='rgb_to_0-1', choices=nr_choices, help='normalizer to use for preprocessing')
    argument_parser.add_argument('-tr', '--source_range',              required=False, type=str,   default='[]',                             help='source-range for normalizer')
    argument_parser.add_argument('-sr', '--target_range',              required=False, type=str,   default='[]',                             help='target-range for normalizer')
    argument_parser.add_argument('-fc', '--full_connectivity',         action='store_true',                                                  help='full connectivity matrix for region filtering')
    argument_parser.add_argument('-un', '--unrestrict_network',        action='store_true',                                                  help='unrestrict network for fixed inputs and outputs')
    argument_parser.add_argument('-ta', '--augment',                   action='store_true',                                                  help='enable test time augmentation')
    argument_parser.add_argument('-s',  '--soft',                      action='store_true',                                                  help='get soft classification from the network')
    argument_parser.add_argument('-q',  '--quantize',                  action='store_true',                                                  help='quantize result to [0, 255]')
    argument_parser.add_argument('-x',  '--no_mask',                   action='store_true',                                                  help='do not use mask from the data configuration file')
    argument_parser.add_argument('-ki', '--keep_intermediates',        action='store_true',                                                  help='keep intermediate files')
    argument_parser.add_argument('-kc', '--keep_copies',               action='store_true',                                                  help='keep copied image files')
    argument_parser.add_argument('-w',  '--overwrite',                 action='store_true',                                                  help='overwrite existing results')

    # Parse arguments.
    #
    arguments = vars(argument_parser.parse_args())

    # Collect arguments.
    #
    parsed_image_path = arguments['input']
    parsed_image_filter_regexp = arguments['input_filter']
    parsed_image_spacing = arguments['input_spacing']
    parsed_mask_path = arguments['mask']
    parsed_output_path = arguments['output']
    parsed_output_spacing = arguments['output_spacing']
    parsed_model_path = arguments['model']
    parsed_patch_size = arguments['patch_size']
    parsed_output_class = arguments['output_class']
    parsed_num_class = arguments['num_classes']
    parsed_channels_str = arguments['channels']
    parsed_padding_mode = arguments['padding_mode']
    parsed_padding_constant = arguments['padding_constant']
    parsed_confidence = arguments['confidence']
    parsed_region_diagonal_threshold = arguments['region_diagonal_threshold']
    parsed_dilation_distance = arguments['dilation_distance']
    parsed_hole_diagonal_threshold = arguments['hole_diagonal_threshold']
    parsed_spacing_tolerance = arguments['tolerance']
    parsed_order = arguments['order']
    parsed_purposes_str = arguments['purposes']
    parsed_categories_str = arguments['categories']
    parsed_input_override_str = arguments['input_override']
    parsed_copy_directory = arguments['copy_directory']
    parsed_work_directory = arguments['work_directory']
    parsed_normalizer = arguments['normalizer']
    parsed_source_range_str = arguments['source_range']
    parsed_target_range_str = arguments['target_range']
    parsed_full_connectivity = arguments['full_connectivity']
    parsed_unrestrict_network = arguments['unrestrict_network']
    parsed_augmentation = arguments['augment']
    parsed_soft_mode = arguments['soft']
    parsed_quantize = arguments['quantize']
    parsed_no_mask = arguments['no_mask']
    parsed_keep_intermediates = arguments['keep_intermediates']
    parsed_keep_copies = arguments['keep_copies']
    parsed_overwrite = arguments['overwrite']

    # Evaluate expressions.
    #
    parsed_channels = eval(parsed_channels_str)
    parsed_purposes = eval(parsed_purposes_str) if parsed_purposes_str else None
    parsed_categories = eval(parsed_categories_str) if parsed_categories_str else None
    parsed_input_override = eval(parsed_input_override_str) if parsed_input_override_str else None
    parsed_source_range = eval(parsed_source_range_str)
    parsed_target_range = eval(parsed_target_range_str)

    # Print parameters.
    #
    print(argument_parser.description)
    print('Image path: {path}'.format(path=parsed_image_path))
    print('Image filter: {regexp}'.format(regexp=parsed_image_filter_regexp))
    print('Image pixel spacing: {spacing} um'.format(spacing=parsed_image_spacing))
    print('Pixel spacing tolerance: {tolerance}'.format(tolerance=parsed_spacing_tolerance))
    print('Mask path: {path}'.format(path=parsed_mask_path))
    print('Output path: {path}'.format(path=parsed_output_path))
    print('Output spacing: {spacing} um'.format(spacing=parsed_output_spacing))
    print('Model path: {path}'.format(path=parsed_model_path))
    print('Patch size: {size}'.format(size=parsed_patch_size))
    print('Output class: {index}'.format(index=(parsed_output_class if 0 <= parsed_output_class else 'all')))
    print('Number of classes: {count}'.format(count=parsed_num_class))
    print('Input channels: {channels}'.format(channels=parsed_channels))
    print('Padding mode: {mode}'.format(mode=parsed_padding_mode))
    print('Padding constant: {value}'.format(value=parsed_padding_constant))
    print('Network confidence: {confidence}'.format(confidence=parsed_confidence))
    print('Region diagonal threshold: {threshold} um'.format(threshold=parsed_region_diagonal_threshold))
    print('Dilation distance: {distance} um'.format(distance=parsed_dilation_distance))
    print('Hole diagonal threshold: {threshold} um'.format(threshold=parsed_hole_diagonal_threshold))
    print('Interpolation order: {order}'.format(order=parsed_order))
    print('Purposes in data config: {purposes}'.format(purposes=parsed_purposes))
    print('Categories in data config: {categories}'.format(categories=parsed_categories))
    print('Path override in data config: {override}'.format(override=parsed_input_override))
    print('Copy directory path: {path}'.format(path=parsed_copy_directory))
    print('Work directory path: {path}'.format(path=parsed_work_directory))
    print('Normalizer: {normalizer}'.format(normalizer=parsed_normalizer))
    print('Target-range normalizer: {values}'.format(values=parsed_target_range))
    print('Source-range normalizer: {values}'.format(values=parsed_source_range))
    print('Full connectivity matrix: {flag}'.format(flag=parsed_full_connectivity))
    print('Unrestrict network: {flag}'.format(flag=parsed_unrestrict_network))
    print('Test time augmentation: {flag}'.format(flag=parsed_augmentation))
    print('Soft classification: {flag}'.format(flag=parsed_soft_mode))
    print('Quantize: {flag}'.format(flag=parsed_quantize))
    print('Skip masks in data config: {flag}'.format(flag=parsed_no_mask))
    print('Keep intermediate files: {flag}'.format(flag=parsed_keep_intermediates))
    print('Keep copied files: {flag}'.format(flag=parsed_keep_copies))
    print('Overwrite existing results: {flag}'.format(flag=parsed_overwrite))

    # Return parsed values.
    #
    return (parsed_image_path,
            parsed_image_filter_regexp,
            parsed_image_spacing,
            parsed_mask_path,
            parsed_output_path,
            parsed_output_spacing,
            parsed_model_path,
            parsed_patch_size,
            parsed_output_class,
            parsed_num_class,
            parsed_channels,
            parsed_padding_mode,
            parsed_padding_constant,
            parsed_confidence,
            parsed_region_diagonal_threshold,
            parsed_dilation_distance,
            parsed_hole_diagonal_threshold,
            parsed_spacing_tolerance,
            parsed_order,
            parsed_purposes,
            parsed_categories,
            parsed_input_override,
            parsed_copy_directory,
            parsed_work_directory,
            parsed_normalizer,
            parsed_source_range,
            parsed_target_range,
            parsed_full_connectivity,
            parsed_unrestrict_network,
            parsed_augmentation,
            parsed_soft_mode,
            parsed_quantize,
            parsed_no_mask,
            parsed_keep_intermediates,
            parsed_keep_copies,
            parsed_overwrite)

#----------------------------------------------------------------------------------------------------

def main():
    """
    Main function.

    Returns:
        int: Error code.
    """

    # Collect command line arguments.
    #
    (image_path,
     image_filter_regexp,
     input_spacing,
     mask_path,
     output_path,
     output_spacing,
     network_model_path,
     processing_patch_size,
     network_output_class,
     network_num_classes,
     input_channels,
     padding_mode,
     padding_constant,
     network_confidence,
     region_diagonal_threshold,
     dilation_distance,
     hole_diagonal_threshold,
     spacing_tolerance,
     zoom_interpolation_order,
     purposes,
     categories,
     input_override,
     copy_directory,
     work_directory,
     normalizer,
     source_range,
     target_range,
     full_connectivity,
     unrestrict_network,
     test_augmentation,
     soft_classification_mode,
     quantize_output,
     no_mask,
     keep_intermediate_files,
     keep_copied_files,
     overwrite) = collect_arguments()

    # Assemble jobs.
    #
    job_list = assemble_jobs(image_path=image_path,
                             image_filter_regexp=image_filter_regexp,
                             mask_path=mask_path,
                             model_path=network_model_path,
                             purpose_filter=purposes,
                             category_filter=categories,
                             path_override=input_override,
                             output_path=output_path,
                             no_mask_from_config=no_mask)

    # Check if there are any identified jobs.
    #
    if job_list:
        # Init the logger to print to the console.
        #
        dptloggers.init_console_logger(debug=True)

        # Execute jobs.
        #
        successful_items, failed_items = dptinference.apply_network_batch(job_list=job_list,
                                                                          model_path=network_model_path,
                                                                          patch_size=processing_patch_size,
                                                                          output_class=network_output_class,
                                                                          number_of_classes=network_num_classes,
                                                                          normalizer=normalizer,
                                                                          normalizer_source_range=source_range,
                                                                          normalizer_target_range=target_range,
                                                                          soft_mode=soft_classification_mode,
                                                                          input_spacing=input_spacing,
                                                                          output_spacing=output_spacing,
                                                                          spacing_tolerance=spacing_tolerance,
                                                                          unrestrict_network=unrestrict_network,
                                                                          input_channels=input_channels,
                                                                          padding_mode=padding_mode,
                                                                          padding_constant=padding_constant,
                                                                          confidence=network_confidence,
                                                                          test_augmentation=test_augmentation,
                                                                          minimum_region_diagonal=region_diagonal_threshold,
                                                                          dilation_distance=dilation_distance,
                                                                          minimum_hole_diagonal=hole_diagonal_threshold,
                                                                          full_connectivity=full_connectivity,
                                                                          quantize=quantize_output,
                                                                          interpolation_order=zoom_interpolation_order,
                                                                          copy_path=copy_directory,
                                                                          work_path=work_directory,
                                                                          clear_cache=not keep_copied_files,
                                                                          keep_intermediates=keep_intermediate_files,
                                                                          single_mode=False,
                                                                          overwrite=overwrite)

        # Print the collection of failed cases.
        #
        if failed_items:
            print('Failed on {count} items:'.format(count=len(failed_items)))
            for path in failed_items:
                print('{path}'.format(path=path))

        error_code = len(failed_items)

    else:
        # Failed to identify any jobs.
        #
        print('No images matched the input filter.')

        error_code = -1

    return error_code

#----------------------------------------------------------------------------------------------------


if __name__ == '__main__':

    # Return error code.
    #
    sys.exit(main())
