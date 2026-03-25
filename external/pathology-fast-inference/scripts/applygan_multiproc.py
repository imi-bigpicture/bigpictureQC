"""
This module can load a network model and apply classification on a whole slide image.
"""
import digitalpathology.generator.batch.batchsource as dptbatchsource
import digitalpathology.utils.foldercontent as dptfoldercontent
import fastinference.gan_inference.gan_wsi_consumer as dptasyncwsiconsumer
import argparse
import os


# ----------------------------------------------------------------------------------------------------

def assemble_jobs(image_path, image_filter_regexp, mask_path, model_path, purpose_filter, category_filter,
                  path_override, output_path, no_mask_from_config, cache_path=None):
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
    file_mode = os.path.isfile(image_path) and not data_config
    print('assemble jobs for image_path %s' % image_path)
    print('Mode: {mode}'.format(mode=('file' if file_mode else 'folder')))

    model_base = os.path.splitext(os.path.basename(model_path))[0]
    result_job_list = []
    if file_mode:
        # Return a single triplet if the paths were existing files.
        #
        image_base = os.path.splitext(os.path.basename(image_path))[0]
        target_output_path = output_path.format(image=image_base, model=model_base)

        # Add job item to the list.
        #
        job_item = (image_path, mask_path, target_output_path, cache_path)
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

            target_purposes = [purpose_id for purpose_id in batch_source.purposes() if
                               purpose_id in purpose_filter] if purpose_filter else None
            available_categories = batch_source.categories(purpose_id=target_purposes)
            target_categories = [category_id for category_id in available_categories if
                                 category_id in category_filter] if category_filter else None

            # Go through the filtered purposes and collect all images.
            #
            for source_item in batch_source.items(purpose_id=target_purposes, category_id=target_categories,
                                                  replace=True):
                image_item_path = source_item.image
                mask_item_path = source_item.mask
                image_key = os.path.splitext(os.path.basename(image_item_path))[0]
                mask_key = os.path.splitext(os.path.basename(mask_item_path))[0]

                current_mask_path = None
                if not no_mask_from_config:
                    current_mask_path = mask_item_path
                if mask_path is not None: #given mask path overwrites config mask_path
                    if 'image' in mask_path:
                        current_mask_path = mask_path.format(image=image_key)
                    else:
                        raise ValueError('mask path %s doesnt contain "{image}"' % mask_path)


                target_output_path = output_path.format(image=image_key, mask=mask_key, model=model_base)

                # Add job item to the list.
                #
                job_item = (image_item_path, current_mask_path, target_output_path, cache_path)
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
            image_file_path_list = dptfoldercontent.folder_content(folder_path=image_path,
                                                                   filter_exp=image_filter_regexp, recursive=False)
            mask_file_path_list = dptfoldercontent.folder_content(folder_path=mask_wildcard_path, filter_exp=None,
                                                                  recursive=False)
            print('%d image and %d mask pathes found' % (len(image_file_path_list), len(mask_file_path_list)))
            # Build file name to path maps.
            #
            image_file_path_map = {os.path.splitext(os.path.basename(image_path_item))[0]: image_path_item for
                                   image_path_item in image_file_path_list}
            mask_file_path_map = {os.path.splitext(os.path.basename(mask_path_item))[0]: mask_path_item for
                                  mask_path_item in mask_file_path_list}

            # Construct image match expression.
            #
            if mask_path:
                mask_match_base = '{image}' if os.path.isdir(mask_path) else \
                os.path.splitext(os.path.basename(mask_path))[0]
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
                    job_item = (current_image_path, current_mask_path, target_output_path, cache_path)
                    result_job_list.append(job_item)

            # Print match count for checking.
            #
            print('Matching image count: {match_count}'.format(match_count=len(result_job_list)))
            if len(result_job_list)==0:
                print('NO IMAGE MATCHES!')

    # Return the result list.
    #
    return result_job_list


# ----------------------------------------------------------------------------------------------------

def collect_arguments():
    """
    Collect command line arguments.
    Returns:
        (str, str, float, str, str, float, str, int, int, int, list, str, int, float, float, float, float, float, int, list, list, dict, str, str, str, list, list, bool, bool, bool, bool,
            bool, bool, bool, bool): The parsed command line arguments: input image path, input file filter regular expression, input image pixel spacing, mask image path, output image, output
            image pixel spacing, model file path, processing patch size, output class, number of output classes, input channel indices, padding mode, padding mode constant, network confidence,
            region diagonal threshold, dilation distance, hole diagonal threshold, pixel spacing tolerance, interpolation order, purpose list for data config files, category list for data
            config files, path override map for data config files, target directory path for data copy, work directory path, normalizer type, normalizer input interval, normalizer output
            interval, full connectivity matrix flag, unrestrict network flag, soft classification mode flag, quantization flag, no masks from data config files flag, keep intermediate files
            flag, keep copied files flag, and the overwrite flag.
    """
    # Configure argument parser.
    #
    argument_parser = argparse.ArgumentParser(
        description='Apply segmentation on mr-image using the given network model.',
        epilog='It can either work with file paths, file filter expressions or a data configuration JSON/YAML file. If filter expressions are set '
               'the matching is based on file names. Use {image} and {mask} as replacement for image and mask file base names in output and mask '
               'file name specification and image file to mask file matching. Additionally, {model} can be used for model for replacement for model '
               'file base name in output file specifications. With distributed JSON/YAML files the purpose can be filtered with the optional purpose '
               'option.')

    argument_parser.add_argument('--model_path', dest='model_path', required=True, help='model path')
    argument_parser.add_argument('--cache_directory', dest='cache_directory', required=False, default=None,
                                 help='directory for caching the slides.')
    argument_parser.add_argument('--input_wsi_path', dest='input_wsi_path', required=True, help='input WSI')
    argument_parser.add_argument('--input_filter', required=False, type=str, default=None,
                                 help='image file filter regular expression')
    argument_parser.add_argument('--output_wsi_path', dest='output_wsi_path', required=True, help='output WSI')
    argument_parser.add_argument('--mask_wsi_path', dest='mask_wsi_path', required=False, default=None,
                                 help='mask of WSI')
    argument_parser.add_argument('--read_spacing', dest='read_spacing', required=True, type=float, help='')
    argument_parser.add_argument('--mask_spacing', dest='mask_spacing', required=False, default=None, type=float,
                                 help='')
    argument_parser.add_argument('--write_spacing', dest='write_spacing', required=False, default=None, type=float,
                                 help='')
    argument_parser.add_argument('--purposes', dest='purposes', required=False, default=None, type=str, help='')
    argument_parser.add_argument('--categories', dest='categories', required=False, default=None, type=str, help='')
    argument_parser.add_argument('--overrides', dest='overrides', required=False, default=None, type=str, help='')

    argument_parser.add_argument('--axes_order', dest='axes_order', required=False, default='whc', type=str, help='')
    argument_parser.add_argument('--tile_size', dest='tile_size', required=False, default=512, type=int,
                                 help='tile size')
    argument_parser.add_argument('--output_tile_size', dest='output_tile_size', required=False, default=512, type=int,
                                 help='')
    argument_parser.add_argument('--augment', dest='augment', action='store_true', help='enable test time augmentation')
    argument_parser.add_argument('--normalizer', dest='normalizer', required=False, default='default', type=str,
                                 help='')
    argument_parser.add_argument('--output_channels', dest='output_channels', required=False, default=None, type=str,
                                 help='')
    argument_parser.add_argument('--gpu_count', dest='gpu_count', required=False, default=1, type=int,
                                 help='amount of gpus to use')
    argument_parser.add_argument('--batch_size', dest='batch_size', required=False, default=8, type=int,
                                 help='batch size for gpu')
    argument_parser.add_argument('--reconstruction_information', dest='reconstruction_information', required=False,
                                 type=str, default='[[0,0,0,0],[1,1],[0,0,0,0]]', help='')
    argument_parser.add_argument('--readers', dest='readers', required=False, type=int, default=3, help='')
    argument_parser.add_argument('--writers', dest='writers', required=False, type=int, default=1, help='')
    argument_parser.add_argument('--profiler', dest='profiler', required=False, type=str, default=None, help='')
    argument_parser.add_argument('--work_directory', dest='work_directory', required=False, default=None, type=str,
                                 help='')
    argument_parser.add_argument('--unfix_network', action='store_true', help='allow arbitrarily sized input')
    argument_parser.add_argument('--overwrite', action='store_true', help='overwrite existing results')
    argument_parser.add_argument('--touch', action='store_true', help='touch target output files first')

    arguments = vars(argument_parser.parse_args())

    if arguments['purposes'] is not None:
        arguments['purposes'] = eval(arguments['purposes'])
    if arguments['categories'] is not None:
        arguments['categories'] = eval(arguments['categories'])
    if arguments['overrides'] is not None:
        arguments['overrides'] = eval(arguments['overrides'])
    if arguments['reconstruction_information']:
        arguments['reconstruction_information'] = eval(arguments['reconstruction_information'])
    arguments['output_channels'] = eval(arguments['output_channels']) if arguments['output_channels'] else None

    print(argument_parser.description)
    print('Image path: {path}'.format(path=arguments['input_wsi_path']))
    print('Image filter: {regexp}'.format(regexp=arguments['input_filter']))
    print('Mask path: {path}'.format(path=arguments['mask_wsi_path']))
    print('Output path: {path}'.format(path=arguments['output_wsi_path']))
    print('Model path: {path}'.format(path=arguments['model_path']))
    print('Patch size: {size}'.format(size=arguments['tile_size']))
    print('Axes order: {order}'.format(order=arguments['axes_order']))
    print('Output channels: {channels}'.format(channels=arguments['output_channels']))
    print('Network input tile size: {tile}'.format(tile=arguments['tile_size']))
    print('Read spacing: {read}'.format(read=arguments['read_spacing']))
    print('Write spacing: {write}'.format(write=arguments['write_spacing']))
    print('Mask spacing: {mask}'.format(mask=arguments['mask_spacing']))
    print('WSI output tile size: {tile}'.format(tile=arguments['output_tile_size']))
    print('Normalizer: {norm}'.format(norm=arguments['normalizer']))
    print('GPU count: {gpu}'.format(gpu=arguments['gpu_count']))
    print('GPU batch size: {batch}'.format(batch=arguments['batch_size']))
    print('overwrite existing ouput: {flag}'.format(flag=arguments['overwrite']))
    print('Purposes in data config: {purposes}'.format(purposes=arguments['purposes']))
    print('Categories in data config: {categories}'.format(categories=arguments['categories']))
    print('Path override in data config: {override}'.format(override=arguments['overrides']))
    print('Work directory path: {path}'.format(path=arguments['work_directory']))
    print('Unfix network: {flag}'.format(flag=arguments['unfix_network']))
    print('Test time augmentation: {flag}'.format(flag=arguments['augment']))
    print('Overwrite existing results: {flag}'.format(flag=arguments['overwrite']))
    print('Touch output files first: {flag}'.format(flag=arguments['touch']))
    

    # Return parsed values.
    #
    return arguments

import random
# ----------------------------------------------------------------------------------------------------

def main():
    """
    Main function.
    Returns:
        int: Error code.
    """

    # Collect command line arguments.
    #
    arguments = collect_arguments()

    # Assemble job octets: (source image path, source mask path, copy image path, copy mask path, work output path, work interval path, target output path, target interval path).
    #
    job_list = assemble_jobs(image_path=arguments['input_wsi_path'],
                             image_filter_regexp=arguments['input_filter'],
                             mask_path=arguments['mask_wsi_path'],
                             model_path=arguments['model_path'],
                             purpose_filter=arguments['purposes'],
                             category_filter=arguments['categories'],
                             path_override=arguments['overrides'],
                             output_path=arguments['output_wsi_path'],
                             cache_path=arguments['cache_directory'],
                             no_mask_from_config=False)
    print("number of jobs found: {}".format(len(job_list)))
    
    consumer = dptasyncwsiconsumer.gan_wsi_consumer(model_path=arguments['model_path'],
                                                    axes_order=arguments['axes_order'],
                                                    batch_size=arguments['batch_size'],
                                                    gpu_count=arguments['gpu_count'],
                                                    write_spacing=arguments['write_spacing'],
                                                    mask_spacing=arguments['mask_spacing'],
                                                    read_spacing=arguments['read_spacing'],
                                                    normalizer=arguments['normalizer'],
                                                    augment=arguments['augment'],
                                                    tile_size=arguments['tile_size'],
                                                    recon_info=arguments['reconstruction_information'],
                                                    output_channels=arguments['output_channels'],
                                                    readers=arguments['readers'],
                                                    writers=arguments['writers'],
                                                    profiler=arguments['profiler'],
                                                    work_directory=arguments['work_directory'],
                                                    unfix_network=arguments['unfix_network'],
                                                    overwrite=arguments['overwrite'],
                                                    touch=arguments['touch'])

    consumer.apply_network_on_joblist(job_list)


# ----------------------------------------------------------------------------------------------------


if __name__ == '__main__':
    main()
