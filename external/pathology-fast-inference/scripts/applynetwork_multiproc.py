"""
This module can load a network model and apply classification on a whole slide image.
"""
import digitalpathology.generator.batch.batchsource as dptbatchsource
import digitalpathology.utils.foldercontent as dptfoldercontent
import fastinference.async_wsi_consumer as dptasyncwsiconsumer
import argparse
import os
import pandas as pd

import random
import functools
print = functools.partial(print, flush=True)
# ----------------------------------------------------------------------------------------------------

def load_whitelist(cases_csv, key_col='slide_id'):
    """
    Returns:
      - whitelist_keys: set of basenames-without-extension (strings), or None if no CSV given
    """
    if not cases_csv:
        return None

    df = pd.read_csv(cases_csv)
    if key_col not in df.columns:
        raise ValueError(f"CSV must contain '{key_col}' column.")

    return set(df[key_col].astype(str).str.strip())



def assemble_jobs(image_path, image_filter_regexp, mask_path, model_path, purpose_filter, category_filter,
                  path_override, output_path, no_mask_from_config, cache_path=None, whitelist_keys=None, dicom=False, recursive=False):
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
        if whitelist_keys and image_base not in whitelist_keys:
            return []  # nothing to do
        
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

                if whitelist_keys and image_key not in whitelist_keys:
                    continue

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
            mask_wildcard_path = mask_path.format(image='*') if mask_path else ''

            if dicom:
                # DICOM series mode: each immediate subfolder is one case, choose one .dcm file per subfolder.
                # image_key = subfolder name (stable), image_path = chosen .dcm inside the subfolder.
                subdirs = [
                    os.path.join(image_path, d) for d in os.listdir(image_path)
                    if os.path.isdir(os.path.join(image_path, d))
                ]

                image_file_path_map = {}
                for sd in sorted(subdirs):
                    case_key = os.path.basename(sd)
                    if whitelist_keys and case_key not in whitelist_keys:
                        continue

                    # pick one .dcm file
                    dcm_files = dptfoldercontent.folder_content(folder_path=sd, filter_exp=r".*\.dcm$", recursive=False)
                    if not dcm_files:
                        continue

                    chosen = sorted(dcm_files)[0]
                    image_file_path_map[case_key] = chosen

                # Collect masks (still flat) as before
                mask_file_path_list = dptfoldercontent.folder_content(folder_path=mask_wildcard_path, filter_exp=None, recursive=False)
                mask_file_path_map = {os.path.splitext(os.path.basename(p))[0]: p for p in mask_file_path_list}

            else:
                # Normal folder mode: collect images, optionally recursing into subdirectories
                image_file_path_list = dptfoldercontent.folder_content(folder_path=image_path, filter_exp=image_filter_regexp, recursive=recursive)
                mask_file_path_list = dptfoldercontent.folder_content(folder_path=mask_wildcard_path, filter_exp=None, recursive=False)

                image_file_path_map = {os.path.splitext(os.path.basename(p))[0]: p for p in image_file_path_list}
                mask_file_path_map = {os.path.splitext(os.path.basename(p))[0]: p for p in mask_file_path_list}

                if whitelist_keys:
                    image_file_path_map = {k: v for k, v in image_file_path_map.items() if k in whitelist_keys}

            print('%d image and %d mask pathes found' % (len(image_file_path_map), len(mask_file_path_map)))

            # Construct image match expression (same logic)
            if mask_path:
                mask_match_base = '{image}' if os.path.isdir(mask_path) else os.path.splitext(os.path.basename(mask_path))[0]
                mask_match = mask_match_base if 0 <= mask_match_base.find('{image}') else '{image}'
            else:
                mask_match = ''

            for image_key, current_image_path in image_file_path_map.items():
                mask_key = mask_match.format(image=image_key)
                mask_replace = mask_key if mask_key in mask_file_path_map else ''
                target_output_path = output_path.format(image=image_key, mask=mask_replace, model=model_base)

                current_mask_path = mask_file_path_map.get(mask_key, None)

                if mask_path is None or current_mask_path is not None:
                    result_job_list.append((current_image_path, current_mask_path, target_output_path, cache_path))

            print('Matching image count: {match_count}'.format(match_count=len(result_job_list)))
            if len(result_job_list) == 0:
                print('NO IMAGE MATCHES!')

    # Return the result list.
    #
    return result_job_list


# ----------------------------------------------------------------------------------------------------

def collect_arguments():
    """
    Collect command line arguments.
    """
    argument_parser = argparse.ArgumentParser(
        description='Apply segmentation on mr-image using the given network model.',
        epilog='It can either work with file paths, file filter expressions or a data configuration JSON/YAML file. If filter expressions are set '
               'the matching is based on file names. Use {image} and {mask} as replacement for image and mask file base names in output and mask '
               'file name specification and image file to mask file matching. Additionally, {model} can be used for model for replacement for model '
               'file base name in output file specifications. With distributed JSON/YAML files the purpose can be filtered with the optional purpose '
               'option.')

    argument_parser.add_argument('--model_path', dest='model_path', required=True, help='filepath to the model.')
    argument_parser.add_argument('--input_wsi_path', dest='input_wsi_path', required=True,
                                 help='filepath to the input WSI(s), used by glob.')
    argument_parser.add_argument('--cache_path', dest='cache_directory', required=False, default=None,
                                 help='directory for caching the slides.')
    argument_parser.add_argument('--input_filter', required=False, type=str, default=None,
                                 help='image file filter regular expression')
    argument_parser.add_argument('--output_wsi_path', dest='output_wsi_path', required=True,
                                 help='output path filter. Use {image} to match with input wsi path filter.')
    argument_parser.add_argument('--mask_wsi_path', dest='mask_wsi_path', required=False, default=None,
                                 help='mask filter path of WSI. put {image} to match with input wsi path filter.')

    argument_parser.add_argument('--read_spacing', dest='read_spacing', required=True, type=float,
                                 help='spacing to sample the input WSI(s) at.')
    argument_parser.add_argument('--mask_spacing', dest='mask_spacing', required=False, default=None, type=float,
                                 help='spacing to sample the mask WSI at.')
    argument_parser.add_argument('--mask_class', dest='mask_class', required=False, default=1, type=int,
                                 help='the value/class that should be regarded as foreground in the mask')
    argument_parser.add_argument('--write_spacing', dest='write_spacing', required=False, default=None, type=float,
                                 help='spacing to write the image in.')
    argument_parser.add_argument('--lookup_table', dest='lookup_table', required=False, default=None, type=str, help='')
    argument_parser.add_argument('--purposes', dest='purposes', required=False, default=None, type=str, help='')
    argument_parser.add_argument('--categories', dest='categories', required=False, default=None, type=str, help='')
    argument_parser.add_argument('--overrides', dest='overrides', required=False, default=None, type=str, help='')

    argument_parser.add_argument('--no_mask', action='store_true', help='wether to use the masks of a config.')
    argument_parser.add_argument('--custom_processor', dest='custom_processor', required=False, default=None, type=str, help='')
    argument_parser.add_argument('--custom_writer', required=False, default=None, type=str, help='')
    argument_parser.add_argument('--custom_reader', required=False, default=None, type=str, help='')

    argument_parser.add_argument('--axes_order', dest='axes_order', required=False, default='whc', type=str,
                                 help='choose whc or cwh.')
    argument_parser.add_argument('--tile_size', dest='tile_size', required=False, default=512, type=int,
                                 help='input tile size.')
    argument_parser.add_argument('--augment', dest='augment', action='store_true',
                                 help='enable test time augmentation. Will use all 90 degree rotations with/without flipping.')
    argument_parser.add_argument('--soft', dest='soft', action='store_true', help='set soft classification')
    argument_parser.add_argument('--normalizer', dest='normalizer', required=False, default='default', type=str,
                                 help='do not use')
    argument_parser.add_argument('--output_channels', dest='output_channels', required=False, default=None, type=str,
                                 help='array of channels to write. Only works with `soft` configuration at the moment.')
    argument_parser.add_argument('--gpu_count', dest='gpu_count', required=False, default=1, type=int,
                                 help='amount of gpus to use')
    argument_parser.add_argument('--batch_size', dest='batch_size', required=False, default=8, type=int,
                                 help='batch size for gpu')
    argument_parser.add_argument('--reconstruction_information', dest='reconstruction_information', required=False,
                                 type=str, default='[[0,0,0,0],[1,1],[0,0,0,0]]',
                                 help='Manually set the reconstruction information of the network.')
    argument_parser.add_argument('--readers', dest='readers', required=False, type=int, default=1,
                                 help='Amount of WSI file readers to have open simultaneously.')
    argument_parser.add_argument('--writers', dest='writers', required=False, type=int, default=1,
                                 help='Amount of writer objects to create. Tasks will be distributed evenly.')
    argument_parser.add_argument('--profiler', dest='profiler', required=False, type=str, default=None, help='do not use')
    argument_parser.add_argument('--work_directory', dest='work_directory', required=False, default="/home/user/work", type=str,
                                 help='local directory to write to. Once finished will move the WSI to the output folder')
    argument_parser.add_argument('--unfix_network', action='store_true', help='removes the input dimensions constraint from the network')
    argument_parser.add_argument('--quantize', action='store_true', help='quantize float32 result to int8 [0, 255]')
    argument_parser.add_argument('--overwrite', action='store_true', help='overwrite existing results')
    argument_parser.add_argument('--touch', action='store_true', help='touch target output files first')
    argument_parser.add_argument('--verbose', action='store_true', help='verbose print statements')
    argument_parser.add_argument('--decoder_name-artifact', type=str, default='deeplabv3plus')
    argument_parser.add_argument('--encoder_name-artifact', type=str, default='tu-convnext_base')
    argument_parser.add_argument('--encoder_weights-artifact', type=str, default='imagenet')
    argument_parser.add_argument('--cases_csv', type=str, default=None, help='Optional CSV with cases to process.')
    argument_parser.add_argument('--num_shards', type=int, default=1,
                             help='split the job list into N shards for parallel runs')
    argument_parser.add_argument('--shard_id', type=int, default=0,
                             help='0-based index of the shard to run (must be < num_shards)')
    argument_parser.add_argument(
        "--postprocess",
        action="store_true",
        help="Apply region filtering and hole filling after WSI reconstruction"
    )
    argument_parser.add_argument(
        "--use_tta",
        action="store_true",
        help="Apply test time augmentation"
    )
    argument_parser.add_argument(
        "--dicom",
        action="store_true",
        help="Treat each subfolder in input_wsi_path as one DICOM series; pick one .dcm file per subfolder."
    )
    argument_parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively search subdirectories for images when in folder mode."
    )


    arguments = vars(argument_parser.parse_args())

    if arguments['purposes'] is not None:
        arguments['purposes'] = eval(arguments['purposes'])
    if arguments['categories'] is not None:
        arguments['categories'] = eval(arguments['categories'])
    if arguments['overrides'] is not None:
        arguments['overrides'] = eval(arguments['overrides'])
    if arguments['reconstruction_information']:
        arguments['reconstruction_information'] = eval(arguments['reconstruction_information'])

    print(argument_parser.description)
    print('Image path: {path}'.format(path=arguments['input_wsi_path']))
    print('Image filter: {regexp}'.format(regexp=arguments['input_filter']))
    print('Mask path: {path}'.format(path=arguments['mask_wsi_path']))
    print('Output path: {path}'.format(path=arguments['output_wsi_path']))
    print('Cache directory: {path}'.format(path=arguments['cache_directory']))
    print('Lookup table path: {path}'.format(path=arguments['lookup_table']))
    print('Model path: {path}'.format(path=arguments['model_path']))
    print('Patch size: {size}'.format(size=arguments['tile_size']))
    print('Axes order: {order}'.format(order=arguments['axes_order']))
    print('Output channels: {channels}'.format(channels=arguments['output_channels']))
    print('Network input tile size: {tile}'.format(tile=arguments['tile_size']))
    print('Read spacing: {read}'.format(read=arguments['read_spacing']))
    print('Write spacing: {write}'.format(write=arguments['write_spacing']))
    print('Mask spacing: {mask}'.format(mask=arguments['mask_spacing']))
    print('Mask class: {mask_class}'.format(mask_class=arguments['mask_class']))
    print('Soft classification: {flag}'.format(flag=arguments['soft']))
    print('Normalizer: {norm}'.format(norm=arguments['normalizer']))
    print('GPU count: {gpu}'.format(gpu=arguments['gpu_count']))
    print('GPU batch size: {batch}'.format(batch=arguments['batch_size']))
    print('Quantize: {flag}'.format(flag=arguments['quantize']))
    print('overwrite existing ouput: {flag}'.format(flag=arguments['overwrite']))
    print('Purposes in data config: {purposes}'.format(purposes=arguments['purposes']))
    print('Categories in data config: {categories}'.format(categories=arguments['categories']))
    print("Custom processors: {processor}".format(processor=arguments['custom_processor']))
    print('Path override in data config: {override}'.format(override=arguments['overrides']))
    print('Work directory path: {path}'.format(path=arguments['work_directory']))
    print('Unfix network: {flag}'.format(flag=arguments['unfix_network']))
    print('Test time augmentation: {flag}'.format(flag=arguments['augment']))
    print('Soft classification: {flag}'.format(flag=arguments['soft']))
    print('Quantize: {flag}'.format(flag=arguments['quantize']))
    print('Skip masks in data config: {flag}'.format(flag=arguments['no_mask']))
    print('Overwrite existing results: {flag}'.format(flag=arguments['overwrite']))
    print('Touch output files first: {flag}'.format(flag=arguments['touch']))

    # Return parsed values.
    #
    return arguments


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

    whitelist_keys = load_whitelist(
        arguments['cases_csv']
    )

    job_list = assemble_jobs(image_path=arguments['input_wsi_path'],
                             image_filter_regexp=arguments['input_filter'],
                             mask_path=arguments['mask_wsi_path'],
                             model_path=arguments['model_path'],
                             purpose_filter=arguments['purposes'],
                             category_filter=arguments['categories'],
                             path_override=arguments['overrides'],
                             output_path=arguments['output_wsi_path'],
                             cache_path=arguments['cache_directory'],
                             no_mask_from_config=arguments['no_mask'],
                             whitelist_keys=whitelist_keys,
                             dicom=arguments['dicom'],
                             recursive=arguments['recursive'])
    print("number of jobs found: {}".format(len(job_list)))

    random.shuffle(job_list)
    if arguments['num_shards'] > 1:
        job_list = job_list[arguments['shard_id']::arguments['num_shards']]
        print(f"Shard {arguments['shard_id']}/{arguments['num_shards']} -> {len(job_list)} jobs")


    output_channels = eval(arguments['output_channels']) if arguments['output_channels'] else None
    consumer = dptasyncwsiconsumer.async_wsi_consumer(model_path=arguments['model_path'],
                                                      lookup_table_path=arguments['lookup_table'],
                                                      axes_order=arguments['axes_order'],
                                                      batch_size=arguments['batch_size'],
                                                      gpu_count=arguments['gpu_count'],
                                                      write_spacing=arguments['write_spacing'],
                                                      mask_spacing=arguments['mask_spacing'],
                                                      mask_class=arguments['mask_class'],
                                                      read_spacing=arguments['read_spacing'],
                                                      normalizer=arguments['normalizer'],
                                                      augment=arguments['augment'],
                                                      soft=arguments['soft'],
                                                      quantize=arguments['quantize'],
                                                      tile_size=arguments['tile_size'],
                                                      recon_info=arguments['reconstruction_information'],
                                                      output_channels=output_channels,
                                                      readers=arguments['readers'],
                                                      writers=arguments['writers'],
                                                      profiler=arguments['profiler'],
                                                      work_directory=arguments['work_directory'],
                                                      unfix_network=arguments['unfix_network'],
                                                      overwrite=arguments['overwrite'],
                                                      touch=arguments['touch'],
                                                      custom_processor=arguments['custom_processor'],
                                                      custom_writer=arguments['custom_writer'],
                                                      custom_reader=arguments['custom_reader'],
                                                      verbose=arguments['verbose'],
                                                      encoder_name_artifact=arguments['encoder_name_artifact'],
                                                      encoder_weights_artifact=arguments['encoder_weights_artifact'],
                                                      decoder_name_artifact=arguments['decoder_name_artifact'], 
                                                      postprocess=arguments['postprocess'],
                                                      use_tta=arguments['use_tta'])
    print("Model path:", arguments['model_path'])
    print("Lookup table path:", arguments['lookup_table'])
    print("Axes order:", arguments['axes_order'])
    print("Batch size:", arguments['batch_size'])
    print("GPU count:", arguments['gpu_count'])
    print("Write spacing:", arguments['write_spacing'])
    print("Mask spacing:", arguments['mask_spacing'])
    print("Mask class:", arguments['mask_class'])
    print("Read spacing:", arguments['read_spacing'])
    print("Normalizer:", arguments['normalizer'])
    print("Augment:", arguments['augment'])
    print("Soft:", arguments['soft'])
    print("Quantize:", arguments['quantize'])
    print("Tile size:", arguments['tile_size'])
    print("Reconstruction information:", arguments['reconstruction_information'])
    print("Output channels:", output_channels)
    print("Readers:", arguments['readers'])
    print("Writers:", arguments['writers'])
    print("Profiler:", arguments['profiler'])
    print("Work directory:", arguments['work_directory'])
    print("Unfix network:", arguments['unfix_network'])
    print("Overwrite:", arguments['overwrite'])
    print("Touch:", arguments['touch'])
    print("Custom processor:", arguments['custom_processor'])
    print("Custom writer:", arguments['custom_writer'])
    print("Custom reader:", arguments['custom_reader'])
    print("Verbose:", arguments['verbose'])


    consumer.apply_network_on_joblist(job_list)


# ----------------------------------------------------------------------------------------------------


if __name__ == '__main__':
    main()
