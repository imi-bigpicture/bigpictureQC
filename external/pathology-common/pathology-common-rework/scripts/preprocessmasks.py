"""
This module contains a function that can convert mask files to PatchSampler stat files.
"""

import digitalpathology.generator.batch.batchsource as dptbatchsource
import digitalpathology.generator.patch.patchsource as dptpatchsource
import digitalpathology.generator.mask.maskstats as dptmaskstats
import digitalpathology.image.io.imagereader as dptimagereader
import digitalpathology.utils.loggers as dptloggers

import argparse
import logging
import os
import sys

#----------------------------------------------------------------------------------------------------

def preprocess_mask(patch_source, stat_path_exp, item_index, total_count, pixel_spacing, spacing_tolerance, overwrite):
    """
    Convert mask file to PatchSampler stats file.

    Args:
        patch_source (dptpatchsource.PatchSource): Patch source with mask image path and processing level.
        stat_path_exp (str, None): Path of the result stat file.
        item_index (int): Index of current item to process.
        total_count (int): Total number of items to process.
        pixel_spacing (float): Pixel spacing (micrometer).
        spacing_tolerance (float): Pixel spacing tolerance (percentage).
        overwrite (bool): If true existing targets will be overwritten.

    Raises:
        DigitalPathologyConfigError: Configuration errors.
        DigitalPathologyImageError: Image errors.
        DigitalPathologyStatError: Stats errors.
    """

    # Construct the result stats path.
    #
    if stat_path_exp:
        # Path expression is given. If it is a directory just use the basename of the mask file and add '.stat' extension it. If it is an expression
        # replace the {mask} tag in it with the mask file basename.
        #
        mask_base = os.path.splitext(os.path.basename(patch_source.mask))[0]
        if os.path.isdir(stat_path_exp):
            target_stat_path = os.path.join(stat_path_exp, '{base}.{ext}'.format(base=mask_base, ext='stat'))
        else:
            target_stat_path = stat_path_exp.format(mask=mask_base)
    elif patch_source.stat:
        # Target path is not given but the paths are set in the data configuration file.
        #
        target_stat_path = patch_source.stat
    else:
        # Nothing found. Just store the result stat file next to the mask with '.stat' extension.
        #
        target_stat_path = '{path}.{ext}'.format(path=os.path.splitext(patch_source.mask)[0], ext='stat')

    # Print progress.
    #
    logger = logging.getLogger(name=__name__)
    logger.info('Processing [{index}/{count}]: {path}'.format(path=patch_source.mask, index=item_index+1, count=total_count))

    # Check if target already exists.
    #
    if not os.path.exists(target_stat_path) or overwrite:
        # Open the mask to check if it has pixel spacing information. Older versions of ASAP did not add the pixel spacing information to the mask file when converting from annotations.
        # If the information is missing it can be still assumed that on the level where the mask has exactly the same shape as the source image, they have the same pixel spacing too.
        #
        mask_image = dptimagereader.ImageReader(image_path=patch_source.mask, spacing_tolerance=spacing_tolerance, input_channels=None, cache_path=None)
        if any(mask_spacing is None for mask_spacing in mask_image.spacings):

            source_image = dptimagereader.ImageReader(image_path=patch_source.image, spacing_tolerance=spacing_tolerance, input_channels=None, cache_path=None)
            for image_level in range(len(source_image.shapes)):
                if source_image.shapes[image_level] == mask_image.shapes[0]:
                    mask_image.correct(spacing=source_image.spacings[image_level], level=0)
                    break

        # Process mask and save the result.
        #
        mask_stats = dptmaskstats.MaskStats(file=mask_image, mask_spacing=pixel_spacing, spacing_tolerance=spacing_tolerance, mask_labels=patch_source.labels)
        mask_stats.save(file_path=target_stat_path)
    else:
        # Print progress.
        #
        logger.info('Skipping, target file already exits: {path}'.format(path=target_stat_path))

#----------------------------------------------------------------------------------------------------

def preprocess_batch_source(data_path, path_replacements, pixel_spacing, spacing_tolerance, stat_path_exp, overwrite):
    """
    Convert each mask file in the BatchSource to stat file.

    Args:
        data_path (str): Batch source data file path.
        path_replacements (dict): Path replacements to add to batch source file.
        pixel_spacing (float): Pixel spacing (micrometer).
        spacing_tolerance (float): Pixel spacing tolerance (percentage).
        stat_path_exp (str, None): Stat output directory path.
        overwrite (bool): If true existing targets will be overwritten.

    Returns:
        (list, list): List of successfully processed items and list of failed items.

    Raises:
        DigitalPathologyConfigError: Configuration errors.
        DigitalPathologyDataError: Data errors.
        DigitalPathologyImageError: Image errors.
        DigitalPathologyStatError: Stats errors.
    """

    # Load JSON file to batch source.
    #
    batch_source = dptbatchsource.BatchSource(source_items=None)
    batch_source.load(file_path=data_path)
    batch_source.update(path_replacements=path_replacements)

    # Report the number of files to process.
    #
    mask_count = batch_source.count(purpose_id=None)

    logger = logging.getLogger(name=__name__)
    logger.info('Item count: {count}'.format(count=mask_count))

    # Process each image-mask pair in the batch source.
    #
    failed_collection = []
    successful_collection = []
    item_index = 0

    for source_item in batch_source.items(purpose_id=None, category_id=None, replace=True):
        try:
            preprocess_mask(patch_source=source_item,
                            stat_path_exp=stat_path_exp,
                            item_index=item_index,
                            total_count=mask_count,
                            pixel_spacing=pixel_spacing,
                            spacing_tolerance=spacing_tolerance,
                            overwrite=overwrite)

        except Exception as exception:
            # Add case to the error collection.
            #
            failed_collection.append(source_item.mask)
            logger.error('Error: {exception}'.format(exception=exception))

        else:
            # Save the result to the list of successful exports.
            #
            successful_collection.append(source_item.mask)

        item_index += 1

    # Return a list of successful computations.
    #
    return successful_collection, failed_collection

#----------------------------------------------------------------------------------------------------

def collect_arguments():
    """
    Collect command line arguments.

    Returns:
        (str, dict, float, float, str, bool): Data file path, path replacements, processing pixel spacing, and tolerance, stat target directory path, and overwrite flag.
    """

    # Configure argument parser.
    #
    argument_parser = argparse.ArgumentParser(description='Pre-process all mask files in the configuration file.',
                                              epilog='The target stat path can either be a directory or an expression with {mask} tag, that is replaced with the source mask file base name '
                                                     'for each processed stat file. If the target directory is not given the script will try to use stat path from the data configuration '
                                                     'file if possible. If it is not given either, it will just put the result file next to the source mask file.')

    argument_parser.add_argument('-d', '--data',         required=True,  type=str,                 help='data file path to process')
    argument_parser.add_argument('-r', '--replacements', required=False, type=str,   default=None, help='data path replacements')
    argument_parser.add_argument('-s', '--spacing',      required=True,  type=float,               help='processing pixel spacing (micrometer)')
    argument_parser.add_argument('-t', '--tolerance',    required=False, type=float, default=0.25, help='pixel spacing tolerance (percentage)')
    argument_parser.add_argument('-g', '--stat',         required=False, type=str,   default=None, help='stat directory expression')
    argument_parser.add_argument('-w', '--overwrite',    action='store_true',                      help='overwrite existing files')

    # Parse arguments.
    #
    arguments = vars(argument_parser.parse_args())

    # Collect arguments.
    #
    parsed_data_path = arguments['data']
    parsed_replacements_str = arguments['replacements']
    parsed_pixel_spacing = arguments['spacing']
    parsed_spacing_tolerance = arguments['tolerance']
    parsed_stat_path = arguments['stat']
    parsed_overwrite = arguments['overwrite']

    # Evaluate label expression.
    #
    parsed_replacements = eval(parsed_replacements_str) if parsed_replacements_str else {}

    # Print parameters.
    #
    print(argument_parser.description)
    print('Data configuration path: {path}'.format(path=parsed_data_path))
    print('Data path replacements: {map}'.format(map=parsed_replacements))
    print('Processing pixel spacing: {spacing} um'.format(spacing=parsed_pixel_spacing))
    print('Pixel spacing tolerance: {tolerance}'.format(tolerance=parsed_spacing_tolerance))
    print('Stat directory path: {path}'.format(path=parsed_stat_path))
    print('Overwrite existing results: {overwrite_flag}'.format(overwrite_flag=parsed_overwrite))

    # Return parsed values.
    #
    return parsed_data_path, parsed_replacements, parsed_pixel_spacing, parsed_spacing_tolerance, parsed_stat_path, parsed_overwrite

#----------------------------------------------------------------------------------------------------

def main():
    """
    Main function.

    Returns:
        int: Error code.
    """

    # Collect command line arguments.
    #
    data_config_path, replacements, spacing, tolerance, stat_path, overwrite_target = collect_arguments()

    # Init the logger to print to the console.
    #
    dptloggers.init_console_logger(debug=True)

    # Pre-process all masks.
    #
    successful_items, failed_items = preprocess_batch_source(data_path=data_config_path,
                                                             path_replacements=replacements,
                                                             pixel_spacing=spacing,
                                                             spacing_tolerance=tolerance,
                                                             stat_path_exp=stat_path,
                                                             overwrite=overwrite_target)

    # Print the collection of failed cases.
    #
    if failed_items:
        print('Failed on {count} items:'.format(count=len(failed_items)))
        for path in failed_items:
            print('{path}'.format(path=path))

    return len(failed_items)

#----------------------------------------------------------------------------------------------------


if __name__ == '__main__':

    sys.exit(main())
