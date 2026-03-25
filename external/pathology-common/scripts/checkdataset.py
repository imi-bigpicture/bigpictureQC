"""
This module can check if a dataset descriptor YAML file is valid.
"""

import digitalpathology.generator.batch.batchsource as dptbatchsource
import digitalpathology.image.io.imagereader as dptimagereader
import digitalpathology.utils.loggers as dptloggers

import argparse
import logging
import numpy as np
import sys

#----------------------------------------------------------------------------------------------------

def check_data_config(data_config_path, match_spacing, path_override_map, check_spacing, fix_spacing):
    """
    Check the data configuration file.

    Args:
        data_config_path (str): Data config file path.
        match_spacing (float): Spacing where the image should match its mask.
        path_override_map (dict): Path overrides for the data config.
        check_spacing (float): Spacing where the labels in the mask should be checked.
        fix_spacing (float, None): Spacing to set at level 0 of the mask in case the spacing is missing from it.

    Returns:
        list, list: List of correct items, list of incorrect items.
    """

    # Create logger for printing to the console.
    #
    logger = logging.getLogger(name=__name__)

    # Load the data source.
    #
    logger.info('Loading: {path}'.format(path=data_config_path))

    batch_source = dptbatchsource.BatchSource()
    batch_source.load(file_path=data_config_path)
    batch_source.update(path_replacements=path_override_map)

    # Check each item.
    #
    correct_items = []
    failed_items = []

    for source_item in batch_source.items(replace=True):
        try:
            logger.info('Processing: {path}'.format(path=source_item.image))

            image = dptimagereader.ImageReader(image_path=source_item.image, spacing_tolerance=0.25, input_channels=None, cache_path=None)
            mask = dptimagereader.ImageReader(image_path=source_item.mask, spacing_tolerance=0.25, input_channels=None, cache_path=None)

            if fix_spacing is not None and None in mask.spacings:
                mask.correct(spacing=fix_spacing, level=0)

            # Check if the labels match the configured ones.
            #
            available_labels = np.unique(mask.content(spacing=check_spacing)).tolist()
            if set(available_labels) == set(source_item.labels + (0,)):
                # Check if the image matches the mask at the given spacing.
                #
                if image.test(spacing=match_spacing):
                    if mask.test(spacing=match_spacing):
                        if image.shapes[image.level(spacing=match_spacing)] == mask.shapes[mask.level(spacing=match_spacing)]:
                            # Save the result to the list of successful zooms.
                            #
                            correct_items.append(source_item.image)
                        else:
                            # Missing level from mask: add case to the error collection.
                            #
                            failed_items.append(source_item.image)
                            logger.error('No matching level in mask: {path}'.format(path=source_item.mask))
                    else:
                        # Missing image spacing: add case to the error collection.
                        #
                        failed_items.append(source_item.image)
                        logger.error('Missing spacing from mask: {path}'.format(path=source_item.mask))
                else:
                    # Missing image spacing: add case to the error collection.
                    #
                    failed_items.append(source_item.image)
                    logger.error('Missing spacing from image: {path}'.format(path=source_item.image))
            else:
                # Mask label mismatch: add case to the error collection.
                #
                failed_items.append(source_item.image)
                logger.error('Labels mismatch: {available} != {config} for {path}'.format(available=available_labels, config=source_item.labels, path=source_item.mask))

            mask.close()
            image.close()

        except Exception as exception:
            # Add case to the error collection.
            #
            failed_items.append(source_item.image)
            logger.error('Error: {exception}'.format(exception=exception))

    return correct_items, failed_items

#----------------------------------------------------------------------------------------------------

def collect_arguments():
    """
    Collect command line arguments.

    Returns:
        (str, float, dict, float, float): Data file path, image to mask matching spacing, data path override map, label checking spacing, and mask fix spacing.
    """

    # Configure argument parser.
    #
    argument_parser = argparse.ArgumentParser(description='Check dataset configuration.')

    argument_parser.add_argument('-d', '--data',     required=True,  type=str,                 help='input data file')
    argument_parser.add_argument('-m', '--match',    required=True,  type=float,               help='image to mask matching spacing')
    argument_parser.add_argument('-o', '--override', required=False, type=str,   default=None, help='path overrides')
    argument_parser.add_argument('-c', '--check',    required=False, type=float, default=8.0,  help='checking mask spacing')
    argument_parser.add_argument('-f', '--fix',      required=False, type=float, default=None, help='fix mask spacing at level 0')

    # Parse arguments.
    #
    arguments = vars(argument_parser.parse_args())

    # Collect arguments.
    #
    parsed_data_config_path = arguments['data']
    parsed_match_spacing = arguments['match']
    parsed_path_override_map_str = arguments['override']
    parsed_check_spacing = arguments['check']
    parsed_fix_spacing = arguments['fix']

    # Evaluate expressions
    #
    parsed_path_override_map = eval(parsed_path_override_map_str) if parsed_path_override_map_str else None

    # Print parameters.
    #
    print(argument_parser.description)
    print('Data file: {data}'.format(data=parsed_data_config_path))
    print('Match spacing: {spacing} um'.format(spacing=parsed_match_spacing))
    print('Path overrides: {map}'.format(map=parsed_path_override_map))
    print('Check spacing: {spacing} um'.format(spacing=parsed_check_spacing))
    print('Fix mask spacing to: {spacing}{measure}'.format(spacing=parsed_fix_spacing, measure=' um' if parsed_fix_spacing is not None else ''))

    # Return parsed values.
    #
    return parsed_data_config_path, parsed_match_spacing, parsed_path_override_map, parsed_check_spacing, parsed_fix_spacing

#----------------------------------------------------------------------------------------------------

def main():
    """Main function."""

    # Collect command line arguments.
    #
    data_config_path, match_spacing, path_override_map, check_spacing, fix_spacing = collect_arguments()

    # Init the logger to print to the console.
    #
    dptloggers.init_console_logger(debug=True)

    # Check data configuration file.
    #
    correct_items, failed_items = check_data_config(data_config_path=data_config_path,
                                                    match_spacing=match_spacing,
                                                    path_override_map=path_override_map,
                                                    check_spacing=check_spacing,
                                                    fix_spacing=fix_spacing)

    # Print the collection of failed cases.
    #
    if failed_items:
        print('Failed on {count} items:'.format(count=len(failed_items)))
        for path in failed_items:
            print('{path}'.format(path=path))
    else:
        print('All {count} source items are okay'.format(count=len(correct_items)))

    return len(failed_items)

# ----------------------------------------------------------------------------------------------------


if __name__ == '__main__':

    sys.exit(main())
