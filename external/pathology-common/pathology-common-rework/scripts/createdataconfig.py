"""
This module contains a function that build a file database file for the digitalpathology.batch.batchsource.BatchSource class.
"""

import digitalpathology.utils.dataconfig as dptdataconfig
import digitalpathology.utils.loggers as dptloggers

import argparse
import logging
import datetime
import time
import os

#----------------------------------------------------------------------------------------------------

def create_data_container(output_path, image_path, mask_path, stat_path, labels, purpose_ratios, mask_level, mask_spacing, mask_spacing_tolerance, random_item_order, overwrite):
    """
    Build a data container and write it to file.

    Args:
        output_path (str): Target path.
        image_path (str): Whole-slide image filter expression.
        mask_path (str): Mask image filter expression.
        stat_path (str): Stat file filter expression.
        labels (list): List of labels to add.
        purpose_ratios (dict, None): Purpose ratios.
        mask_level (int, None): Level to read the masks on for extracting label information. Setting both this and mask_spacing arguments to None disables this functionality.
        mask_spacing (float, None): Pixel spacing to read the masks on for extracting label information (micrometer). Setting both this and mask_level arguments to None disables this functionality.
        mask_spacing_tolerance (float, None): Pixel spacing tolerance (percentage).
        random_item_order (bool): Randomize item order in the output file.
        overwrite (bool): Overwrite flag.

    Raises:
        DigitalPathologyConfigError: Configuration errors.
        DigitalPathologyImageError: Image errors.
    """

    # Start time measurement.
    #
    start_time = time.time()

    # Get a logger.
    #
    logger = logging.getLogger(name=__name__)

    # Check if target already exists.
    #
    if not os.path.exists(path=output_path) or overwrite:
        # Create the BatchSource object.
        #
        batch_source = dptdataconfig.build_batch_source(image_path=image_path,
                                                        mask_path=mask_path,
                                                        stat_path=stat_path,
                                                        labels=labels,
                                                        read_level=mask_level,
                                                        read_spacing=mask_spacing,
                                                        purpose_distribution=purpose_ratios,
                                                        mask_spacing_tolerance=mask_spacing_tolerance,
                                                        random_item_order=random_item_order)

        # Export the object to file.
        #
        logger.info('Writing data config: {path}'.format(path=output_path))

        batch_source.save(file_path=output_path)

        # Report execution time.
        #
        execution_time = time.time() - start_time
        logger.debug('Done in {delta}'.format(delta=datetime.timedelta(seconds=execution_time)))

    else:
        # Target already exists.
        #
        logger.info('Skipping, target already exists: {path}'.format(path=output_path))

#----------------------------------------------------------------------------------------------------

def collect_arguments():
    """
    Collect command line arguments.

    Returns:
        (str, str, str, str, list, dict, int, float, float, bool bool): Parsed output file path, image, mask and stat file filters, label list, purpose ratios, label checking level, pixel spacing,
            and pixel spacing tolerance, random oder flag and overwrite flag.
    """

    # Configure argument parser.
    #
    argument_parser = argparse.ArgumentParser(description='Create data source configuration files.',
                                              epilog='Image, mask and stat file matching is based on file names. Use {image} replacement for image file base names in mask and stat file name '
                                                     'specification for filtering and matching.')

    argument_parser.add_argument('-o', '--output',   required=True,  type=str,               help='output JSON or YAML file path')
    argument_parser.add_argument('-i', '--images',   required=True,  type=str,               help='whole-slide image path filter')
    argument_parser.add_argument('-m', '--masks',    required=False, type=str, default=None, help='mask path filter')
    argument_parser.add_argument('-d', '--stats',    required=False, type=str, default=None, help='stat path filter')
    argument_parser.add_argument('-l', '--labels',   required=True,  type=str,               help='label list')
    argument_parser.add_argument('-p', '--purposes', required=False, type=str, default=None, help='dictionary of purpose ratios')

    argument_group = argument_parser.add_mutually_exclusive_group(required=True)
    argument_group.add_argument('-r', '--level',   type=int,   default=None, help='mask level to read labels from')
    argument_group.add_argument('-s', '--spacing', type=float, default=None, help='mask pixel spacing to read labels from (micrometer)')

    argument_parser.add_argument('-t', '--tolerance', required=False, type=float, default=0.25, help='pixel spacing tolerance (percentage)')
    argument_parser.add_argument('-n', '--random',    action='store_true',                      help='add the items in random order')
    argument_parser.add_argument('-w', '--overwrite', action='store_true',                      help='overwrite existing output')

    # Parse arguments.
    #
    arguments = vars(argument_parser.parse_args())

    # Collect arguments.
    #
    parsed_output_path = arguments['output']
    parsed_image_filter = arguments['images']
    parsed_mask_filter = arguments['masks']
    parsed_stat_filter = arguments['stats']
    parsed_label_list_str = arguments['labels']
    parsed_purposes_str = arguments['purposes']
    parsed_level = arguments['level']
    parsed_pixel_spacing = arguments['spacing']
    parsed_spacing_tolerance = arguments['tolerance']
    parsed_random = arguments['random']
    parsed_overwrite = arguments['overwrite']

    # Parse label list and purposes.
    #
    parsed_label_list = eval(parsed_label_list_str)
    parsed_purposes = eval(parsed_purposes_str) if parsed_purposes_str else None

    # Print parameters.
    #
    print(argument_parser.description)
    print('Output file path: {path}'.format(path=parsed_output_path))
    print('Image filter: {name}'.format(name=parsed_image_filter))
    print('Mask filter: {name}'.format(name=parsed_mask_filter))
    print('Stat filter: {name}'.format(name=parsed_stat_filter))
    print('Label list: {labels}'.format(labels=parsed_label_list))
    print('Purpose ratios: {purposes}'.format(purposes=parsed_purposes))

    if parsed_pixel_spacing is not None:
        print('Reading labels from mask at spacing: {spacing} um'.format(spacing=parsed_pixel_spacing))
        print('Pixel spacing tolerance: {tolerance}'.format(tolerance=parsed_spacing_tolerance))
    else:
        print('Reading labels from mask at level: {level}'.format(level=parsed_level))

    print('Adding items in random order: {flag}'.format(flag=parsed_random))
    print('Overwrite existing target: {flag}'.format(flag=parsed_overwrite))

    # Return parsed values.
    #
    return (parsed_output_path,
            parsed_image_filter,
            parsed_mask_filter,
            parsed_stat_filter,
            parsed_label_list,
            parsed_purposes,
            parsed_level,
            parsed_pixel_spacing,
            parsed_spacing_tolerance,
            parsed_random,
            parsed_overwrite)

#----------------------------------------------------------------------------------------------------

def main():
    """Main function."""

    # Retrieve command line arguments.
    #
    result_path, image_filter, mask_filter, stat_filter, label_list, purposes, level, spacing, tolerance, random_order, overwrite_flag = collect_arguments()

    # Init the logger to print to the console.
    #
    dptloggers.init_console_logger(debug=True)

    # Create data configuration file.
    #
    create_data_container(output_path=result_path,
                          image_path=image_filter,
                          mask_path=mask_filter,
                          stat_path=stat_filter,
                          labels=label_list,
                          purpose_ratios=purposes,
                          mask_level=level,
                          mask_spacing=spacing,
                          mask_spacing_tolerance=tolerance,
                          random_item_order=random_order,
                          overwrite=overwrite_flag)

#----------------------------------------------------------------------------------------------------


if __name__ == '__main__':

    main()
