"""
This script can generate a shell script to copy all files to local directory from a configuration JSON file.
"""

import digitalpathology.generator.batch.batchsource as dptbatchsource
import digitalpathology.utils.imagefile as dptimagefile
import digitalpathology.utils.loggers as dptloggers

import argparse
import logging
import datetime
import time

#----------------------------------------------------------------------------------------------------

def copy_files_in_config(data_config_path, source_path_replacements, target_path_replacements, purpose_filter, category_filter, allow_missing_stat, overwrite):
    """
    Copy files in batch source configuration.

    Args:
        data_config_path (str): JSON or YAML configuration file path.
        source_path_replacements (dict): Source path replacements.
        target_path_replacements (dict): Target path replacements.
        purpose_filter (list, None): Purpose filter list.
        category_filter (list, None): Category filter list.
        allow_missing_stat (bool): Allow configured but missing stat file.
        overwrite (bool): Overwrite existing targets.

    Raises:
        InvalidDataFileExtensionError: The format cannot be derived from the file extension.
        InvalidDataSourceTypeError: Invalid JSON or YAML file.
        PurposeListAndRatioMismatchError: The configured purpose distribution does not match the available purposes.
    """

    # Start time measurement.
    #
    start_time = time.time()

    # Get a logger.
    #
    logger = logging.getLogger(name=__name__)

    # Load JSON file to batch source without replacements.
    #
    batch_source = dptbatchsource.BatchSource(source_items=None)
    batch_source.load(file_path=data_config_path)

    # Copy the batch source.
    #
    copied_items, skipped_items = dptimagefile.copy_batch_source(batch_source=batch_source,
                                                                 source_replacements=source_path_replacements,
                                                                 target_replacements=target_path_replacements,
                                                                 purposes=purpose_filter,
                                                                 categories=category_filter,
                                                                 allow_missing_stat=allow_missing_stat,
                                                                 overwrite=overwrite)

    # Report item counts.
    #
    logger.info('Copied items: {count}'.format(count=copied_items))
    logger.info('Skipped items: {count}'.format(count=skipped_items))

    # Report execution time.
    #
    execution_time = time.time() - start_time
    logger.debug('Done in {delta}'.format(delta=datetime.timedelta(seconds=execution_time)))

#----------------------------------------------------------------------------------------------------

def collect_arguments():
    """
    Collect command line arguments.

    Returns:
        (str, dict, dict, list, list, bool, bool): The parsed command line arguments: Data file path, source, and target path replacements, purpose and category filers,
            allow missing stats flag, and overwrite existing flag.
    """

    # Configure argument parser.
    #
    argument_parser = argparse.ArgumentParser(description='Copy all files in the configuration file from the given directories.')

    argument_parser.add_argument('-d', '--data',          required=True,  type=str,               help='data file path to process')
    argument_parser.add_argument('-s', '--source',        required=False, type=str, default=None, help='source path replacements')
    argument_parser.add_argument('-t', '--target',        required=True,  type=str,               help='target path replacements')
    argument_parser.add_argument('-p', '--purposes',      required=False, type=str, default=None, help='purpose list to copy')
    argument_parser.add_argument('-c', '--categories',    required=False, type=str, default=None, help='category list to copy')
    argument_parser.add_argument('-m', '--missing_stats', action='store_true',                    help='allow missing stat files')
    argument_parser.add_argument('-w', '--overwrite',     action='store_true',                    help='overwrite existing results')

    # Parse arguments.
    #
    arguments = vars(argument_parser.parse_args())

    # Collect arguments.
    #
    parsed_data_path = arguments['data']
    parsed_source_map_str = arguments['source']
    parsed_target_map_str = arguments['target']
    parsed_purposes_str = arguments['purposes']
    parsed_categories_str = arguments['categories']
    parsed_missing_stats = arguments['missing_stats']
    parsed_overwrite = arguments['overwrite']

    # Evaluate parsed expressions.
    #
    parsed_source_map = eval(parsed_source_map_str) if parsed_source_map_str else {}
    parsed_target_map = eval(parsed_target_map_str)
    parsed_purposes = eval(parsed_purposes_str) if parsed_purposes_str else None
    parsed_categories = eval(parsed_categories_str) if parsed_categories_str else None

    # Print parameters.
    #
    print(argument_parser.description)
    print('Data path: {path}'.format(path=parsed_data_path))
    print('Source replacements: {map}'.format(map=parsed_source_map))
    print('Target replacements: {map}'.format(map=parsed_target_map))
    print('Purpose filer: {purposes}'.format(purposes=parsed_purposes))
    print('Category filer: {categories}'.format(categories=parsed_categories))
    print('Allow missing stat files: {missing_flag}'.format(missing_flag=parsed_missing_stats))
    print('Overwrite existing targets: {overwrite_flag}'.format(overwrite_flag=parsed_overwrite))

    # Return parsed values.
    #
    return parsed_data_path, parsed_source_map, parsed_target_map, parsed_purposes, parsed_categories, parsed_missing_stats, parsed_overwrite

#----------------------------------------------------------------------------------------------------

def main():
    """Main function."""

    # Collect command line arguments.
    #
    data_path, source_replacements, target_replacements, purposes, categories, allow_missing_stats_flag, overwrite_flag = collect_arguments()

    # Init the logger to print to the console.
    #
    dptloggers.init_console_logger(debug=True)

    # Copy all files.
    #
    copy_files_in_config(data_config_path=data_path,
                         source_path_replacements=source_replacements,
                         target_path_replacements=target_replacements,
                         purpose_filter=purposes,
                         category_filter=categories,
                         allow_missing_stat=allow_missing_stats_flag,
                         overwrite=overwrite_flag)

#----------------------------------------------------------------------------------------------------


if __name__ == '__main__':

    main()
