"""
This module creates YAML files for k-fold cross validation from a given list of images.
"""

import digitalpathology.generator.batch.batchsource as dptbatchsource
import digitalpathology.utils.loggers as dptloggers

import argparse
import logging
import datetime
import time
import os

#----------------------------------------------------------------------------------------------------

def create_folds(image_list_path, output_pattern, number_of_folds, overwrite):
    """
    Distribute the content of a batch source to folds.

    Args:
        image_list_path (str): Input batch source path.
        output_pattern (str): Output patch source pattern with possible {input} and {fold} keys.
        number_of_folds (int): Number of folds.
        overwrite (bool): Overwrite flag.

    Raises:
        ValueError: The fold count is smaller than 1.

        InvalidDataFileExtensionError: The format cannot be derived from the file extension.
        InvalidDataSourceTypeError: Invalid JSON or YAML file.
        PurposeListAndRatioMismatchError: The configured purpose distribution does not match the available purposes.
        InvalidDataFileExtensionError: The format cannot be derived from the file extension.
    """

    # Start time measurement.
    #
    start_time = time.time()

    # Get a logger.
    #
    logger = logging.getLogger(name=__name__)

    # Check fold count.
    #
    if number_of_folds < 1:
        raise ValueError('Fold count ({count}) is smaller than 1.'.format(count=number_of_folds))

    # Construct distribution.
    #
    fold_distribution = {'fold_{index}'.format(index=index): 1.0 for index in range(number_of_folds)}

    # Open input list.
    #
    logger.info('Reading source: {path}'.format(path=image_list_path))

    batch_source = dptbatchsource.BatchSource(source_items=None)
    batch_source.load(file_path=image_list_path)
    batch_source.distribute(purpose_distribution=fold_distribution)

    # Build a per-fold set of items.
    #
    fold_data = []
    for index in range(number_of_folds):
        fold_items = {category_id: batch_source.items(purpose_id='fold_{index}'.format(index=index), category_id=category_id, replace=False) for category_id in batch_source.categories(purpose_id=None)}
        fold_data.append(fold_items)

    # Dump each fold to a separate list file.
    #
    base = os.path.splitext(os.path.basename(image_list_path))[0]
    for index in range(number_of_folds):
        # Construct target path.
        #
        target_file_path = output_pattern.format(input=base, fold=index)

        # Write out fold: check if target exists.
        #
        if not os.path.isfile(target_file_path) or overwrite:

            logger.info('Writing fold {index}: {path}'.format(index=index, path=target_file_path))

            fold_batch_source = dptbatchsource.BatchSource(source_items=None)
            fold_batch_source.update(path_replacements=batch_source.replacements())

            for add_index in range(number_of_folds):
                add_purpose = 'training' if add_index != index else 'validation'
                fold_batch_source.push(source_items=fold_data[add_index], purpose_id=add_purpose)

            fold_batch_source.save(file_path=target_file_path)

        else:
            logger.info('Target of fold {index} already exits: {path}'.format(index=index, path=target_file_path))

    # Report execution time.
    #
    execution_time = time.time() - start_time
    logger.debug('Done in {delta}'.format(delta=datetime.timedelta(seconds=execution_time)))

#----------------------------------------------------------------------------------------------------

def collect_arguments():
    """
    Collect command line arguments.

    Returns:
        str, str, int, bool: The parsed command line arguments: input batch source, output batch source pattern, fold count and overwrite flag.
    """

    # Configure argument parser.
    #
    argument_parser = argparse.ArgumentParser(description='Creates YAML files for k-fold cross validation from a given list of images.',
                                              epilog='It distributes a YAML batch source list file to the given folds. Use {input} for input file basename'
                                                     'and {fold} for fold index replacement.')

    argument_parser.add_argument('-i', '--input',     required=True, type=str, help='input batch source')
    argument_parser.add_argument('-o', '--output',    required=True, type=str, help='output batch sources')
    argument_parser.add_argument('-f', '--folds',     required=True, type=int, help='number of data folds')
    argument_parser.add_argument('-w', '--overwrite', action='store_true',     help='overwrite existing results')

    # Parse arguments.
    #
    arguments = vars(argument_parser.parse_args())

    # Collect arguments.
    #
    parsed_input_path = arguments['input']
    parsed_output_path = arguments['output']
    parsed_fold_count = arguments['folds']
    parsed_overwrite_flag = arguments['overwrite']

    # Print parameters.
    #
    print(argument_parser.description)
    print('Input batch source path: {path}'.format(path=parsed_input_path))
    print('Output batch source path: {path}'.format(path=parsed_output_path))
    print('Number of folds: {count}'.format(count=parsed_fold_count))
    print('Overwrite existing results: {flag}'.format(flag=parsed_overwrite_flag))

    return parsed_input_path, parsed_output_path, parsed_fold_count, parsed_overwrite_flag

#----------------------------------------------------------------------------------------------------

def main():
    """Main function."""

    # Collect arguments.
    #
    input_path, output_path, fold_count, overwrite_flag = collect_arguments()

    # Init the logger to print to the console.
    #
    dptloggers.init_console_logger(debug=True)

    # Distribute images among folds.
    #
    create_folds(image_list_path=input_path, output_pattern=output_path, number_of_folds=fold_count, overwrite=overwrite_flag)

#----------------------------------------------------------------------------------------------------


if __name__ == '__main__':

    main()
