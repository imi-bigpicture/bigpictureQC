"""
This module can combine mask images by applying arithmetic operation on them.
"""

import digitalpathology.image.processing.arithmetic as dptarithmetic
import digitalpathology.utils.foldercontent as dptfoldercontent
import digitalpathology.utils.loggers as dptloggers

import argparse
import os
import sys
import re

#----------------------------------------------------------------------------------------------------

def assemble_jobs(base_expression, left_path, right_path, result_path, collect_singles):
    """
    Assemble (left path, right path, result path) job triplets for arithmetic operation on masks.

    Args:
        base_expression (str): File name base part.
        left_path (str): Left part of operation.
        right_path (str): Right part of operation.
        result_path (str): Result of operation.
        collect_singles (bool): Create job items with a single participant for renaming.

    Returns:
        list: List of job tuples.
    """

    # Try to construct actual paths.
    #
    left_wildcard_path = left_path.format(base=base_expression)
    right_wildcard_path = right_path.format(base=base_expression)
    result_wildcard_path = result_path.format(base=base_expression)

    # Find out operation mode. If all paths are file paths the mode is 'file'.
    #
    result_job_list = []
    if os.path.isfile(left_wildcard_path) and os.path.isfile(right_wildcard_path):
        # Return a single triplet if the paths were existing files.
        #
        result_job_list.append((left_wildcard_path, right_wildcard_path, result_wildcard_path))
    else:
        # Replace the image matching string in the mask path to an asterisk to be able to collect all possible annotation files.
        #
        left_wildcard_path = left_path.format(base=base_expression)
        right_wildcard_path = right_path.format(base=base_expression)

        # Construct regular expressions for matching files.
        #
        left_regexp = re.compile(pattern=os.path.basename(left_path).format(base='(?P<base>.*)'))
        result_regexp = re.compile(pattern='{base}')
        right_basename = os.path.basename(right_path)

        # Collect all source images and masks and match their file name for processing.
        #
        left_file_path_list = dptfoldercontent.folder_content(folder_path=left_wildcard_path, recursive=False)
        right_file_path_list = dptfoldercontent.folder_content(folder_path=right_wildcard_path, recursive=False)

        # Build file name to path maps.
        #
        left_file_path_map = {os.path.basename(path_item): path_item for path_item in left_file_path_list}
        right_file_path_map = {os.path.basename(path_item): path_item for path_item in right_file_path_list}

        # Assemble list.
        #
        for left_key in left_file_path_map:
            left_base = left_regexp.match(left_key).group('base')
            right_key = right_basename.format(base=left_base)
            actual_result_path = result_regexp.sub(repl=left_base, string=result_path)

            if right_key in right_file_path_map or collect_singles:
                result_job_list.append((left_file_path_map[left_key], right_file_path_map.get(right_key, None), actual_result_path))

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
        (str, str, str, str, str, bool, bool): The parsed command line arguments: base part, left path, right path, result path, operand, overwrite flag and rename singles flag.
    """

    # Configure argument parser.
    #
    argument_parser = argparse.ArgumentParser(description='Apply arithmetic operation on mask images.',
                                              epilog='It can either work with file paths and file filter expressions. If filter expressions are set the matching is based on file names. '
                                                     'Use {base} as replacement for mask image file base names in left, right and result mask file name specification. The {base} will be '
                                                     'replaced in the left part first and the actual match value will be used in the right and result paths.')

    argument_parser.add_argument('-b', '--base',      required=False, type=str, default='', help='base part for matching')
    argument_parser.add_argument('-l', '--left',      required=True,  type=str,             help='left side of the operation')
    argument_parser.add_argument('-r', '--right',     required=True,  type=str,             help='right side of the operation')
    argument_parser.add_argument('-e', '--result',    required=True,  type=str,             help='result of the operation')
    argument_parser.add_argument('-o', '--operand',   required=True,  type=str,             help='operand: \'+\', \'-\' or \'*\'')
    argument_parser.add_argument('-w', '--overwrite', action='store_true',                  help='overwrite existing results')
    argument_parser.add_argument('-s', '--singles',   action='store_true',                  help='rename single items to result')

    # Parse arguments.
    #
    arguments = vars(argument_parser.parse_args())

    # Collect arguments.
    #
    parsed_base_part = arguments['base']
    parsed_left_path = arguments['left']
    parsed_right_path = arguments['right']
    parsed_result_path = arguments['result']
    parsed_operand = arguments['operand']
    parsed_overwrite = arguments['overwrite']
    parsed_singles = arguments['singles']

    # Print parameters.
    #
    print(argument_parser.description)
    print('Base part: {base_part}'.format(base_part=parsed_base_part))
    print('Left path: {left_path}'.format(left_path=parsed_left_path))
    print('Right path: {right_path}'.format(right_path=parsed_right_path))
    print('Result path: {result_path}'.format(result_path=parsed_result_path))
    print('Operand: \'{operand}\''.format(operand=parsed_operand))
    print('Overwrite existing results: {overwrite}'.format(overwrite=parsed_overwrite))
    print('Rename single files to result: {singles}'.format(singles=parsed_singles))

    # Return parsed values.
    #
    return parsed_base_part, parsed_left_path, parsed_right_path, parsed_result_path, parsed_operand, parsed_overwrite, parsed_singles

#----------------------------------------------------------------------------------------------------

def main():
    """
    Main function.

    Returns:
        int: Error code.
    """

    # Collect command line arguments.
    #
    base_part, left_path, right_path, result_path, operand_id, overwrite_flag, singles_flag = collect_arguments()

    # Assemble job triplets: (left path, right path, result path).
    #
    job_list = assemble_jobs(base_expression=base_part, left_path=left_path, right_path=right_path, result_path=result_path, collect_singles=singles_flag)

    # Check if there are any identified jobs.
    #
    if job_list:
        # Init the logger to print to the console.
        #
        dptloggers.init_console_logger(debug=True)

        # Execute jobs.
        #
        successful_items, failed_items = dptarithmetic.image_arithmetic_batch(job_list=job_list,
                                                                              operand=operand_id,
                                                                              accept_singles=singles_flag,
                                                                              overwrite=overwrite_flag)

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
        print('No images matched the input filter')

        error_code = -1

    return error_code

#----------------------------------------------------------------------------------------------------


if __name__ == '__main__':

    sys.exit(main())
