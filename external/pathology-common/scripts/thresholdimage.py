"""
This module can threshold a given image and save it as a binary.
"""

import digitalpathology.image.processing.threshold as dptthreshold
import digitalpathology.utils.foldercontent as dptfoldercontent
import digitalpathology.utils.loggers as dptloggers

import argparse
import os
import sys

#----------------------------------------------------------------------------------------------------

def assemble_jobs(input_path, output_path):
    """
    Assemble (input image path, output image path) job pairs.

    Args:
        input_path (str): Input image path filter expression.
        output_path (str): Output image path filter expression.

    Returns:
        list: List of job tuples.
    """

    # Find out operation mode. If all paths are file paths the mode is 'file'.
    #
    result_job_list = []
    if os.path.isfile(input_path):
        # Return a single triplet if the paths were existing files.
        #
        image_base = os.path.splitext(os.path.basename(input_path))[0]
        target_output_path = output_path.format(image=image_base)
        result_job_list.append((input_path, target_output_path))
    else:
        # Collect all source images and build file name to path maps.
        #
        image_file_path_list = dptfoldercontent.folder_content(folder_path=input_path, recursive=False)
        image_file_path_map = {os.path.splitext(os.path.basename(image_path_item))[0]: image_path_item for image_path_item in image_file_path_list}

        # Assemble list.
        #
        for image_key in image_file_path_map:
            target_image_path = output_path.format(image=image_key)
            result_job_list.append((image_file_path_map[image_key], target_image_path))

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
        (str, str, float, bool): The parsed command line arguments: input, output image paths, threshold value, the load all flag, and the overwrite flags.
    """

    # Configure argument parser.
    #
    argument_parser = argparse.ArgumentParser(description='Apply low threshold on mr-image.',
                                              epilog='Use {image} as replacement for output file name specification.')

    argument_parser.add_argument('-i', '--input',     required=True, type=str,   help='input image')
    argument_parser.add_argument('-o', '--output',    required=True, type=str,   help='output image')
    argument_parser.add_argument('-t', '--threshold', required=True, type=float, help='low threshold value or list (per channel)')
    argument_parser.add_argument('-w', '--overwrite', action='store_true',       help='overwrite existing results')

    # Parse arguments.
    #
    arguments = vars(argument_parser.parse_args())

    # Collect arguments.
    #
    parsed_input_path = arguments['input']
    parsed_output_path = arguments['output']
    parsed_low_threshold = arguments['threshold']
    parsed_overwrite = arguments['overwrite']

    # Print parameters.
    #
    print(argument_parser.description)
    print('Input image: {path}'.format(path=parsed_input_path))
    print('Output image: {path}'.format(path=parsed_output_path))
    print('Threshold value: {threshold}'.format(threshold=parsed_low_threshold))
    print('Overwrite existing results: {flag}'.format(flag=parsed_overwrite))

    return parsed_input_path, parsed_output_path, parsed_low_threshold, parsed_overwrite

#----------------------------------------------------------------------------------------------------

def main():
    """
    Main function.

    Returns:
        int: Error code.
    """

    # Collect command line arguments.
    #
    input_path, output_path, low_threshold, overwrite = collect_arguments()

    # Assemble job pairs: (mrimage path, image path).
    #
    job_list = assemble_jobs(input_path=input_path, output_path=output_path)

    # Check if there are any identified jobs.
    #
    if job_list:
        # Init the logger to print to the console.
        #
        dptloggers.init_console_logger(debug=True)

        # Execute jobs.
        #
        successful_items, failed_items = dptthreshold.low_threshold_image_batch(job_list=job_list, low_threshold=low_threshold, overwrite=overwrite)

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
