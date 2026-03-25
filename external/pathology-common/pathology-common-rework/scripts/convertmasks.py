"""
This module can convert mask files to ASAP annotations.
"""

import digitalpathology.image.processing.conversion as dptconversion
import digitalpathology.utils.foldercontent as dptfoldercontent
import digitalpathology.utils.loggers as dptloggers

import argparse
import os
import sys

#----------------------------------------------------------------------------------------------------

def assemble_jobs(mask_path, annotation_path):
    """
    Assemble job tuples.

    Args:
        mask_path (str): Mask path expression.
        annotation_path (str): Annotation path filter expression with possible {mask} replacement keys.

    Returns:
        list: List of job (input mask path, target annotation path) tuples.
    """

    # Find out operation mode. If all paths are file paths the mode is 'file'.
    #
    result_job_list = []
    if os.path.isfile(mask_path):
        # Return a single pair if the paths were existing files.
        #
        mask_base = os.path.splitext(os.path.basename(mask_path))[0]
        target_annotation_path = annotation_path.format(mask=mask_base)

        # Add the assembled job item to the list.
        #
        result_job_list.append((mask_path, target_annotation_path))
    else:
        # Collect all source images and annotations and match their file name for processing.
        #
        mask_file_path_list = dptfoldercontent.folder_content(folder_path=mask_path, recursive=False)

        # Assemble list.
        #
        for mask_path_item in mask_file_path_list:
            # Calculate output path.
            #
            mask_base = os.path.splitext(os.path.basename(mask_path_item))[0]
            target_annotation_path = annotation_path.format(mask=mask_base)

            # Add the assembled job item to the list.
            #
            result_job_list.append((mask_path_item, target_annotation_path))

        # Print match count for checking.
        #
        print('Matching mask count: {match_count}'.format(match_count=len(result_job_list)))

    # Return the result list.
    #
    return result_job_list

#----------------------------------------------------------------------------------------------------

def collect_arguments():
    """
    Collect command line arguments.

    Returns:
        (str, str, dict, float, float, float, float, bool, bool): The parsed command line arguments: mask path, output annotation path, label grouping, conversion pixel spacing, target pixel
            spacing, pixel spacing tolerance, RDP epsilon, keep single points flag, and the overwrite flag.
    """

    # Configure argument parser.
    #
    argument_parser = argparse.ArgumentParser(description='Convert region map to ASAP annotation.',
                                              epilog='It can either work with file paths and file filter expressions. If filter expressions are set the matching is based on file names. '
                                                     'Use {mask} as replacement for mask image file base name in output annotation file name specification.')

    argument_parser.add_argument('-m', '--mask',               required=True,  type=str,                 help='input mask image or directory')
    argument_parser.add_argument('-a', '--annotation',         required=True,  type=str,                 help='output annotation xml or directory')
    argument_parser.add_argument('-g', '--grouping',           required=False, type=str,   default=None, help='annotation group to label list mapping')
    argument_parser.add_argument('-c', '--conversion_spacing', required=True,  type=float,               help='pixel spacing of processing of the input mask (micrometer)')
    argument_parser.add_argument('-r', '--target_spacing',     required=False, type=float, default=None, help='target pixel spacing for saving the annotations (micrometer)')
    argument_parser.add_argument('-t', '--tolerance',          required=False, type=float, default=0.25, help='pixel spacing tolerance (percentage)')
    argument_parser.add_argument('-e', '--epsilon',            required=False, type=float, default=1.0,  help='RDP epsilon')
    argument_parser.add_argument('-p', '--singles',            action='store_true',                      help='keep single points (Dot annotations)')
    argument_parser.add_argument('-w', '--overwrite',          action='store_true',                      help='overwrite existing results')

    # Parse arguments.
    #
    arguments = vars(argument_parser.parse_args())

    # Collect arguments.
    #
    parsed_mask_path = arguments['mask']
    parsed_annotation_path = arguments['annotation']
    parsed_grouping_str = arguments['grouping']
    parsed_conversion_pixel_spacing = arguments['conversion_spacing']
    parsed_target_pixel_spacing = arguments['target_spacing']
    parsed_spacing_tolerance = arguments['tolerance']
    parsed_epsilon = arguments['epsilon']
    parsed_singles = arguments['singles']
    parsed_overwrite = arguments['overwrite']

    # Evaluate grouping descriptor.
    #
    parsed_grouping = eval(parsed_grouping_str) if parsed_grouping_str is not None else None

    # Print parameters.
    #
    print(argument_parser.description)
    print('Input mask: {path}'.format(path=parsed_mask_path))
    print('Output annotation: {path}'.format(path=parsed_annotation_path))
    print('Label groups: {map}'.format(map=parsed_grouping))
    print('Conversion pixel spacing: {spacing} um'.format(spacing=parsed_conversion_pixel_spacing))
    print('Target pixel spacing: {spacing}{measure}'.format(spacing=parsed_target_pixel_spacing, measure=' um' if parsed_target_pixel_spacing is not None else ''))
    print('Pixel spacing tolerance: {tolerance}'.format(tolerance=parsed_spacing_tolerance))
    print('RDP epsilon: {value}'.format(value=parsed_epsilon))
    print('Keep single point annotations: {flag}'.format(flag=parsed_singles))
    print('Overwrite existing results: {flag}'.format(flag=parsed_overwrite))

    return (parsed_mask_path,
            parsed_annotation_path,
            parsed_grouping,
            parsed_conversion_pixel_spacing,
            parsed_target_pixel_spacing,
            parsed_spacing_tolerance,
            parsed_epsilon,
            parsed_singles,
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
    mask_path, annotation_path, grouping, conversion_pixel_spacing, target_pixel_spacing, spacing_tolerance, epsilon, singles, overwrite = collect_arguments()

    # Assemble job pairs: (input mask path, output annotation path).
    #
    job_list = assemble_jobs(mask_path=mask_path, annotation_path=annotation_path)

    # Check if there are any identified jobs.
    #
    if job_list:
        # Init the logger to print to the console.
        #
        dptloggers.init_console_logger(debug=True)

        # Execute jobs.
        #
        successful_items, failed_items = dptconversion.create_mask_annotation_batch(job_list=job_list,
                                                                                    label_map=grouping,
                                                                                    conversion_spacing=conversion_pixel_spacing,
                                                                                    target_spacing=target_pixel_spacing,
                                                                                    spacing_tolerance=spacing_tolerance,
                                                                                    keep_singles=singles,
                                                                                    rdp_epsilon=epsilon,
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
        print('No images matched the input filter')

        error_code = -1

    return error_code

#----------------------------------------------------------------------------------------------------


if __name__ == '__main__':

    sys.exit(main())
