"""
Save image at a different level.
"""

import digitalpathology.image.processing.zoom as dptzoom
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
        (str, str, int, float, float, int, str, bool, bool): The parsed command line arguments: input and output image paths, the processing level, processing pixel spacing and pixel
            spacing tolerance, JPEG quality setting work, directory path, keep copied files flag, and the overwrite flag.
    """

    # Configure argument parser.
    #
    argument_parser = argparse.ArgumentParser(description='Save mr-image at the required level.',
                                              epilog='Use {image} as replacement for output file name specification.')

    argument_parser.add_argument('-i', '--input',  required=True, type=str, help='input image')
    argument_parser.add_argument('-o', '--output', required=True, type=str, help='output image')

    argument_group = argument_parser.add_mutually_exclusive_group(required=True)
    argument_group.add_argument('-l', '--level',   type=int,   default=None, help='processing image level')
    argument_group.add_argument('-s', '--spacing', type=float, default=None, help='processing pixel spacing (micrometer)')

    argument_parser.add_argument('-t',  '--tolerance',      required=False, type=float, default=0.25, help='pixel spacing tolerance (percentage)')
    argument_parser.add_argument('-q',  '--quality',        required=False, type=int,   default=None, help='JPEG quality (1-100), if JPEG compression is used')
    argument_parser.add_argument('-wd', '--work_directory', required=False, type=str,   default=None, help='intermediate work directory path')
    argument_parser.add_argument('-k',  '--keep_copies',    action='store_true',                      help='keep copied image files')
    argument_parser.add_argument('-w',  '--overwrite',      action='store_true',                      help='overwrite existing results')

    # Parse arguments.
    #
    arguments = vars(argument_parser.parse_args())

    # Collect arguments.
    #
    parsed_input_path = arguments['input']
    parsed_output_path = arguments['output']
    parsed_level = arguments['level']
    parsed_pixel_spacing = arguments['spacing']
    parsed_spacing_tolerance = arguments['tolerance']
    parsed_jpeg_quality = arguments['quality']
    parsed_work_directory = arguments['work_directory']
    parsed_keep_copies = arguments['keep_copies']
    parsed_overwrite = arguments['overwrite']

    # Print parameters.
    #
    print(argument_parser.description)
    print('Input image: {input_path}'.format(input_path=parsed_input_path))
    print('Output image: {output_path}'.format(output_path=parsed_output_path))

    if parsed_pixel_spacing is not None:
        print('Processing pixel spacing: {spacing} um'.format(spacing=parsed_pixel_spacing))
        print('Pixel spacing tolerance: {tolerance}'.format(tolerance=parsed_spacing_tolerance))
    else:
        print('Processing level: {level}'.format(level=parsed_level))

    print('JPEG quality: {quality}'.format(quality=parsed_jpeg_quality))
    print('Work directory path: {path}'.format(path=parsed_work_directory))
    print('Keep copied files: {flag}'.format(flag=parsed_keep_copies))
    print('Overwrite existing results: {overwrite}'.format(overwrite=parsed_overwrite))

    return parsed_input_path, parsed_output_path, parsed_level, parsed_pixel_spacing, parsed_spacing_tolerance, parsed_jpeg_quality, parsed_work_directory, parsed_keep_copies, parsed_overwrite

#----------------------------------------------------------------------------------------------------

def main():
    """
    Main function.

    Returns:
        int: Error code.
    """

    # Collect command line arguments.
    #
    input_path, output_path, level, spacing, tolerance, jpeg_quality, work_directory, keep_copied_files, overwrite = collect_arguments()

    # Assemble job pairs: (input path, output path).
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
        successful_items, failed_items = dptzoom.save_image_at_level_batch(job_list=job_list,
                                                                           level=level,
                                                                           pixel_spacing=spacing,
                                                                           spacing_tolerance=tolerance,
                                                                           jpeg_quality=jpeg_quality,
                                                                           work_path=work_directory,
                                                                           clear_cache=not keep_copied_files,
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

    # Return error code.
    #
    sys.exit(main())
