"""
This module can zoom a given image to the size of the given template image on the desired level.
"""

import digitalpathology.image.processing.zoom as dptzoom
import digitalpathology.utils.foldercontent as dptfoldercontent
import digitalpathology.utils.loggers as dptloggers

import argparse
import os
import sys

#----------------------------------------------------------------------------------------------------

def assemble_jobs(input_path, template_path, result_path):
    """
    Assemble (input path, template path, result path) job triplets for zooming images.

    Args:
        input_path (str): Input file path.
        template_path (str): Path to the template image.
        result_path (str): Result of operation.

    Returns:
        list: List of job tuples.
    """

    # Find out operation mode. If all paths are file paths the mode is 'file'.
    #
    result_job_list = []
    if os.path.isfile(input_path) and os.path.isfile(template_path):
        # Return a single triplet if the paths were existing files.
        #
        result_job_list.append((input_path, template_path, result_path))
    else:
        template_wildcard_path = template_path.format(image='*')

        # Collect all source images and masks and match their file name for processing.
        #
        input_file_path_list = dptfoldercontent.folder_content(folder_path=input_path, recursive=False)
        template_file_path_list = dptfoldercontent.folder_content(folder_path=template_wildcard_path, recursive=False)

        # Build file name to path maps.
        #
        input_file_path_map = {os.path.splitext(os.path.basename(path_item))[0]: path_item for path_item in input_file_path_list}
        template_file_path_map = {os.path.splitext(os.path.basename(path_item))[0]: path_item for path_item in template_file_path_list}

        # Construct image match expression.
        #
        image_match_base = '{image}' if os.path.isdir(template_path) else os.path.splitext(os.path.basename(template_path))[0]
        image_match = image_match_base if 0 <= image_match_base.find('{image}') else '{image}'

        # Assemble list.
        #
        for input_key in input_file_path_map:
            template_key = image_match.format(image=input_key)

            if template_key in template_file_path_map:
                result_job_list.append((input_file_path_map[input_key], template_file_path_map[template_key], result_path.format(image=input_key)))

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
        (str, str, str, int, float, float, int, int, str, bool, bool): The parsed command line arguments: input, template, output image path, template level, template pixel spacing and pixel
            spacing tolerance, interpolation order, JPEG quality setting, work directory path, keep copied files flag, and the overwrite flags.
    """

    # Configure argument parser.
    #
    argument_parser = argparse.ArgumentParser(description='Zoom the mr-image to the resolution of the template image at the required level.',
                                              epilog='It can either work with file paths and file filter expressions. If filter expressions are set the matching is based on file names. '
                                                     'Use {image} as replacement for image file base names in template and output file name to image file matching.')

    argument_parser.add_argument('-i', '--image',    required=True, type=str, help='input image')
    argument_parser.add_argument('-m', '--template', required=True, type=str, help='template image')
    argument_parser.add_argument('-o', '--output',   required=True, type=str, help='output image')

    argument_group = argument_parser.add_mutually_exclusive_group(required=True)
    argument_group.add_argument('-l', '--level',   type=int,   default=None, help='template image level')
    argument_group.add_argument('-s', '--spacing', type=float, default=None, help='template pixel spacing (micrometer)')

    argument_parser.add_argument('-t',  '--tolerance',      required=False, type=float, default=0.25, help='pixel spacing tolerance (percentage)')
    argument_parser.add_argument('-r',  '--order',          required=False, type=int,   default=0,    help='interpolation order')
    argument_parser.add_argument('-q',  '--quality',        required=False, type=int,   default=None, help='JPEG quality (1-100), if JPEG compression is used')
    argument_parser.add_argument('-wd', '--work_directory', required=False, type=str,   default=None, help='intermediate work directory path')
    argument_parser.add_argument('-k',  '--keep_copies',    action='store_true',                      help='keep copied image files')
    argument_parser.add_argument('-w',  '--overwrite',      action='store_true',                      help='overwrite existing results')

    # Parse arguments.
    #
    arguments = vars(argument_parser.parse_args())

    # Collect arguments.
    #
    parsed_input_path = arguments['image']
    parsed_template_path = arguments['template']
    parsed_output_path = arguments['output']
    parsed_template_level = arguments['level']
    parsed_pixel_spacing = arguments['spacing']
    parsed_spacing_tolerance = arguments['tolerance']
    parsed_interpolation_order = arguments['order']
    parsed_jpeg_quality = arguments['quality']
    parsed_work_directory = arguments['work_directory']
    parsed_keep_copies = arguments['keep_copies']
    parsed_overwrite = arguments['overwrite']

    # Print parameters.
    #
    print(argument_parser.description)
    print('Input image: {path}'.format(path=parsed_input_path))
    print('Template image: {path}'.format(path=parsed_template_path))
    print('Output image: {path}'.format(path=parsed_output_path))

    if parsed_pixel_spacing is not None:
        print('Template pixel spacing: {spacing} um'.format(spacing=parsed_pixel_spacing))
        print('Pixel spacing tolerance: {tolerance}'.format(tolerance=parsed_spacing_tolerance))
    else:
        print('Template level: {level}'.format(level=parsed_template_level))

    print('Interpolation order: {order}'.format(order=parsed_interpolation_order))
    print('JPEG quality: {quality}'.format(quality=parsed_jpeg_quality))
    print('Work directory path: {path}'.format(path=parsed_work_directory))
    print('Keep copied files: {flag}'.format(flag=parsed_keep_copies))
    print('Overwrite existing results: {overwrite}'.format(overwrite=parsed_overwrite))

    return (parsed_input_path,
            parsed_template_path,
            parsed_output_path,
            parsed_template_level,
            parsed_pixel_spacing,
            parsed_spacing_tolerance,
            parsed_interpolation_order,
            parsed_jpeg_quality,
            parsed_work_directory,
            parsed_keep_copies,
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
    input_path, template_path, output_path, level, spacing, tolerance, interpolation_order, jpeg_quality, work_directory, keep_copied_files, overwrite = collect_arguments()

    # Assemble job triplets: (input path, template path, output path).
    #
    job_list = assemble_jobs(input_path=input_path, template_path=template_path, result_path=output_path)

    # Check if there are any identified jobs.
    #
    if job_list:
        # Init the logger to print to the console.
        #
        dptloggers.init_console_logger(debug=True)

        # Execute jobs.
        #
        successful_items, failed_items = dptzoom.zoom_image_to_template_batch(job_list=job_list,
                                                                              image_level=0,
                                                                              image_pixel_spacing=None,
                                                                              template_level=level,
                                                                              template_pixel_spacing=spacing,
                                                                              spacing_tolerance=tolerance,
                                                                              interpolation_order=interpolation_order,
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

    sys.exit(main())
