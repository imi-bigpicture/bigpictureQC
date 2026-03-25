"""
This module can combine a whole-slide image and a network output to an overlay image on the selected level.
"""

import digitalpathology.image.processing.conversion as dptconversion
import digitalpathology.utils.foldercontent as dptfoldercontent
import digitalpathology.utils.loggers as dptloggers

import argparse
import os
import sys

#----------------------------------------------------------------------------------------------------

def assemble_jobs(image_path, mask_path, preview_path):
    """
    Assemble (image path, mask path, output path) job triplets.

    Args:
        image_path (str): Image path filter expression.
        mask_path (str): Mask path filter expression with possible {image} replacement key.
        preview_path (str): Preview path expression with possible {image} replacement key.

    Returns:
        list: List of job tuples.
    """

    # Find out operation mode. If all paths are file paths the mode is 'file'.
    #
    result_job_list = []
    if os.path.isfile(image_path):
        # Return a single triplet if the paths were existing files.
        #
        image_base = os.path.splitext(os.path.basename(image_path))[0]
        target_mask_path = mask_path.format(image=image_base)
        target_preview_path = preview_path.format(image=image_base)

        # Add job item to the list.
        #
        job_item = (image_path, target_mask_path, target_preview_path)
        result_job_list.append(job_item)
    else:
        # Replace the image matching string in the mask path to an asterisk to be able to collect all possible mask files.
        #
        mask_wildcard_path = mask_path.format(image='*')

        # Collect all source images and masks and match their file name for processing.
        #
        image_file_path_list = dptfoldercontent.folder_content(folder_path=image_path, recursive=False)
        mask_file_path_list = dptfoldercontent.folder_content(folder_path=mask_wildcard_path, recursive=False)

        # Build file name to path maps.
        #
        image_file_path_map = {os.path.splitext(os.path.basename(image_path_item))[0]: image_path_item for image_path_item in image_file_path_list}
        mask_file_path_map = {os.path.splitext(os.path.basename(mask_path_item))[0]: mask_path_item for mask_path_item in mask_file_path_list}

        # Construct image match expression.
        #
        image_match_base = '{image}' if os.path.isdir(mask_path) else os.path.splitext(os.path.basename(mask_path))[0]
        image_match = image_match_base if 0 <= image_match_base.find('{image}') else '{image}'

        # Assemble list.
        #
        for image_key in image_file_path_map:
            mask_key = image_match.format(image=image_key)
            if mask_key in mask_file_path_map:
                target_preview_path = preview_path.format(image=image_key)

                # Add job item to the list.
                #
                job_item = (image_file_path_map[image_key], mask_file_path_map[mask_key], target_preview_path)
                result_job_list.append(job_item)

    # Return the result list.
    #
    return result_job_list

#----------------------------------------------------------------------------------------------------

def collect_arguments():
    """
    Collect command line arguments.

    Returns:
        (str, str, str, int, float, float, list, float, str, bool, bool): The parsed command line arguments: input path, mask path, preview path, processing level, processing pixel spacing,
            and spacing tolerance, palette, alpha value, target directory path for data copy, keep copied files flag, and the overwrite flag.
    """

    # Configure argument parser.
    #
    argument_parser = argparse.ArgumentParser(description='Create preview image from mr-image and mask.',
                                              epilog='It can either work with file paths and file filter expressions. If filter expressions are set the matching is based on file names. '
                                                     'Use {image} as replacement for image file base names in mask and output path specifications.')

    argument_parser.add_argument('-i', '--image',   required=True, type=str, help='input image')
    argument_parser.add_argument('-m', '--mask',    required=True, type=str, help='mask image')
    argument_parser.add_argument('-p', '--preview', required=True, type=str, help='output image')

    argument_group = argument_parser.add_mutually_exclusive_group(required=True)
    argument_group.add_argument('-l', '--level',   type=int,   default=None, help='processing image level')
    argument_group.add_argument('-s', '--spacing', type=float, default=None, help='processing pixel spacing (micrometer)')

    argument_parser.add_argument('-t', '--tolerance',      required=False, type=float, default=0.25, help='pixel spacing tolerance (percentage)')
    argument_parser.add_argument('-e', '--palette',        required=False, type=str,   default=None, help='color palette list')
    argument_parser.add_argument('-a', '--alpha',          required=False, type=float, default=0.2,  help='overlay alpha value')
    argument_parser.add_argument('-c', '--copy_directory', required=False, type=str,   default=None, help='data copy target directory path')
    argument_parser.add_argument('-k', '--keep_copies',    action='store_true',                      help='keep copied image files')
    argument_parser.add_argument('-w', '--overwrite',      action='store_true',                      help='overwrite existing results')

    # Parse arguments.
    #
    arguments = vars(argument_parser.parse_args())

    # Collect arguments.
    #
    parsed_image_path = arguments['image']
    parsed_mask_path = arguments['mask']
    parsed_preview_path = arguments['preview']
    parsed_level = arguments['level']
    parsed_pixel_spacing = arguments['spacing']
    parsed_spacing_tolerance = arguments['tolerance']
    parsed_preview_palette_str = arguments['palette']
    parsed_preview_alpha = arguments['alpha']
    parsed_copy_directory = arguments['copy_directory']
    parsed_keep_copies = arguments['keep_copies']
    parsed_overwrite = arguments['overwrite']

    # Convert palette descriptor to map.
    #
    parsed_preview_palette = eval(parsed_preview_palette_str) if parsed_preview_palette_str is not None else None

    # Print parameters.
    #
    print(argument_parser.description)
    print('Input path: {path}'.format(path=parsed_image_path))
    print('Mask path: {path}'.format(path=parsed_mask_path))
    print('Preview path: {path}'.format(path=parsed_preview_path))

    if parsed_pixel_spacing is not None:
        print('Processing pixel spacing: {spacing} um'.format(spacing=parsed_pixel_spacing))
        print('Pixel spacing tolerance: {tolerance}'.format(tolerance=parsed_spacing_tolerance))
    else:
        print('Processing level: {level}'.format(level=parsed_level))

    print('Palette: {palette}'.format(palette=parsed_preview_palette))
    print('Alpha: {alpha}'.format(alpha=parsed_preview_alpha))
    print('Copy directory path: {path}'.format(path=parsed_copy_directory))
    print('Keep copied files: {flag}'.format(flag=parsed_keep_copies))
    print('Overwrite existing results: {flag}'.format(flag=parsed_overwrite))

    return (parsed_image_path,
            parsed_mask_path,
            parsed_preview_path,
            parsed_level,
            parsed_pixel_spacing,
            parsed_spacing_tolerance,
            parsed_preview_palette,
            parsed_preview_alpha,
            parsed_copy_directory,
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
    (image_path,
     mask_path,
     preview_path,
     processing_level,
     processing_pixel_spacing,
     spacing_tolerance,
     preview_palette,
     preview_alpha,
     copy_directory,
     keep_copied_files,
     overwrite) = collect_arguments()

    # Assemble job triplets: (image, mask, preview).
    #
    job_list = assemble_jobs(image_path=image_path, mask_path=mask_path, preview_path=preview_path)

    # Check if there are any identified jobs.
    #
    if job_list:
        # Init the logger to print to the console.
        #
        dptloggers.init_console_logger(debug=True)

        # Calculate preview with the constructed palette.
        #
        successful_items, failed_items = dptconversion.calculate_preview_batch(job_list=job_list,
                                                                               level=processing_level,
                                                                               pixel_spacing=processing_pixel_spacing,
                                                                               spacing_tolerance=spacing_tolerance,
                                                                               alpha=preview_alpha,
                                                                               palette=preview_palette,
                                                                               copy_path=copy_directory,
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
