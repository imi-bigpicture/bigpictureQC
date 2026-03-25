"""
This module can convert ASAP annotation descriptor XMLs to mask files.
"""

import digitalpathology.image.processing.conversion as dptconversion
import digitalpathology.utils.foldercontent as dptfoldercontent
import digitalpathology.utils.loggers as dptloggers

import argparse
import os
import sys

#----------------------------------------------------------------------------------------------------

def assemble_jobs(image_path, annotation_path, mask_path):
    """
    Assemble (image path, annotation path, output path) job triplets.

    Args:
        image_path (str): Image path filter expression.
        annotation_path (str): Annotation path filter expression.
        mask_path (str): Mask path expression with possible {image} and {annotation} replacement keys.

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
        target_annotation_path = annotation_path.format(image=image_base)
        target_mask_path = mask_path.format(image=image_base)

        # Add job item to the list.
        #
        job_item = (image_path, target_annotation_path, target_mask_path)
        result_job_list.append(job_item)
    else:
        # Replace the image matching string in the annotation path to an asterisk to be able to collect all possible annotation files.
        #
        annotation_wildcard_path = annotation_path.format(image='*')

        # Collect all source images and annotations and match their file name for processing.
        #
        image_file_path_list = dptfoldercontent.folder_content(folder_path=image_path, recursive=False)
        annotation_file_path_list = dptfoldercontent.folder_content(folder_path=annotation_wildcard_path, recursive=False)

        # Build file name to path maps.
        #
        image_file_path_map = {os.path.splitext(os.path.basename(image_path_item))[0]: image_path_item for image_path_item in image_file_path_list}
        annotation_file_path_map = {os.path.splitext(os.path.basename(annotation_path_item))[0]: annotation_path_item for annotation_path_item in annotation_file_path_list}

        # Construct image match expression.
        #
        image_match_base = '{image}' if os.path.isdir(annotation_path) else os.path.splitext(os.path.basename(annotation_path))[0]
        image_match = image_match_base if 0 <= image_match_base.find('{image}') else '{image}'

        # Assemble list.
        #
        for image_key in image_file_path_map:
            annotation_key = image_match.format(image=image_key)
            if annotation_key in annotation_file_path_map:
                target_mask_path = mask_path.format(image=image_key, annotation=annotation_key)

                # Add job item to the list.
                #
                job_item = (image_file_path_map[image_key], annotation_file_path_map[annotation_key], target_mask_path)
                result_job_list.append(job_item)

        # Print match count for checking.
        #
        print('Matching annotation count: {match_count}'.format(match_count=len(result_job_list)))

    # Return the result list.
    #
    return result_job_list

#----------------------------------------------------------------------------------------------------

def collect_arguments():
    """
    Collect command line arguments.

    Returns:
        (str, str, str, dict, list, float, float, str, bool, bool, bool, bool): The parsed command line arguments: image path, annotation path, and output mask path, label mapping,
            conversion order, input pixel spacing, pixel spacing tolerance, work directory path, accept all empty groups flag, strict flag, keep copied files flag, and the overwrite flag.
    """

    # Configure argument parser.
    #
    argument_parser = argparse.ArgumentParser(description='Convert ASAP annotation to region map.',
                                              epilog='It can either work with file paths and file filter expressions. If filter expressions are set the matching is based on file names. '
                                                     'Use {image} and {annotation} as replacement for image and annotation file base names in output mask specification and annotation '
                                                     'file name to image file matching.')

    argument_parser.add_argument('-i',  '--image',          required=True,  type=str,                 help='input multi-resolution image or directory')
    argument_parser.add_argument('-a',  '--annotation',     required=True,  type=str,                 help='input annotation xml or directory')
    argument_parser.add_argument('-m',  '--mask',           required=True,  type=str,                 help='output annotation mask or directory')
    argument_parser.add_argument('-l',  '--labels',         required=False, type=str,   default=None, help='annotation group to label mapping')
    argument_parser.add_argument('-o',  '--order',          required=False, type=str,   default=None, help='order of annotation group conversion')
    argument_parser.add_argument('-s',  '--spacing',        required=False, type=float, default=None, help='pixel spacing of conversion of the input image (micrometer)')
    argument_parser.add_argument('-t',  '--tolerance',      required=False, type=float, default=0.25, help='pixel spacing tolerance (percentage)')
    argument_parser.add_argument('-wd', '--work_directory', required=False, type=str,   default=None, help='intermediate work directory path')
    argument_parser.add_argument('-e',  '--empty',          action='store_true',                      help='accept if all groups are empty')
    argument_parser.add_argument('-r',  '--strict',         action='store_true',                      help='stop if unknown annotation group found')
    argument_parser.add_argument('-k',  '--keep_copies',    action='store_true',                      help='keep copied image files')
    argument_parser.add_argument('-w',  '--overwrite',      action='store_true',                      help='overwrite existing results')

    # Parse arguments.
    #
    arguments = vars(argument_parser.parse_args())

    # Collect arguments.
    #
    parsed_image_path = arguments['image']
    parsed_annotation_path = arguments['annotation']
    parsed_mask_path = arguments['mask']
    parsed_label_map_str = arguments['labels']
    parsed_conversion_order_str = arguments['order']
    parsed_pixel_spacing = arguments['spacing']
    parsed_spacing_tolerance = arguments['tolerance']
    parsed_work_directory = arguments['work_directory']
    parsed_empty = arguments['empty']
    parsed_strict = arguments['strict']
    parsed_keep_copies = arguments['keep_copies']
    parsed_overwrite = arguments['overwrite']

    # Evaluate label map and label order descriptors.
    #
    parsed_label_map = eval(parsed_label_map_str) if parsed_label_map_str is not None else None
    parsed_conversion_order = eval(parsed_conversion_order_str) if parsed_conversion_order_str is not None else None

    # Print parameters.
    #
    print(argument_parser.description)
    print('Input image: {path}'.format(path=parsed_image_path))
    print('Input annotation: {path}'.format(path=parsed_annotation_path))
    print('Output mask: {path}'.format(path=parsed_mask_path))
    print('Labels: {map}'.format(map=parsed_label_map))
    print('Order: {order}'.format(order=parsed_conversion_order))
    print('Processing pixel spacing: {spacing}{measure}'.format(spacing=parsed_pixel_spacing, measure=' um' if parsed_pixel_spacing is not None else ''))
    print('Pixel spacing tolerance: {tolerance}'.format(tolerance=parsed_spacing_tolerance))
    print('Work directory path: {path}'.format(path=parsed_work_directory))
    print('Accept if all groups are empty: {flag}'.format(flag=parsed_empty))
    print('Stop on unknown groups: {flag}'.format(flag=parsed_strict))
    print('Keep copied files: {flag}'.format(flag=parsed_keep_copies))
    print('Overwrite existing results: {flag}'.format(flag=parsed_overwrite))

    return (parsed_image_path,
            parsed_annotation_path,
            parsed_mask_path,
            parsed_label_map,
            parsed_conversion_order,
            parsed_pixel_spacing,
            parsed_spacing_tolerance,
            parsed_work_directory,
            parsed_empty,
            parsed_strict,
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
     annotation_path,
     mask_path,
     label_map,
     conversion_order,
     pixel_spacing,
     spacing_tolerance,
     work_directory,
     empty_ok,
     strict,
     keep_copied_files,
     overwrite) = collect_arguments()

    # Assemble job triplets: (image, annotation, result mask).
    #
    job_list = assemble_jobs(image_path=image_path, annotation_path=annotation_path, mask_path=mask_path)

    # Check if there are any identified jobs.
    #
    if job_list:
        # Init the logger to print to the console.
        #
        dptloggers.init_console_logger(debug=True)

        # Execute jobs.
        #
        successful_items, failed_items = dptconversion.create_annotation_mask_batch(job_list=job_list,
                                                                                    label_map=label_map,
                                                                                    conversion_order=conversion_order,
                                                                                    conversion_spacing=pixel_spacing,
                                                                                    spacing_tolerance=spacing_tolerance,
                                                                                    strict=strict,
                                                                                    accept_all_empty=empty_ok,
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
