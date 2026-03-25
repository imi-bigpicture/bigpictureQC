"""
This module contains a function that collects images and annotation from a table to a data set.
"""

import digitalpathology.errors.imageerrors as dptimageerrors
import digitalpathology.utils.anonymize as dptanonymize
import digitalpathology.utils.imagefile as dptimagefile
import digitalpathology.utils.loggers as dptloggers
import digitalpathology.image.processing.zoom as dptzoom

import pandas as ps
import argparse
import logging
import os
import sys

#----------------------------------------------------------------------------------------------------

def collect_single_image(image_source_path,
                         image_target_folder_path,
                         annotation_source_path,
                         annotation_target_folder_path,
                         target_basename,
                         index,
                         level,
                         pixel_spacing,
                         spacing_tolerance,
                         jpeg_quality,
                         anonymize,
                         convert,
                         overwrite):
    """
    Collect a single image: copy, anonymize, convert and rename it.

    Args:
        image_source_path (str): Source image path.
        image_target_folder_path (str): Target image folder path.
        annotation_source_path (str, None): Annotation source path.
        annotation_target_folder_path (str, None): Target annotation folder path.
        target_basename (str, None): Target basename for renaming the image and the annotation.
        index (tuple): Tuple of (current, total) processed images.
        level (int, None): Collection level of the image.
        pixel_spacing (float, None): Collection pixel spacing of the image to process. (micrometer).
        spacing_tolerance (float): Pixel spacing tolerance (percentage).
        jpeg_quality (int, None): JPEG quality (1-100) when using JPEG as compression method. If not set, the default of ImageWriter is used.
        anonymize (bool): Anonymization flag.
        convert (bool): Image conversion to TIFF format flag.
        overwrite (bool): Overwrite flag.

    Raises:
        ValueError: The target file name for conversion is not a TIFF file name.
        ValueError: The source file is renamed to a different file format extension.

        DigitalPathologyConfigError: Configuration errors.
        DigitalPathologyImageError: Image errors.
        DigitalPathololgyProcessingError: Processing errors.
    """

    # Log progress.
    #
    logger = logging.getLogger(name=__name__)

    logger.info('Processing [{index}/{count}]: {name}'.format(index=index[0], count=index[1], name=os.path.basename(image_source_path)))
    # Normalize paths.
    #
    norm_image_source_path = os.path.normpath(image_source_path)
    image_basename = os.path.basename(norm_image_source_path)
    image_base, image_extentsion = os.path.splitext(image_basename)

    # Check if conversion needed. It can be the conversion flag or the fact that the images are saved at a given level or pixel spacing.
    #
    conversion_enabled = convert or level is not None or pixel_spacing is not None

    # Construct the target paths.
    #
    if target_basename:
        if os.path.splitext(target_basename)[1]:
            final_target_basename = target_basename
        else:
            final_target_basename_ext = '.tif' if conversion_enabled else image_extentsion
            final_target_basename = '{base}{ext}'.format(base=target_basename, ext=final_target_basename_ext)
    else:
        final_target_basename_ext = '.tif' if conversion_enabled else image_extentsion
        final_target_basename = '{base}{ext}'.format(base=image_base, ext=final_target_basename_ext)

    norm_image_target_path = os.path.join(os.path.normpath(image_target_folder_path), final_target_basename)

    # Check if the final target image path (after anonymizing, renaming or converting) exists.
    #
    if not os.path.exists(norm_image_target_path) or overwrite:
        # Check if conversion is configured.
        #
        tiff_extensions = ('.tif', '.tiff')
        if conversion_enabled:
            # Conversion is necessary. Check it the target extension for TIFF conversion is correct.
            #
            if target_basename and os.path.splitext(norm_image_target_path)[1].lower() not in tiff_extensions:
                raise ValueError('Invalid extension for conversion: {name}'.format(name=target_basename))

            # Check if collecting an image at a given level or pixel spacing is necessary. That means separate anonymization and conversion steps
            # are not necessary since the internal TIFF writer library ignore everything except pixel data and can only write TIFF file format.
            #
            if level is None and pixel_spacing is None:
                # Check if conversion is necessary. Copy or convert the image and anonymize it.
                #
                if image_extentsion.lower() not in tiff_extensions:
                    if anonymize:
                        intermediate_target_path = os.path.join(os.path.normpath(image_target_folder_path), image_basename)

                        # Copy the image to an intermediate path, anonymize it, convert it to the target path and remove the intermediate image.
                        #
                        dptimagefile.copy_image(source_path=norm_image_source_path, target_path=intermediate_target_path, overwrite=overwrite)
                        dptanonymize.anonymize_image(image_path=intermediate_target_path)

                        dptzoom.save_image_at_level(image=intermediate_target_path,
                                                    output_path=norm_image_target_path,
                                                    level=0,
                                                    pixel_spacing=None,
                                                    spacing_tolerance=spacing_tolerance,
                                                    jpeg_quality=jpeg_quality,
                                                    work_path=None,
                                                    clear_cache=True,
                                                    overwrite=overwrite)

                        dptimagefile.remove_image(image_path=intermediate_target_path)
                    else:
                        # Move the image by directly writing he result of the conversion to the target path.
                        #
                        dptzoom.save_image_at_level(image=norm_image_source_path,
                                                    output_path=norm_image_target_path,
                                                    level=0,
                                                    pixel_spacing=None,
                                                    spacing_tolerance=spacing_tolerance,
                                                    jpeg_quality=jpeg_quality,
                                                    work_path=None,
                                                    clear_cache=True,
                                                    overwrite=overwrite)
                else:
                    # The source image is already TIFF. Just copy it and anonymize it if necessary.
                    #
                    dptimagefile.copy_image(source_path=norm_image_target_path, target_path=norm_image_target_path, overwrite=overwrite)
                    if anonymize:
                        dptanonymize.anonymize_image(image_path=norm_image_target_path)
            else:
                try:
                    # Collect the images on a given level or pixel spacing.
                    #
                    dptzoom.save_image_at_level(image=norm_image_source_path,
                                                output_path=norm_image_target_path,
                                                level=level,
                                                pixel_spacing=pixel_spacing,
                                                spacing_tolerance=spacing_tolerance,
                                                jpeg_quality=jpeg_quality,
                                                work_path=None,
                                                clear_cache=True,
                                                overwrite=overwrite)

                except dptimageerrors.PixelSpacingLevelError:
                    # There is no appropriate level found for the given pixel spacing. Try to find a level with smaller pixel spacing and zoom out to get to the required pixel spacing.
                    #
                    alternative_pixel_spacing = pixel_spacing / 2.0

                    logger.info('The {spacing} um pixel spacing is not available in the source image, trying to zoom out from {alternative} um'.format(spacing=pixel_spacing,
                                                                                                                                                       alternative=alternative_pixel_spacing))

                    dptzoom.zoom_image(image=norm_image_source_path,
                                       output_path=norm_image_target_path,
                                       zoom=0.5,
                                       level=level,
                                       pixel_spacing=alternative_pixel_spacing,
                                       spacing_tolerance=spacing_tolerance,
                                       round_shape=False,
                                       interpolation_order=2,
                                       jpeg_quality=jpeg_quality,
                                       work_path=None,
                                       clear_cache=True,
                                       overwrite=overwrite)
                except Exception:
                    # Some other, not expected error happened.
                    #
                    raise
        else:
            # Check extension changes.
            #
            target_extension = os.path.splitext(norm_image_target_path)[1]
            if image_extentsion.lower() != target_extension.lower() and (image_extentsion.lower() not in tiff_extensions or target_extension.lower() not in tiff_extensions):
                raise ValueError('Extension mismatch for renaming from {source} to {target}.'.format(source=os.path.basename(norm_image_target_path), target=target_basename))

            # Copy the image and the annotation to the target folder and anonymize the image.
            #
            dptimagefile.copy_image(source_path=norm_image_source_path, target_path=norm_image_target_path, overwrite=overwrite)
            if anonymize:
                dptanonymize.anonymize_image(image_path=norm_image_target_path)
    else:
        logger.info('Skipping, target already exists: {path}'.format(path=norm_image_target_path))

    # Handle the annotation file.
    #
    if annotation_source_path:
        # Construct target annotation path.
        #
        norm_annotation_source_path = os.path.normpath(annotation_source_path)
        final_annotation_target_path = os.path.join(os.path.normpath(annotation_target_folder_path), '{base}{ext}'.format(base=os.path.splitext(final_target_basename)[0],
                                                                                                                          ext=os.path.splitext(norm_annotation_source_path)[1]))

        # Copy annotation file.
        #
        dptimagefile.copy_image(source_path=norm_annotation_source_path, target_path=final_annotation_target_path, overwrite=overwrite)

#----------------------------------------------------------------------------------------------------

def collect_data_set(source_path,
                     path_replacements,
                     image_column,
                     annotation_column,
                     target_column,
                     image_path,
                     annotation_path,
                     level,
                     pixel_spacing,
                     spacing_tolerance,
                     jpeg_quality,
                     anonymize,
                     convert,
                     overwrite):
    """
    Organize images and annotations into a data set. The function copies the source images and annotations into a target folder, anonymizes, converts and renames them as configured. If the collection
    level or pixel spacing is set to other than the values corresponding to the intrinsic level (0) the images will be saved to a TIFF file with internal functions. This also means that the images
    will be converted to tiff and anonymized (since the internal functions does not transfer anything else than image data to the new file). For this the output format must be set to TIFF and the
    converter binary path does not have to be set.

    Args:
        source_path (str): CSV table file path.
        path_replacements (dict, None): Path replacements.
        image_column (str): Column of the source image in the table.
        annotation_column (str, None): Column of the annotation file in the table.
        target_column (str, None): Column of the target basename in the file.
        image_path (str): Target image directory path.
        annotation_path (str, None): Target annotation directory path.
        level (int, None): Collection level of the image.
        pixel_spacing (float, None): Collection pixel spacing of the image to process. (micrometer).
        spacing_tolerance (float): Pixel spacing tolerance (percentage).
        jpeg_quality (int, None): JPEG quality (1-100) when using JPEG as compression method. If not set, the default of ImageWriter is used.
        anonymize (bool): Anonymization flag.
        convert (bool): Image conversion to TIFF format flag.
        overwrite (bool): Overwrite flag.

    Returns:
        (list, list): Collection of successful and failed cases.

    Raises:
        ValueError: Image column is not in data frame.
        ValueError: Annotation column is set but not in data frame.
        ValueError: Target column is set but not in data frame.
        ValueError: Annotation column is configured as source but destination folder is not set.
    """

    # Read the source item list.
    #
    item_df = ps.read_csv(source_path)

    # Check if all the columns are in the data frame.
    #
    if image_column not in item_df:
        raise ValueError('Missing image column from the data frame: {column}.'.format(column=image_column))

    if annotation_column is not None and annotation_column not in item_df:
        raise ValueError('Missing annotation column from the data frame: {column}.'.format(column=annotation_column))

    if target_column is not None and target_column not in item_df:
        raise ValueError('Missing target column from the data frame: {column}.'.format(column=target_column))

    # If annotation column is given, the target folder must be also set.
    #
    if annotation_column is not None and annotation_path is None:
        raise ValueError('Annotation source set but destination is empty.')

    # Create logger for printing to the console.
    #
    logger = logging.getLogger(name=__name__)

    # Count the valid entries.
    #
    item_count = 0
    for _, series in item_df.iterrows():
        image_source_path = series[image_column]
        if type(image_source_path) is str:
            item_count += 1

    logger.info('Entries: {count}'.format(count=item_count))

    # Go through the row items and process each image.
    #
    successful_collection = {}
    failed_collection = {}
    processing_index = 0
    for index, series in item_df.iterrows():
        # Extract data from table. Skip empty rows.
        #
        image_source_path = series[image_column]
        if type(image_source_path) is str:
            processing_index += 1

            annotation_source_path = series[annotation_column] if annotation_column is not None else None
            target_basename = series[target_column] if target_column is not None else None

            annotation_source_path = annotation_source_path if type(annotation_source_path) is str else None
            target_basename = target_basename if type(target_basename) is str else None

            # Apply path replacements.
            #
            if path_replacements:
                image_source_path = image_source_path.format(**path_replacements)
                image_path = image_path.format(**path_replacements)

                if annotation_source_path:
                    annotation_source_path = annotation_source_path.format(**path_replacements)
                if annotation_path:
                    annotation_path = annotation_path.format(**path_replacements)

                if target_basename:
                    target_basename = target_basename.format(**path_replacements)

            # Process single image.
            #
            try:
                collect_single_image(image_source_path=image_source_path,
                                     image_target_folder_path=image_path,
                                     annotation_source_path=annotation_source_path,
                                     annotation_target_folder_path=annotation_path,
                                     target_basename=target_basename,
                                     index=(processing_index, item_count),
                                     level=level,
                                     pixel_spacing=pixel_spacing,
                                     spacing_tolerance=spacing_tolerance,
                                     jpeg_quality=jpeg_quality,
                                     anonymize=anonymize,
                                     convert=convert,
                                     overwrite=overwrite)

            except Exception as exception:
                # Add case to the error collection.
                #
                failed_collection[index] = image_source_path
                logger.error('Error: {exception}'.format(exception=exception))

            else:
                # Save the result to the list of successful zooms.
                #
                successful_collection[index] = image_source_path

    # Return a list of successful and failed cases.
    #
    return successful_collection, failed_collection

#----------------------------------------------------------------------------------------------------

def collect_arguments():
    """
    Collect command line arguments.

    Returns:
        (str, dict, str, str, str, str, str, int, float, float, int, bool, bool, bool): Parsed input table path, path replacement map, image column, annotation column, target column,
            image folder path, annotation folder path, collection level, collection pixel spacing, pixel spacing tolerance, JPEG quality setting, anonymize flag, convert flag, and
            overwrite flag.
    """

    # Configure argument parser.
    #
    argument_parser = argparse.ArgumentParser(description='Collect source image files and annotations into a data set.',
                                              epilog='Collect source image files and annotations and copy them to a common place to form a data set. The script can also rename, anonymize '
                                                     'and convert to TIFF the source images. Level or pixel spacing other than the lover is only supported if the output file format is TIFF. '
                                                     'If the images are saved in a non-intrinsic level, the images are converted to TIFF internally and the converter path is not required.')

    argument_parser.add_argument('-b', '--table',             required=True,  type=str,               help='input table file')
    argument_parser.add_argument('-r', '--replacements',      required=False, type=str, default=None, help='path replacements')
    argument_parser.add_argument('-i', '--image_column',      required=True,  type=str,               help='image column')
    argument_parser.add_argument('-a', '--annotation_column', required=False, type=str, default=None, help='annotation column')
    argument_parser.add_argument('-g', '--target_column',     required=False, type=str, default=None, help='target column')
    argument_parser.add_argument('-m', '--image_folder',      required=True,  type=str,               help='image folder path')
    argument_parser.add_argument('-n', '--annotation_folder', required=False, type=str, default=None, help='annotation folder path')

    argument_group = argument_parser.add_mutually_exclusive_group(required=False)
    argument_group.add_argument('-l', '--level',   type=int,   default=None, help='collection image level')
    argument_group.add_argument('-s', '--spacing', type=float, default=None, help='collection pixel spacing (micrometer)')

    argument_parser.add_argument('-t', '--tolerance', required=False, type=float, default=0.25, help='pixel spacing tolerance (percentage)')
    argument_parser.add_argument('-q', '--quality',   required=False, type=int,   default=None, help='JPEG quality (1-100), if JPEG compression is used')
    argument_parser.add_argument('-z', '--anonymize', action='store_true',                      help='anonymize data')
    argument_parser.add_argument('-c', '--convert',   action='store_true',                      help='convert data to TIFF')
    argument_parser.add_argument('-w', '--overwrite', action='store_true',                      help='overwrite existing files')

    # Parse arguments.
    #
    arguments = vars(argument_parser.parse_args())

    # Collect arguments.
    #
    parsed_list_path = arguments['table']
    parsed_replacements_str = arguments['replacements']
    parsed_image_column = arguments['image_column']
    parsed_annotation_column = arguments['annotation_column']
    parsed_target_column = arguments['target_column']
    parsed_image_folder_path = arguments['image_folder']
    parsed_annotation_folder_path = arguments['annotation_folder']
    parsed_level = arguments['level']
    parsed_pixel_spacing = arguments['spacing']
    parsed_spacing_tolerance = arguments['tolerance']
    parsed_jpeg_quality = arguments['quality']
    parsed_anonymize = arguments['anonymize']
    parsed_convert = arguments['convert']
    parsed_overwrite = arguments['overwrite']

    # Evaluate parameters.
    #
    parsed_replacements = eval(parsed_replacements_str) if parsed_replacements_str else None

    # Print parameters.
    #
    print(argument_parser.description)
    print('Input list file path: {path}'.format(path=parsed_list_path))
    print('Path replacements: {map}'.format(map=parsed_replacements))
    print('Image column: {name}'.format(name=parsed_image_column))
    print('Annotation column: {name}'.format(name=parsed_annotation_column))
    print('Target column: {name}'.format(name=parsed_target_column))
    print('Image folder path: {path}'.format(path=parsed_image_folder_path))
    print('Annotation folder path: {path}'.format(path=parsed_annotation_folder_path))

    if parsed_pixel_spacing is not None:
        print('Processing pixel spacing: {spacing} um'.format(spacing=parsed_pixel_spacing))
        print('Pixel spacing tolerance: {tolerance}'.format(tolerance=parsed_spacing_tolerance))
    else:
        print('Processing level: {level}'.format(level=parsed_level))

    print('JPEG quality: {quality}'.format(quality=parsed_jpeg_quality))
    print('Anonymize images: {flag}'.format(flag=parsed_anonymize))
    print('Convert images: {flag}'.format(flag=parsed_convert))
    print('Overwrite existing target: {flag}'.format(flag=parsed_overwrite))

    # Return parsed values.
    #
    return (parsed_list_path,
            parsed_replacements,
            parsed_image_column,
            parsed_annotation_column,
            parsed_target_column,
            parsed_image_folder_path,
            parsed_annotation_folder_path,
            parsed_level,
            parsed_pixel_spacing,
            parsed_spacing_tolerance,
            parsed_jpeg_quality,
            parsed_anonymize,
            parsed_convert,
            parsed_overwrite)

#----------------------------------------------------------------------------------------------------

def main():
    """
    Main function.

    Returns:
        int: Error code.
    """

    # Retrieve command line arguments.
    #
    (list_path,
     replacements,
     source_table_image_column,
     source_table_annotation_column,
     source_table_target_column,
     image_folder_path,
     annotation_folder_path,
     collection_level,
     collection_pixel_spacing,
     collection_spacing_tolerance,
     jpeg_compression_quality,
     anonymize_flag,
     convert_flag,
     overwrite_flag) = collect_arguments()

    # Init the logger to print to the console.
    #
    dptloggers.init_console_logger(debug=True)

    # Organize images to a data set.
    #
    successful_items, failed_items = collect_data_set(source_path=list_path,
                                                      path_replacements=replacements,
                                                      image_column=source_table_image_column,
                                                      annotation_column=source_table_annotation_column,
                                                      target_column=source_table_target_column,
                                                      image_path=image_folder_path,
                                                      annotation_path=annotation_folder_path,
                                                      level=collection_level,
                                                      pixel_spacing=collection_pixel_spacing,
                                                      spacing_tolerance=collection_spacing_tolerance,
                                                      jpeg_quality=jpeg_compression_quality,
                                                      anonymize=anonymize_flag,
                                                      convert=convert_flag,
                                                      overwrite=overwrite_flag)

    # Print the collection of failed cases.
    #
    if failed_items:
        print('Failed on {count} items:'.format(count=len(failed_items)))
        for path in failed_items:
            print('{path}'.format(path=path))

    return len(failed_items)

#----------------------------------------------------------------------------------------------------


if __name__ == '__main__':

    sys.exit(main())
