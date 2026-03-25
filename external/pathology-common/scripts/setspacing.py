"""
This module can set spacing to a mask image.
"""

import digitalpathology.image.io.imagereader as dptimagereader
import digitalpathology.image.io.imagewriter as dptimagewriter
import digitalpathology.utils.loggers as dptloggers

import argparse
import logging
import datetime
import time
import os

#----------------------------------------------------------------------------------------------------

def set_spacing(input_path, output_path, pixel_spacing, overwrite):
    """
    Set the pixel spacing in a multi-resolution image.

    Args:
        input_path (str): Input image path.
        output_path (str): Output image path.
        pixel_spacing (float): Pixel spacing on the lowest level.
        overwrite (bool): If true existing targets will be overwritten.

    Raises:
        DigitalPathologyConfigError: Configuration errors.
        DigitalPathologyImageError: Image errors.
    """

    # Start time measurement.
    #
    start_time = time.time()

    # Print progress.
    #
    logger = logging.getLogger(name=__name__)

    logger.info('Writing: {path}'.format(path=output_path))

    # Hardcoded values.
    #
    tile_size = 512

    # Check if the target image already exits.
    #
    if not os.path.isfile(output_path) or overwrite:
        # Open image.
        #
        image_reader = dptimagereader.ImageReader(image_path=input_path, spacing_tolerance=0.25, input_channels=None, cache_path=None)

        # Fix the spacing if is completely missing.
        #
        if any(level_spacing is None for level_spacing in image_reader.spacings):
            image_reader.correct(spacing=1.0, level=0)

        # Configure the image writer.
        #
        image_writer = dptimagewriter.ImageWriter(image_path=output_path,
                                                  shape=image_reader.shapes[0],
                                                  spacing=pixel_spacing,
                                                  dtype=image_reader.dtype,
                                                  coding=image_reader.coding,
                                                  compression=None,
                                                  interpolation=None,
                                                  tile_size=tile_size,
                                                  jpeg_quality=None,
                                                  empty_value=0,
                                                  skip_empty=None,
                                                  cache_path=None)

        # Re-compress the input image tile by tile.
        #
        input_shape = image_reader.shapes[0]

        for row in range(0, input_shape[0], tile_size):
            for col in range(0, input_shape[1], tile_size):
                image_writer.write(tile=image_reader.read(spacing=image_reader.spacings[0], row=row, col=col, height=tile_size, width=tile_size), row=row, col=col)

        # Finalize the output image and close the input.
        #
        image_writer.close()
        image_reader.close()

        # Report execution time.
        #
        execution_time = time.time() - start_time
        logger.debug('Done in {delta}'.format(delta=datetime.timedelta(seconds=execution_time)))

    else:
        logger.info('Skipping, target file already exits: {path}'.format(path=output_path))

#----------------------------------------------------------------------------------------------------

def collect_arguments():
    """
    Collect command line arguments.

    Returns:
        (str, str, float, bool): The parsed command line arguments: input, output image paths, pixel spacing, and the overwrite flag.
    """

    # Configure argument parser.
    #
    argument_parser = argparse.ArgumentParser(description='Add missing or overwrite existing spacing in a multi-resolution image.')

    argument_parser.add_argument('-i', '--input',     required=True,  type=str,   help='input image')
    argument_parser.add_argument('-o', '--output',    required=True,  type=str,   help='output image')
    argument_parser.add_argument('-s', '--spacing',   required=False, type=float, help='pixel spacing')
    argument_parser.add_argument('-w', '--overwrite', action='store_true',        help='overwrite existing result')

    # Parse arguments.
    #
    arguments = vars(argument_parser.parse_args())

    # Collect arguments.
    #
    parsed_input_path = arguments['input']
    parsed_output_path = arguments['output']
    parsed_spacing = arguments['spacing']
    parsed_overwrite = arguments['overwrite']

    # Print parameters.
    #
    print(argument_parser.description)
    print('Input image: {path}'.format(path=parsed_input_path))
    print('Output image: {path}'.format(path=parsed_output_path))
    print('Pixel spacing: {spacing}'.format(spacing=parsed_spacing))
    print('Overwrite existing result: {flag}'.format(flag=parsed_overwrite))

    return parsed_input_path, parsed_output_path, parsed_spacing, parsed_overwrite

#----------------------------------------------------------------------------------------------------

def main():
    """Main function."""

    # Collect command line arguments.
    #
    input_image_path, output_image_path, required_pixel_spacing, overwrite_flag = collect_arguments()

    # Init the logger to print to the console.
    #
    dptloggers.init_console_logger(debug=True)

    # Set spacing in the image.
    #
    set_spacing(input_path=input_image_path, output_path=output_image_path, pixel_spacing=required_pixel_spacing, overwrite=overwrite_flag)

#----------------------------------------------------------------------------------------------------


if __name__ == '__main__':

    main()
