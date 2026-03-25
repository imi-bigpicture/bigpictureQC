"""
This script can print shape, pixel spacing and downsampling information from a multi-resolution image.
"""

import digitalpathology.image.io.imagereader as dptimagereader
import digitalpathology.utils.loggers as dptloggers

import argparse
import logging

#----------------------------------------------------------------------------------------------------

def image_info(image_path, verbose):
    """
    Print image information to the console.

    Args:
        image_path (str): Path of the image.
        verbose (bool): Verbosity flag.

    Raises:
        DigitalPathologyConfigError: Configuration errors.
        DigitalPathologyImageError: Image errors.
    """

    image = dptimagereader.ImageReader(image_path=image_path, spacing_tolerance=0.25, input_channels=None, cache_path=None)

    # Get a logger.
    #
    logger = logging.getLogger(name=__name__)

    logger.info('Number of levels: {count}'.format(count=image.levels))
    logger.info('Number of channels: {count}'.format(count=image.channels))
    logger.info('Data type: {module}.{name}'.format(module=image.dtype.__module__, name=image.dtype.__name__))

    for level in range(image.levels):
        if verbose:
            logger.info('Level {level}:'.format(level=level))
            logger.info('    Shape: {shape}'.format(shape=image.shapes[level]))
            logger.info('    Spacing: {spacing}{measure}'.format(spacing=image.spacings[level], measure=' um' if image.spacings[level] is not None else ''))
            logger.info('    Downsampling: {downsampling}x'.format(downsampling=image.downsamplings[level]))
        else:
            logger.info('[{level}] {shape}; {spacing}{measure}; {downsampling}x'.format(level=level,
                                                                                        shape=image.shapes[level],
                                                                                        spacing=image.spacings[level],
                                                                                        measure=' um' if image.spacings[level] is not None else '',
                                                                                        downsampling=image.downsamplings[level]))

    image.close()

#----------------------------------------------------------------------------------------------------

def collect_arguments():
    """
    Collect command line arguments.

    Returns:
        (str, bool): The parsed command line arguments: input image path, verbose flag.
    """

    # Configure argument parser.
    #
    argument_parser = argparse.ArgumentParser(description='Print level, shape, pixel spacing and downsampling information of a multi-resolution image.')

    argument_parser.add_argument('-i', '--image',   required=True, type=str, help='input image')
    argument_parser.add_argument('-v', '--verbose', action='store_true',     help='verbosity')

    # Parse arguments.
    #
    arguments = vars(argument_parser.parse_args())

    # Collect arguments.
    #
    parsed_image_path = arguments['image']
    parsed_verbose_flag = arguments['verbose']

    # Print parameters.
    #
    print(argument_parser.description)
    print('Image path: {path}'.format(path=parsed_image_path))
    print('Verbose: {flag}'.format(flag=parsed_verbose_flag))

    return parsed_image_path, parsed_verbose_flag

#----------------------------------------------------------------------------------------------------

def main():
    """Main function."""

    # Collect command line arguments.
    #
    image_file_path, verbose = collect_arguments()

    # Init the logger to print to the console.
    #
    dptloggers.init_console_logger(debug=True)

    # Collect and print image information.
    #
    image_info(image_path=image_file_path, verbose=verbose)

#----------------------------------------------------------------------------------------------------


if __name__ == '__main__':

    main()
