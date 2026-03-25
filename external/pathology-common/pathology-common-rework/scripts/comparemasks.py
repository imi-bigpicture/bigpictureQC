"""
This module can compare to mask label images.
"""

import digitalpathology.image.processing.comparison as dptcomparison
import digitalpathology.utils.loggers as dptloggers

import argparse

#----------------------------------------------------------------------------------------------------

def collect_arguments():
    """
    Collect command line arguments.

    Returns:
        (str, int, str, int, float, float): The parsed command line arguments: reference image path, reference label value, template image path, template label value,
            pixel spacing, and pixel spacing tolerance.
    """

    # Configure argument parser.
    #
    argument_parser = argparse.ArgumentParser(description='Calculate image similarity.')

    argument_parser.add_argument('-i', '--reference_image', required=True,  type=str,                 help='reference image')
    argument_parser.add_argument('-a', '--reference_label', required=True,  type=float,               help='reference label')
    argument_parser.add_argument('-m', '--template_image',  required=True,  type=str,                 help='template image')
    argument_parser.add_argument('-b', '--template_label',  required=True,  type=float,               help='template label')
    argument_parser.add_argument('-s', '--spacing',         required=True,  type=float,               help='pixel spacing of comparison (micrometer)')
    argument_parser.add_argument('-t', '--tolerance',       required=False, type=float, default=0.25, help='pixel spacing tolerance (percentage)')

    # Parse arguments.
    #
    arguments = vars(argument_parser.parse_args())

    # Collect arguments.
    #
    parsed_ref_image_path = arguments['reference_image']
    parsed_ref_label = arguments['reference_label']
    parsed_tpl_image_path = arguments['template_image']
    parsed_tpl_label = arguments['template_label']
    parsed_pixel_spacing = arguments['spacing']
    parsed_spacing_tolerance = arguments['tolerance']

    # Print parameters.
    #
    print(argument_parser.description)
    print('Reference image: {path}'.format(path=parsed_ref_image_path))
    print('Reference label: {label}'.format(label=parsed_ref_label))
    print('Template image: {path}'.format(path=parsed_tpl_image_path))
    print('Template label: {label}'.format(label=parsed_tpl_label))
    print('Pixel spacing: {spacing} um'.format(spacing=parsed_pixel_spacing))
    print('Pixel spacing tolerance: {tolerance}'.format(tolerance=parsed_spacing_tolerance))

    return parsed_ref_image_path, parsed_ref_label, parsed_tpl_image_path, parsed_tpl_label, parsed_pixel_spacing, parsed_spacing_tolerance

#----------------------------------------------------------------------------------------------------

def main():
    """Main function."""

    # Collect command line arguments.
    #
    reference_image_path, reference_label, template_image_path, template_label, spacing, spacing_tolerance = collect_arguments()

    # Init the logger to print to the console.
    #
    dptloggers.init_console_logger(debug=True)

    # Process the image.
    #
    dptcomparison.image_similarity(reference=reference_image_path,
                                   reference_label=reference_label,
                                   template=template_image_path,
                                   template_label=template_label,
                                   pixel_spacing=spacing,
                                   spacing_tolerance=spacing_tolerance)

#----------------------------------------------------------------------------------------------------


if __name__ == '__main__':

    main()
