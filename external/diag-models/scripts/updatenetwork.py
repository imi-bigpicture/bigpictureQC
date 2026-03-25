"""
Updates and old network file to a new format.
"""

import argparse
import logging
import datetime
import time
import joblib
import pickle
import zlib
import sys
import os
import h5py

#----------------------------------------------------------------------------------------------------

def init_console_logger(debug=False):
    """
    Initialize the logger to log all messages to the console.

    Args:
        debug (bool): If true the logging level will be DEBUG, otherwise INFO.
    """

    # Configure logging.
    #
    logger = logging.getLogger(name=None)
    logger.setLevel(level=logging.NOTSET)

    # Clear current handlers.
    #
    logger.handlers.clear()

    # Add console stream log handler.
    #
    console_log_level = logging.DEBUG if debug else logging.INFO
    console_log_entry_format = '%(message)s'
    console_log_formatter = logging.Formatter(fmt=console_log_entry_format)
    console_log_handler = logging.StreamHandler(stream=sys.stdout)
    console_log_handler.setFormatter(fmt=console_log_formatter)
    console_log_handler.setLevel(level=console_log_level)

    logger.addHandler(hdlr=console_log_handler)

#----------------------------------------------------------------------------------------------------

def update_network_format(network):
    """
    Update a network with a possibly missing 'stamp' key and copy the HF5 file content into the network data structure as byte stream. For opening the HF5 file
    the path of the original network file must be injected in the data dictionary under a 'path' key.

    Args:
        network (dict): Network data structure.

    Returns:
        dict: Updated network data structure.
    """

    # Remove 'stamp' key if present.
    #
    if 'stamp' in network:
        del network['stamp']

    # Store the HF5 file content inside the model file as byte stream.
    #
    if type(network['parameters']) is dict and 'keras_model_file_name' in network['parameters']:
        # Construct the HF4 file path. Some old networks stored it as an absolute path. It is expected to be next to the network file.
        #
        hf5_path = '{base}{ext}'.format(base=os.path.splitext(network['path'])[0], ext='.hf5')

        # Load the file into memory and extract it as a byte stream.
        #
        model_hf5 = h5py.File(name=hf5_path, mode='r', driver='core', backing_store=False)
        model_hf5_bytes = h5py.h5f.FileID.get_file_image(model_hf5.id)

        # Remove the path to the HF5 file from the network parameters and insert it as byte stream.
        #
        del network['parameters']['keras_model_file_name']
        network['parameters']['hf5_image'] = model_hf5_bytes

    # Return the updated network.
    #
    return network

#----------------------------------------------------------------------------------------------------

def update_network(source_path, target_path, updated_description):
    """
    Update a network that may have been saved with joblib to a format that is saved with the digitalpathology.utils.serialize.save_data function.
    The old format files typically have '.pkl' file extension while the new format files are suppose to have '.net' extension. Furthermore it updates
    the description of the network. If the format is already the new '.net' format the function does everything except trying to convert the binary format.

    Args:
        source_path (str): Source network file in old format.
        target_path (str): Target network file in new format.
        updated_description (str, None): Updated description string.
    """

    # Start time measurement.
    #
    start_time = time.time()

    # Get a logger.
    #
    logger = logging.getLogger(name=__name__)

    logger.info('Loading network data: {path}'.format(path=source_path))

    # Load old network. Test if the format is the old joblib format or the new custom '.net' binary format.
    #
    network_extension = os.path.splitext(source_path)[-1].lower()
    if network_extension == '.net':
        with open(file=source_path, mode='rb') as file:
            network = pickle.loads(zlib.decompress(file.read()))
    else:
        network = joblib.load(filename=source_path)

    # Update the description of the network.
    #
    if updated_description is not None:
        network['description'] = updated_description.format(description=network['description'])

    # Insert source path to the network data structure: it is necessary if the old network stored the layer topology and data in a separate HF5 file.
    #
    network['path'] = source_path

    # Update the network data.
    #
    logger.info('Updating network format...')
    updated_network = update_network_format(network=network)

    # Remove the temporary entry from the data structure, it should not be saved.
    #
    del network['path']

    # Update the old digitalpathology library to the new diagmodels for the network object.
    #
    if network['class'].startswith('digitalpathology.'):
        network['class'] = network['class'].replace('digitalpathology.', 'diagmodels.')

    # Save the network in the new format.
    #
    logger.info('Saving network data: {path}'.format(path=target_path))
    with open(file=target_path, mode='wb') as file:
        file.write(zlib.compress(pickle.dumps(updated_network), level=9))

    # Report execution time.
    #
    execution_time = time.time() - start_time
    logger.debug('Done in {delta}'.format(delta=datetime.timedelta(seconds=execution_time)))

#----------------------------------------------------------------------------------------------------

def collect_arguments():
    """
    Collect command line arguments.

    Returns:
        (str, str, str): The parsed command line arguments: Input and output network model files, the updated description string and.
    """

    # Configure argument parser.
    #
    argument_parser = argparse.ArgumentParser(description='Update the network file format.',
                                              epilog='Use {description} as replacement for the current description of the network in the new description string. '
                                                     'The format function of the description string will be called so use {{ and }} instead of { and } if necessary. ')

    argument_parser.add_argument('-s', '--source',      required=True,  type=str,               help='source network file (.pkl or .net)')
    argument_parser.add_argument('-t', '--target',      required=True,  type=str,               help='target network file (.net)')
    argument_parser.add_argument('-d', '--description', required=False, type=str, default=None, help='the new description of the network')

    # Parse arguments.
    #
    arguments = vars(argument_parser.parse_args())

    # Collect arguments.
    #
    parsed_source_path = arguments['source']
    parsed_target_path = arguments['target']
    parsed_description = arguments['description']

    # Print parameters.
    #
    print(argument_parser.description)
    print('Source network file: {path}'.format(path=parsed_source_path))
    print('Target network file: {path}'.format(path=parsed_target_path))
    print('Updated description: {desc}'.format(desc='\'{param}\''.format(param=parsed_description) if parsed_description is not None else parsed_description))

    return parsed_source_path, parsed_target_path, parsed_description

#----------------------------------------------------------------------------------------------------

def main():
    """Main function."""

    # Increase recursion limit to be able to load the networks.
    #
    sys.setrecursionlimit(100000)

    # Parse parameters.
    #
    source_network_path, target_network_path, updated_description = collect_arguments()

    # Init the logger to print to the console.
    #
    init_console_logger(debug=True)

    # Update the network.
    #
    update_network(source_path=source_network_path, target_path=target_network_path, updated_description=updated_description)

#----------------------------------------------------------------------------------------------------


if __name__ == '__main__':

    main()
