from ..async_wsi_writer import async_wsi_writer
import os
import numpy as np


class SSIM(async_wsi_writer):
    """
    Gan tile processor, subclassed fro the async_tile_processor.
    """
    def __init__(self, **kwargs):
        """
        Args:
            read_queue (Process.queue):
            write_queues (Process.queue):
            model_path (str): Path to the model weights
            param_path (str): Path to the parameter file of the gan network.
            augment (bool): Test time augmentation. If enabled, performs inference at four 90-degree angles, flipped and
                unflipped. The outputs are geometrically averaged afterwards.
            soft:
            batch_size:
            ax_order:
            preprocess_batch:
            gpu_device:
            unfix_network:
            tile_size:
            a2b:
        """
        async_wsi_writer.__init__(self, **kwargs)
        print("using custom writer SSIM")

    def _create_file_handle(self, filepath, output_shape, spacing, resample_size):
        print("creating image: {}".format(filepath), flush=True)
        filename = os.path.basename(filepath)
        self._file_handle_dict[filename] = (filename, [], [])

    def _write_tile_batch(self, filename, sequence_nr, image_batch, mask_batch, batch_info):


        self._file_handle_dict[filename] = (writer, sliding_window, write_row, local_sequence_nr, sequence_list, final_sequence_number, resample_size)

    def _handle_final_batch(self, filename, final_sequence_number):
        (writer, sliding_window, write_row, local_sequence_nr, sequence_list, _, resample_size) = self._file_handle_dict[filename]
        if final_sequence_number  == local_sequence_nr:
            self._finish_and_close_writer(write_row, sliding_window, writer, filename)
        else:
            self._file_handle_dict[filename] = (writer, sliding_window, write_row, local_sequence_nr, sequence_list, final_sequence_number, resample_size)
