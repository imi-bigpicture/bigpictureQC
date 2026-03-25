from ..async_tile_processor import async_tile_processor
import os
import numpy as np


class cutout_epithelium(async_tile_processor):
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
        async_tile_processor.__init__(self, **kwargs)
        print("using custom processor cutout_epithelium")

    def _load_network_model(self):
        return None

    def _cutout_epithelium(self, tile_batch=None, mask_batch=None, info=None):
        """
        Performs classification of an image in one go.
        This function using the prediction functions to classify the image.
        If image is not given to this function, it will fill this variable
        from self._inputs. Optionally, msk_data can be provided to mask
        out certain parts of the image.
        """
        for c in range(tile_batch.shape[3]):
            tile_batch[:,:,:,c][mask_batch.squeeze() == 2] = 0
        return tile_batch


    def _run_loop(self):
        while True:
            tile_info = self._fast_read_queue.get()
            writer_nr = tile_info[-1]
            if tile_info[0] == 'finish_image':
                self._write_queues[writer_nr].put(tile_info[:-1])
                continue
            output_filename, sequence_nr, tile_batch, mask_batch, info, _ = tile_info
            result_batch = self._cutout_epithelium(tile_batch, mask_batch, info)
            self._write_queues[writer_nr].put(('write_tile', output_filename, sequence_nr, result_batch, mask_batch, info))

    def _send_reconstruction_info(self):
        """

        """
        self._write_queues[0].put(('recon_info',
                                   [[0, 0, 0, 0], [1, 1], [0, 0, 0, 0]],
                                   3))
