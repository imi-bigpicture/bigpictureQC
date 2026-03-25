import skimage

from ..async_wsi_reader import async_wsi_reader
import os
import numpy as np


class SSIM(async_wsi_reader):
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
        async_wsi_reader.__init__(self, **kwargs)
        print("using custom reader SSIM")

    def _run_loop(self, batch_ind, img_batch, info, msk_batch, normalizer, sequence_nr, tile_list):

        # pr = cProfile.Profile()
        # pr.enable()
        for tile_info in tile_list:
            (tile_x,
             tile_y,
             tile_height,
             tile_width,
             tile_result_x,
             tile_result_y,
             pad_top,
             pad_bottom,
             pad_left,
             pad_right,
             crop_top,
             crop_bottom,
             crop_left,
             crop_right,
             padding_kwargs,
             network_lost,
             network_downsamples,
             interpolation_lost) = tile_info
            if self._mask_wsi is not None:
                msk_tile = self._get_image_data_from_mask([tile_y, tile_y + tile_height, tile_x, tile_x + tile_width])
                if not np.sum(msk_tile, axis=2).all():
                    continue
            else:
                msk_tile = None

            img_tile = self._get_image_data_from_input([tile_y, tile_y + tile_height, tile_x, tile_x + tile_width])

            img_batch[batch_ind] = img_tile
            if msk_tile is not None:
                msk_batch[batch_ind] = msk_tile
            info[batch_ind] = [tile_result_x, tile_result_y]

            batch_ind += 1
            if batch_ind == self._batch_size:
                self._preprocess_and_send_batch(img_batch, info, msk_batch, normalizer, sequence_nr)

                img_batch = np.zeros(
                    (self._batch_size, self._tile_size, self._tile_size, self._input_wsi.channels),
                    dtype=np.float32)
                msk_batch = np.zeros((self._batch_size, self._tile_size, self._tile_size, 1), dtype=np.int8)
                info = np.zeros((self._batch_size, 2), dtype=np.int32) - 1
                batch_ind = 0
                sequence_nr += 1
        if batch_ind > 0:
            self._preprocess_and_send_batch(img_batch, info, msk_batch, normalizer, sequence_nr)
            sequence_nr += 1
        print("sending final message from reader for file: {}".format(self._output_filename))
        self._queue.put(('finish_image', self._output_filename, sequence_nr, self._writer_sequence_nr))
