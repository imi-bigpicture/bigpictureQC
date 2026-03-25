import skimage.transform

from ..async_wsi_writer import async_wsi_writer
import numpy as np
from digitalpathology.image.io import imagewriter as dptimagewriter
import os
import time

class gan_wsi_writer(async_wsi_writer):
    def __init__(self, **kwargs):
        async_wsi_writer.__init__(self, **kwargs)
        self._gradient_shape = None

    def _create_gradient_shape(self, tile_size):
        shift_size = tile_size // 2
        x = y = np.linspace(0, 4, shift_size)
        Z = np.array([min(i, j) for j in y for i in x])
        Z = Z.reshape(shift_size, shift_size)
        gradient_shape = np.empty((tile_size, tile_size))
        Z_sum = Z + Z[::-1, :] + Z[:, ::-1] + Z[::-1, ::-1]
        Z /= Z_sum
        gradient_shape[:shift_size, :shift_size] = Z
        gradient_shape[shift_size:, :shift_size] = Z[::-1, :]
        gradient_shape[:shift_size, shift_size:] = Z[:, ::-1]
        gradient_shape[shift_size:, shift_size:] = Z[::-1, ::-1]
        return np.array([gradient_shape for _ in range(3)]).transpose(1, 2, 0)

    def _postprocess_batch(self, image_batch, mask_batch, batch_info):
        image_batch = np.clip((image_batch + 1) / 2, 0, 1)
        return image_batch

    def _create_file_handle(self, filepath, output_shape, spacing, resample_size):
        """
        Creates an imagewriter for the WSI and a buffer to write tiles to. An entry is added to the _file_handle_dict
        to track the progress. Subclassed from the async_wsi_writer to be suitable for gan inference.
        Args:
            filepath (str): path to the output WSI file.
            output_shape (int,int): shape (y,x) of the output file, must be specified at initialization of the imagewriter.
            spacing (float): target spacing of the file
            resample_size (float): resampling size for when the input and output spacing differ.
        """
        print("creating image: {}".format(filepath))
        print("intermediate path: {}".format(self._work_directory))
	
        sliding_window = np.ones((self._write_tile_size * 2, output_shape[1], len(self._output_channels)),
                                  dtype=np.float32)
        print("Sliding window size", sliding_window.shape)
        writer = dptimagewriter.ImageWriter(image_path=filepath,
                                            shape=output_shape,
                                            spacing=spacing,
                                            dtype=np.uint8,
                                            coding='rgb',
                                            indexed_channels=0,
                                            compression='jpeg',
                                            interpolation='linear',
                                            tile_size=self._write_tile_size,
                                            jpeg_quality=None,
                                            empty_value=0,
                                            skip_empty=True,
                                            cache_path=self._work_directory)

        write_row = 0
        local_sequence_nr = 0
        sequence_list = []
        final_sequence_number = -1
        filename = os.path.basename(filepath)
        self._file_handle_dict[filename] = (writer, sliding_window, write_row, local_sequence_nr, sequence_list, final_sequence_number, resample_size)
        
        
    def _tile_to_buffer_op(self, col, mod_row, sliding_window, tile):
        if self._gradient_shape is None:
            self._gradient_shape = self._create_gradient_shape(tile.shape[1])
        sliding_window[mod_row:mod_row + tile.shape[0], col:col + tile.shape[1], :] -= (1 - tile[:, :sliding_window.shape[1] - col])  * self._gradient_shape[:, :sliding_window.shape[1] - col]

    def _write_buffer_to_file(self, mod_row, row, sliding_window, write_row, writer):
        while (row >= write_row + self._write_tile_size):
            self._write_buffer_row_to_file(write_row, sliding_window, writer)
            sliding_window[:sliding_window.shape[0]-self._write_tile_size, :, :] = sliding_window[self._write_tile_size:, :, :]
            sliding_window[-self._write_tile_size:, :, :] = 1.0
            write_row += self._write_tile_size

        return write_row
            

