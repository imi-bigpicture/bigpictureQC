import skimage.transform

from ..async_wsi_writer import async_wsi_writer
import numpy as np
from digitalpathology.image.io import imagewriter as dptimagewriter
import os

class cutout_epithelium(async_wsi_writer):
    def __init__(self, **kwargs):
        async_wsi_writer.__init__(self, **kwargs)
        print("using custom writer cutout_epithelium")


    def _postprocess_batch(self, image_batch, mask_batch, batch_info):
        return image_batch

    def _create_file_handle(self, filepath, output_shape, spacing, resample_size):
        print("creating image: {}".format(filepath))
        print("intermediate path: {}".format(self._work_directory))

        sliding_window = np.zeros((self._write_tile_size * 2, output_shape[1], 3),
                                  dtype=np.float32)

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
        sliding_window[mod_row:mod_row + tile.shape[0], col:col + tile.shape[1], :] = tile[:, :sliding_window.shape[1] - col]