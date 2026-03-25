from ..async_wsi_reader import async_wsi_reader

class gan_wsi_reader(async_wsi_reader):
    def __init__(self, **kwargs):
        async_wsi_reader.__init__(self, **kwargs)

    def _set_normalization_fun(self):
        """
        creates the normalization functions. Will expanded in the future.
        Returns:
            normalizer: function that normalizes an image batch. Returns none if there is no normalization necessary.
        """
        def normalizer(batch):
            batch /= 127.5
            batch -= 1
        return normalizer

    def _preprocess_and_send_batch(self, img_batch, info, msk_batch, normalizer, sequence_nr):
        normalizer(img_batch)
        if self._mask_wsi:
            self._queue.put([self._output_filename,
                             sequence_nr,
                             img_batch,
                             msk_batch,
                             info,
                             self._writer_sequence_nr])
        else:
            self._queue.put([self._output_filename,
                             sequence_nr,
                             img_batch,
                             None,
                             info,
                             self._writer_sequence_nr])

    def get_ranges(self, cropped_height, cropped_width, lost_l_i, lost_t_i, result_height, result_width):
        y_range = range(-lost_l_i, result_height, self._tile_size // 2)  # add pad left
        x_range = range(-lost_t_i, result_width, self._tile_size // 2)  # add pad top
        return x_range, y_range