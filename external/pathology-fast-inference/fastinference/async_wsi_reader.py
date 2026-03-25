import io
import pstats
import time
from multiprocessing import Process
from PIL import Image
import os
import digitalpathology.image.io.imagereader as dptimagereader

import skimage.transform
import numpy as np

import functools
print = functools.partial(print, flush=True)
import time, os

class async_wsi_reader(Process):
    def __init__(self, input_wsi_path, output_filename, read_tile_queue, network_info, tile_size, output_shape,
                 mask_wsi_path=None, spacing=0, mask_spacing=None, mask_class=1, batch_size=8, preprocess_function=None,
                 writer_sequence_nr=0, cache_path=None, verbose=False, lookup_table_path=None,
                 error_queue=None):
        Process.__init__(self, name='TileReader')
        self.daemon = True
        self._input_wsi = None
        self._mask_wsi = None
        self._output_filename = output_filename
        self._input_wsi_path = input_wsi_path
        self._mask_wsi_path = mask_wsi_path
        self._lookup_table = self.get_lookup_table(lookup_table_path) if lookup_table_path else None
        self._batch_size = batch_size
        self._tile_size = tile_size
        self._output_shape = output_shape
        self._spacing = spacing
        self._mask_spacing = mask_spacing if mask_spacing is not None else spacing
        self._mask_downscale = 1.0
        self._mask_class = mask_class
        self._network_info = network_info
        self._queue = read_tile_queue
        self._preprocess_function = preprocess_function
        self._tile_downsampling = None
        self._writer_sequence_nr = writer_sequence_nr
        self._cache_path = cache_path
        self._verbose = verbose
        self._error_queue = error_queue

    def get_lookup_table(self, lut_path):
        if os.path.exists(lut_path):
            lut = Image.open(lut_path)
            return np.asarray(lut, dtype=np.float32)[:, 0, :]
        else:
            return None

    def _get_image_data_from_input(self, box):
        """
        Get pixel data from the image region defined by box at specified scale.
        Corrects the box specified for the corresponding image level the data
        has to be acquired from. Then obtain the data from the image file and
        return it.
        """
        return self._input_wsi.read(spacing=self._spacing, row=box[0], col=box[2],
                                    height=int(box[1] - box[0]), width=int(box[3] - box[2]))

    def _get_image_data_from_mask(self, box):
        """
        Get mask data from the image file from the area defined by box.
        """
        return self._mask_wsi.read(spacing=self._mask_spacing, row=int(box[0] // self._mask_downscale),
                                   col=int(box[2] // self._mask_downscale),
                                   height=int((box[1] - box[0]) // self._mask_downscale),
                                   width=int((box[3] - box[2]) // self._mask_downscale))

    def _process_image_tiled(self):
        """
        Process the image tile-by-tile feeding tiles of self._tile_size to the network.
        """
        network_lost, network_downsamples, interpolation_lost = self._network_info
        print('reconstruction information:')
        print('network lost:', network_lost)
        print('network_downsamples:', network_downsamples)
        print('interpolation_lost:', interpolation_lost)

        result_height, result_width = self._output_shape
        print("sampling from file {} with size: {} by {}".format(self._input_wsi_path, result_height, result_width))

        # check if tile size minus everything that is lost is more then 0
        #
        if self._tile_size - network_lost[0] - network_lost[1] - interpolation_lost[0] - interpolation_lost[1] <= 0:
            raise ValueError("Tile_size " + str(self._tile_size) + " < than the amount of pixels lost on the edge (" +
                             str(network_lost[0] + network_lost[1] + interpolation_lost[0] + interpolation_lost[
                                 1]) + "px)."
                                       " Try a bigger tile-size.")

        # Determine tile-, cropped-  height and width + the range
        tile_height = self._tile_size
        tile_width = self._tile_size

        cropped_height = self._tile_size - network_lost[2] - network_lost[3] - interpolation_lost[2] - \
                         interpolation_lost[3]
        cropped_width = self._tile_size - network_lost[0] - network_lost[1] - interpolation_lost[0] - \
                        interpolation_lost[1]

        lost_l_i = network_lost[0] + interpolation_lost[0]  # amount of valid result lost at the left
        lost_r_i = network_lost[1] + interpolation_lost[1]  # amount of valid result lost at the right
        lost_t_i = network_lost[2] + interpolation_lost[2]  # etc.
        lost_b_i = network_lost[3] + interpolation_lost[3]

        x_range, y_range = self.get_ranges(cropped_height, cropped_width, lost_l_i, lost_t_i, result_height,
                                           result_width)
        n_tile_infos = len(x_range)*len(y_range)
        print('reading %d tiles' % n_tile_infos)
        # TODO: add dynamic padding constant back in
        # padding_kwargs = {'constant_values': self.padding_constant} if self._padding_mode == 'constant' else {}
        padding_kwargs = {}
        tile_list = []

        for y in y_range:
            for x in x_range:
                if self._verbose and len(tile_list) % 100000 == 0:
                    print('tiled %d/%d' % (len(tile_list), n_tile_infos))
                #
                # tile_{x|y} = coordinates of tile to fetch from input
                # tile_result_{x|y} = coordinates of the resulting network output tile
                # crop_{left|right|top|bottom} = what should be cropped from the network output
                # pad_{left|right|top|bottom} = what should be padded to network input tile
                #

                # Define cropbox
                #
                crop_left = lost_l_i
                crop_right = lost_r_i
                crop_top = lost_t_i
                crop_bottom = lost_b_i

                pad_left = 0
                pad_right = 0
                pad_top = 0
                pad_bottom = 0

                # 'Edge'-case when on the left border
                #
                if x < 0:
                    tile_x = 0
                    # Because in valid-padding networks we loose output on the left *and*
                    # we want the have the same output size as input size, 'x' starts at negative values
                    # corresponding to how much we loose on the left. Since the input image has no pixels
                    # at negative coordinates, we will pad zeros.
                    # It could be that after one tile we are still in the 'negative' coordinates because
                    # the output-width of our network is smaller than the output-lost at the border. So
                    # we need to pad multiple tiles.
                    tile_result_x = lost_l_i + x  # since x is negative this works
                    pad_left = -x  # pad zeros for the amount of input image we do not have
                    crop_left = lost_l_i  # We want to keep interpolation lost at the left border
                # 'Edge'-case when on the right border
                #
                elif x + self._tile_size >= result_width:
                    tile_x = x
                    tile_result_x = crop_left + tile_x
                    # The tile we want to give to the network is larger then the input
                    # This means we have to pad the tile to get to the right size
                    # (result_width here is the same as input_width)
                    #
                    pad_right = (tile_x + self._tile_size) - result_width
                    # If this is the last tile we probably need to crop more on the right
                    # to end up with exactly the width of the final result
                    #
                    # To check if this is the last, check if the valid network result > result_width
                    # (self._tile_size - crop_left - network_lost[1]) is basically:
                    #  = valid output + interpolation lost at the right (since we want to keep it there)
                    #
                    if tile_result_x + (self._tile_size - crop_left - network_lost[1]) >= result_width:
                        crop_right = tile_result_x + (self._tile_size - crop_left) - result_width
                # Normal case
                #
                else:
                    tile_x = x
                    # Coordinates in the result will be input tile coords + what has been cropped left
                    tile_result_x = crop_left + tile_x

                # For explanations see logic above
                #
                if y < 0:
                    tile_y = 0
                    tile_result_y = lost_t_i + y
                    pad_top = -y
                    crop_top = lost_t_i
                elif y + self._tile_size >= result_height:
                    tile_y = y
                    tile_result_y = crop_top + tile_y
                    pad_bottom = (tile_y + self._tile_size) - result_height
                    if tile_result_y + self._tile_size - crop_top - network_lost[3] >= result_height:
                        crop_bottom = tile_result_y + (self._tile_size - crop_top) - result_height
                else:
                    tile_y = y
                    tile_result_y = crop_top + tile_y
                if tile_result_x > result_width or tile_result_y > result_height:
                    continue
                tile_result_width = self._tile_size - pad_left - pad_right
                tile_result_height = self._tile_size - pad_top - pad_bottom

                # If there is a mask fetch the data
                #
                tile_list.append([tile_x,
                                  tile_y,
                                  tile_result_height,
                                  tile_result_width,
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
                                  interpolation_lost])

        return tile_list

    def get_ranges(self, cropped_height, cropped_width, lost_l_i, lost_t_i, result_height, result_width):
        y_range = range(-lost_l_i, result_height, cropped_height)  # add pad left
        x_range = range(-lost_t_i, result_width, cropped_width)  # add pad top
        return x_range, y_range

    def generate_coords(self):
        #TODO
        tile_batch_coordinates = []
        yield tile_batch_coordinates

    def _run_loop2(self, batch_ind, img_batch, info, msk_batch, normalizer, sequence_nr, tile_list):
        #TODO
        return

    def apply_lookup_table(self, tile=None):
        if self._verbose: print('apply_lookup_table')
        # Reshape tile for normalization.
        #
        tile_shape = tile.shape
        reshaped_tile = tile.reshape((tile_shape[0] * tile_shape[1], 3))
        normalized_tile = np.zeros(reshaped_tile.shape)

        # Calculate indices.
        #
        pixel_indices = range(reshaped_tile.shape[0])
        all_indices = reshaped_tile[pixel_indices, 0] * 256 * 256 + reshaped_tile[pixel_indices, 1] * 256 + \
                      reshaped_tile[pixel_indices, 2]

        # Normalize image.
        #
        normalized_tile[pixel_indices] = self._lookup_table[all_indices.astype(int)]
        normalized_tile = normalized_tile.reshape(tile_shape)

        # Return the normalized tile as float from the original [0.0, 1.0] range.
        #
        return normalized_tile.astype(np.uint8)


    def _run_loop(self, batch_ind, img_batch, info, msk_batch, normalizer, sequence_nr, tile_list):
        if self._verbose: print('_run_loop')
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
                msk_tile = (msk_tile == self._mask_class)
                if not msk_tile.any():
                    continue
            else:
                msk_tile = None

            t_read = time.time()
            img_tile = self._get_image_data_from_input([tile_y, tile_y + tile_height, tile_x, tile_x + tile_width])
            if self._mask_wsi:
                msk_tile = skimage.transform.resize(msk_tile, (img_tile.shape[0], img_tile.shape[1]),
                                                    order=0, mode='constant', preserve_range=True)
            img_tile, msk_tile = self._pad_data(img_tile, msk_tile, pad_bottom, pad_left, pad_right, pad_top,
                                                padding_kwargs)

            if self._mask_wsi and crop_bottom > 0 and not (msk_tile[crop_top:-crop_bottom, crop_left:-crop_right]).any():
                continue

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

    def _set_wsi_vars(self):
        self._input_wsi = dptimagereader.ImageReader(image_path=self._input_wsi_path, cache_path=self._cache_path)
        input_level = self._input_wsi.level(spacing=self._spacing)
        self._tile_downsampling = self._input_wsi.downsamplings[input_level]

        if self._mask_wsi_path is not None:
            print('reading from mask %s, spacing=%.1f' % (self._mask_wsi_path, self._mask_spacing))
            mask_cache = None
            if self._cache_path:
                mask_cache = os.path.join(self._cache_path, 'mask')

            self._mask_wsi = dptimagereader.ImageReader(image_path=self._mask_wsi_path,
                                                        spacing_tolerance=1.0,
                                                        cache_path=mask_cache)
            mask_level = self._mask_wsi.level(spacing=self._mask_spacing)
            self._mask_downscale = self._input_wsi.shapes[input_level][1] / self._mask_wsi.shapes[mask_level][1]

    def _clear_wsi_vars(self):
        if self._input_wsi is not None:
            self._input_wsi.close()

        if self._mask_wsi is not None:
            self._mask_wsi.close()

    def _set_batch_vars(self):
        img_batch = np.zeros((self._batch_size, self._tile_size, self._tile_size, self._input_wsi.channels),
                             dtype=np.float32)
        msk_batch = np.zeros((self._batch_size, self._tile_size, self._tile_size, 1), dtype=np.int8)
        info = np.zeros((self._batch_size, 2), dtype=np.int32)
        batch_ind = 0
        sequence_nr = 0
        return batch_ind, img_batch, info, msk_batch, sequence_nr

    def _set_normalization_fun(self):
        """
        Returns a function that normalizes img_batch in-place (N,H,W,C) float32,
        or None for no normalization.
        Accepted values for self._preprocess_function:
        - "none" or ""  -> no normalization
        - "default"     -> divide by 255.0  (backwards compatible)
        - "divide255"   -> divide by 255.0
        - "imagenet"    -> (x/255 - mean) / std  with ImageNet stats
        """
        mode = (self._preprocess_function or "").strip().lower()

        if mode in ("", "none"):
            return None

        if mode in ("default", "simple"):
            def _norm_div255(batch):
                # batch: float32 (N,H,W,C)
                batch /= 255.0
            return _norm_div255

        if mode == "imagenet":
            # ImageNet mean/std in RGB, range 0..1
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
          
            mean = mean.reshape(1,1,1,3)
            std  = std.reshape(1,1,1,3)

            def _norm_imagenet(batch):
                # Ensure float32
                if batch.dtype != np.float32:
                    batch[:] = batch.astype(np.float32)
                batch /= 255.0
                batch -= mean
                batch /= std
            return _norm_imagenet

        # Fallback: behave like old behavior (divide by 255)
        def _norm_div255(batch):
            batch /= 255.0
        return _norm_div255

    def _pad_data(self, img_tile, msk_tile, pad_bottom, pad_left, pad_right, pad_top, padding_kwargs):
        if (pad_top > 0) | (pad_bottom > 0) | (pad_left > 0) | (pad_right > 0):
            img_tile = np.pad(img_tile,
                              ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
                              mode='constant', **padding_kwargs)
            if self._mask_wsi:
                msk_tile = np.pad(msk_tile,
                                  ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
                                  mode='constant', **padding_kwargs)
        return img_tile, msk_tile

    def _preprocess_and_send_batch(self, img_batch, info, msk_batch, normalizer, sequence_nr):
        if self._lookup_table is not None:
            for i in range(img_batch.shape[0]):
                img_batch[i] = self.apply_lookup_table(tile=img_batch[i])

        if self._preprocess_function:
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

    def run(self):
        try:
            tile_list = self._process_image_tiled()
            self._set_wsi_vars()
            batch_ind, img_batch, info, msk_batch, sequence_nr = self._set_batch_vars()
            normalizer = self._set_normalization_fun()
            print(f"Reader normalization: {self._preprocess_function}", flush=True)
            self._run_loop(batch_ind, img_batch, info, msk_batch, normalizer, sequence_nr, tile_list)
            self._clear_wsi_vars()
        except Exception as e:
            import traceback, os
            tb = traceback.format_exc()
            msg = f"[Reader PID {os.getpid()}] uncaught exception:\n{tb}"
            print(msg, flush=True)
            if self._error_queue is not None:
                self._error_queue.put(("error", msg))
            return
