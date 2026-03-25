import io
import time
import pstats
from threading import Thread
import cProfile
import queue
import numpy as np
from multiprocessing import Process
from digitalpathology.image.io import imagewriter as dptimagewriter
from digitalpathology.image.processing.regions import filter_regions_array, fill_holes_array
from .utils.wsd_image import ArrayImageWriter
import skimage.transform
import os
import multiresolutionimageinterface as mir
import time, os

class async_wsi_writer(Process):
    def __init__(self, read_queue, consumer_queue, output_tile_size, rec_info, soft,
                 quantize, output_channels, work_directory, profiler, error_queue=None, postprocess=False):
        Process.__init__(self, name='TileWriter')
        self._profiler = profiler
        self._output_channels = output_channels
        self.daemon = True
        self._writer = None
        self._consumer_queue = consumer_queue
        self._queue = read_queue
        self._write_tile_size = 512
        self._output_tile_size = output_tile_size
        self._soft = soft
        self._quantize = quantize
        self._sliding_window = None
        self._rec_info = rec_info
        self._postprocess_function = None
        self._file_handle_dict = {}
        self._test_time = time.time()
        self._fast_read_queue = None
        self._work_directory = work_directory
        self._error_queue = error_queue
        self._postprocess = postprocess

    def _dump_profiler(self, profiler, sort_mode='cumulative'):
        ps = pstats.Stats(profiler).sort_stats(sort_mode)
        path = '/mnt/netcache/projects/TumorAssociatedStroma/results_profiler/writer_profile-{}'
        ps.dump_stats(path.format(self._profiler))

    def _postprocess_batch(self, image_batch, mask_batch, batch_info):
        pads, downsamples, interpolation_lost = self._rec_info

        if not self._soft:
            # if mask_batch is not None:
            #     image_batch = (np.argmax(image_batch, axis=-1) + 1)[:, :, :, None]
            # else:
            #     image_batch = (np.argmax(image_batch, axis=-1))[:, :, :, None]
            
            image_batch = (np.argmax(image_batch, axis=-1) + 1)[:, :, :, None]
            
        zoomed_result = np.zeros((image_batch.shape[0],
                                  int(image_batch.shape[1] * downsamples[0]),
                                  int(image_batch.shape[2] * downsamples[1]),
                                  image_batch.shape[3]))
        for ind, img in enumerate(image_batch):
            # The newer version of skimage requires an additional dimension.
            try:
                zoomed_result[ind] = skimage.transform.rescale(img, (downsamples[0], downsamples[1]),
                                                               preserve_range=True, order=1, mode="edge")
            except: 
                zoomed_result[ind] = skimage.transform.rescale(img, (downsamples[0], downsamples[1], 1),
                                                               preserve_range=True, order=1, mode="edge")


        zoomed_result = zoomed_result[:, interpolation_lost[0]:zoomed_result.shape[1]-interpolation_lost[1],
                        interpolation_lost[2]:zoomed_result.shape[2]-interpolation_lost[3]]

        total_pads = [pads[0]+ interpolation_lost[0], pads[1]+ interpolation_lost[1], pads[2]+ interpolation_lost[2],
                      pads[3] + interpolation_lost[3]]

        if mask_batch is not None:
            mask_batch = mask_batch[:, total_pads[2]:-total_pads[3], total_pads[0]:-total_pads[1], :]
            zoomed_result *= (mask_batch > 0)

        return zoomed_result

    def _write_buffer_row_to_file(self, current_row, sliding_window, writer):
        print("writing at {} in file: {}, in {}".format(current_row, os.path.basename(writer.path), time.time() - self._test_time),
              flush=True)
        self._test_time = time.time()
        for x in range(0, sliding_window.shape[1], self._write_tile_size):
            tile = sliding_window[:self._write_tile_size, x:x + self._write_tile_size, :]
            self._write_tile_to_file(writer=writer, tile=tile, x=x, y=current_row)
        # if full:
        #     for x in range(0, sliding_window.shape[1], self._write_tile_size):
        #         tile = sliding_window[self._write_tile_size:, x:x + self._write_tile_size, :]
        #         self._write_tile_to_file(writer=writer, tile=tile, x=x, y=current_row + self._write_tile_size)

    def _write_tile_to_file(self, writer, tile, x, y):
        """
        Write a tile to tiff in the correct data format and on the designated location
        """
        if tile.shape[1] < self._write_tile_size or tile.shape[0] < self._write_tile_size:
            tile = np.pad(tile,
                          ((0, max(0, self._write_tile_size - tile.shape[0])),
                           (0, max(0, self._write_tile_size - tile.shape[1])), (0, 0)),
                          'constant')
        if self._soft:
            if self._quantize:
                tile = np.clip(tile * 255, 0, 255).astype(np.uint8)
            
            
            writer.write(tile=tile, row=int(y), col=int(x))
        else:
            writer.write(tile=tile.astype(np.uint8), row=int(y), col=int(x))

    def _create_file_handle(self, filepath, output_shape, spacing, resample_size):
        print("creating image: {}".format(filepath), flush=True)
        y_size = np.max([self._output_tile_size + self._write_tile_size, self._write_tile_size * 2])
        if self._soft:
            sliding_window = np.zeros((y_size, output_shape[1], len(self._output_channels)),
                                      dtype=np.float32)
            data_type = np.uint8 if self._quantize else np.float32
            color_coding = 'indexed'
            interpolation_mode = 'linear'
        else:
            sliding_window = np.zeros((y_size, output_shape[1], 1), dtype=np.uint8)
            data_type = np.uint8
            color_coding = 'monochrome'
            interpolation_mode = 'nearest'

        writer = dptimagewriter.ImageWriter(image_path=filepath,
                                            shape=output_shape,
                                            spacing=spacing,
                                            dtype=data_type,
                                            coding=color_coding,
                                            indexed_channels=len(self._output_channels),
                                            compression='lzw',
                                            interpolation=interpolation_mode,
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
        self._file_handle_dict[filename] = (writer, sliding_window, write_row, local_sequence_nr, sequence_list, final_sequence_number, resample_size, spacing)

    def _write_tiles_to_buffer(self, writer, batch_info, write_row, write_tiles, sliding_window, resample_size):
        for tile, info in zip(write_tiles, batch_info):
            if info[0] < 0:
                continue
            col = int(info[0] * resample_size)  # result X
            row = int(info[1] * resample_size)  # result Y
            mod_row = row % self._write_tile_size

            if resample_size != 1.0:
                resize_shape = (np.ceil(resample_size * tile.shape[0]), np.ceil(resample_size * tile.shape[1]))
                tile = skimage.transform.resize(tile, resize_shape, preserve_range=True, order=self._soft, mode="edge")
            write_row = self._write_buffer_to_file(mod_row, row, sliding_window, write_row, writer)
            self._tile_to_buffer_op(col, mod_row, sliding_window, tile)

        return write_row

    def _write_buffer_to_file(self, mod_row, row, sliding_window, write_row, writer):
        while (row < 0 or row >= write_row + self._write_tile_size):
            if not sliding_window.any():
                write_row = row - mod_row
                break
            self._write_buffer_row_to_file(write_row, sliding_window, writer)
            sliding_window[:sliding_window.shape[0]-self._write_tile_size, :, :] = sliding_window[self._write_tile_size:, :, :]
            sliding_window[-self._write_tile_size:, :, :] = 0
            write_row += self._write_tile_size

        return write_row

    def _tile_to_buffer_op(self, col, mod_row, sliding_window, tile):
        sliding_window[mod_row:mod_row + tile.shape[0], col:col + tile.shape[1], :] = tile[:, :sliding_window.shape[1] - col, self._output_channels]

    def _write_tile_batch(self, filename, sequence_nr, image_batch, mask_batch, batch_info):
        write_tiles = self._postprocess_batch(image_batch, mask_batch, batch_info)
        (writer, sliding_window, write_row, local_sequence_nr, sequence_list, final_sequence_number, resample_size, spacing) = self._file_handle_dict[filename]

        sequence_list.append([sequence_nr, write_tiles, batch_info])
        if sequence_nr != local_sequence_nr:
            return
        else:
            sequence_list.sort(key=lambda x: x[0])
            while len(sequence_list) != 0 and sequence_list[0][0] == local_sequence_nr:
                _, write_tiles, batch_info = sequence_list.pop(0)
                write_row = self._write_tiles_to_buffer(writer, batch_info, write_row, write_tiles, sliding_window, resample_size)
                local_sequence_nr += 1
                if local_sequence_nr == final_sequence_number:
                    self._finish_and_close_writer(write_row, sliding_window, writer, filename)
                    break

        self._file_handle_dict[filename] = (writer, sliding_window, write_row, local_sequence_nr, sequence_list, final_sequence_number, resample_size, spacing)

    def _handle_final_batch(self, filename, final_sequence_number):
        (writer, sliding_window, write_row, local_sequence_nr, sequence_list, _, resample_size, spacing) = self._file_handle_dict[filename]
        if final_sequence_number  == local_sequence_nr:
            self._finish_and_close_writer(write_row, sliding_window, writer, filename)
        else:
            self._file_handle_dict[filename] = (writer, sliding_window, write_row, local_sequence_nr, sequence_list, final_sequence_number, resample_size, spacing)

    def _finish_and_close_writer(self, write_row, sliding_window, writer, filename):
        print("finishing writing of {}".format(filename), flush=True)
        self._write_buffer_to_file(write_row, -1, sliding_window, write_row, writer)
        writer.close(clear=True)

        (writer_, sliding_window_, write_row_, local_sequence_nr_, sequence_list_,
     final_sequence_number_, resample_size, spacing) = self._file_handle_dict[filename]

        post_path = writer.path
        if self._postprocess:
            image_reader = mir.MultiResolutionImageReader()
            image = image_reader.open(post_path)
            
            full_mask = image.getUCharPatch(0, 0, *image.getLevelDimensions(0), 0)
            if full_mask.ndim == 3:
                full_mask = full_mask[:, :, 0]
        
            region_threshold = 10.0
            hole_threshold = 200.0
            num_classes = 7

            binary_art = (full_mask > 1).astype(np.uint8)

            binary_filtered, _, _ = filter_regions_array(binary_art, diagonal_threshold=region_threshold,
                                                full_connectivity=True, foreground_labels=None, background_label=0)

            filtered = np.where(binary_filtered == 1, full_mask, np.where(full_mask > 1, 0, full_mask)).astype(np.uint8)

            for idx in range(2, num_classes + 1):
                cls_bin = (filtered == idx).astype(np.uint8)
                filled, _, _ = fill_holes_array(
                    cls_bin,
                    diagonal_threshold=hole_threshold,
                    full_connectivity=True,
                    foreground_labels=None,
                    fill_value=1
                )
                filtered[filled == 1] = idx
            
            writer = ArrayImageWriter()
            writer.write_array(filtered, post_path, spacing, verbose=True)
        
        del self._file_handle_dict[filename]
        _ = self._consumer_queue.get()
        self._consumer_queue.task_done()

    def _stealer_daemon(self, source_queue, dest_queue):
        def steal(source_queue, dest_queue):
            while True:
                obj = source_queue.get()
                dest_queue.put(obj)
        stealer = Thread(target=steal, args=(source_queue, dest_queue))
        stealer.daemon = True
        stealer.start()

    def _run_loop(self):
        while True:
            tile_info = self._fast_read_queue.get()
            if tile_info[0] == 'new_image':
                self._create_file_handle(*tile_info[1:])
            elif tile_info[0] == 'write_tile':
                self._write_tile_batch(*tile_info[1:])
            elif tile_info[0] == 'finish_image':
                self._handle_final_batch(*tile_info[1:])
            elif tile_info[0] == 'recon_info':
                continue

    def _initiate_fast_queue(self):
        self._fast_read_queue = queue.Queue(maxsize=5)
        self._stealer_daemon(self._queue, self._fast_read_queue)

    def _create_profiler(self):
        if self._profiler:
            pr = cProfile.Profile()
            pr.enable()

    def run(self):
        try:
            self._create_profiler()
            self._initiate_fast_queue()
            self._run_loop()
        except Exception as e:
            import traceback, os
            tb = traceback.format_exc()
            msg = f"[Writer PID {os.getpid()}] uncaught exception:\n{tb}"
            print(msg, flush=True)
            if self._error_queue is not None:
                self._error_queue.put(("error", msg))
            return
