import os
import time
import threading
import importlib
import digitalpathology.image.io.imagereader as dptimagereader
from digitalpathology.errors.imageerrors import ImageOpenError
from digitalpathology.errors.imageerrors import PixelSpacingLevelError
from pathlib import Path
from multiprocessing import Queue, JoinableQueue, set_start_method
from . import async_tile_processor, async_wsi_reader, async_wsi_writer

import traceback
import time, os

class async_wsi_consumer:
    def __init__(self, model_path, axes_order, batch_size,
                 gpu_count, write_spacing, mask_spacing, mask_class, read_spacing, normalizer, augment, tile_size,
                 recon_info, output_channels, readers, writers, work_directory, unfix_network, overwrite, touch,
                 profiler, custom_processor=None, custom_writer=None, custom_reader=None, verbose=False,
                 soft=False, quantize=True,
                 lookup_table_path=None,
                 encoder_name_artifact='tu-convnext_base', encoder_weights_artifact='imagenet',
                 decoder_name_artifact='deeplabv3plus',
                 error_queue=None, postprocess=False, use_tta=False):
        set_start_method(method='spawn')

        self._model_path = model_path
        self._lookup_table_path = lookup_table_path
        self._axes_order = axes_order
        self._batch_size = batch_size
        self._gpu_count = gpu_count

        self._readers = readers
        self._writers = writers

        self._write_spacing = write_spacing if write_spacing is not None else read_spacing
        self._mask_spacing = mask_spacing
        self._mask_class = mask_class
        self._read_spacing = read_spacing
        self._tile_size = tile_size
        self._output_tile_size = None
        self._finish = False

        self._normalizer = normalizer
        self._augment = augment
        self._soft = soft
        self._quantize = quantize
        self._output_channels = [0] if output_channels is None else output_channels  

        self._recon_info = recon_info
        self._work_directory = work_directory
        self._unfix_network = unfix_network
        self._overwrite = overwrite
        self._touch_output = touch
        self._profiler = profiler

        self._custom_processor = custom_processor
        self._custom_writer = custom_writer
        self._custom_reader = custom_reader

        self._encoder_name = encoder_name_artifact
        self._encoder_weights = encoder_weights_artifact
        self._decoder_name = decoder_name_artifact

        self._verbose = verbose
        pads, _, interpolation = self._recon_info
        self._output_tile_size = self._tile_size - pads[2] - pads[3] - interpolation[2] - interpolation[3]

        self._error_queue = error_queue
        self.postprocess = postprocess
        self.use_tta = use_tta

    def _start_error_monitor(self, error_queue):
        """
        Watches the error_queue for error messages from worker processes.
        If any worker reports an error, print it and abort the whole program.
        """
        def monitor():
            while True:
                msg = error_queue.get()  # blocking
                if msg is None:
                    # Optional: stop signal
                    break
                kind, payload = msg
                if kind == "error":
                    print("\n[FASTINFERENCE FATAL ERROR]\n", payload, flush=True)
                    # Hard-exit: kill this process so job fails visibly
                    os._exit(1)

        t = threading.Thread(target=monitor, daemon=True)
        t.start()
        
    def _read_thread(self, consumer_queue, tile_queue, write_queue_list, job_list):
        writer_sequence_nr = 0

        for job in job_list:
            input_path, mask_path, output_path, cache_path = job

            if not self._overwrite and os.path.exists(output_path):
                print(f"{output_path} already exists, continuing..")
                continue

            output_path_obj = Path(output_path)
            output_path_obj.parent.mkdir(parents=True, exist_ok=True)
            if self._touch_output:
                output_path_obj.touch()

            try:
                filename, input_shape, output_resampling, output_shape, output_spacing = \
                    self._init_reader_vars(input_path, output_path)

            except ImageOpenError as e:
                print(f"[Reader] WARNING: Cannot open image '{input_path}': {e}")
                continue

            except PixelSpacingLevelError as e:
                print(f"[Reader] WARNING: Pixel spacing mismatch for '{input_path}': {e} — skipping image")
                continue

            except Exception as e:
                # NEW: fatal unexpected errors
                import traceback
                tb = traceback.format_exc()
                msg = f"[Reader] ERROR while initializing '{input_path}':\n{tb}"
                print(msg, flush=True)

                # report to error_queue if available
                if getattr(self, "_error_queue", None) is not None:
                    self._error_queue.put(("error", msg))

                return  # stop thread → parent error monitor kills job

            # If we get here, initialization succeeded
            consumer_queue.put(None)

            self._start_reader_process(
                input_path, input_shape, mask_path,
                tile_queue, writer_sequence_nr, filename,
                cache_path=cache_path
            )

            write_queue_list[writer_sequence_nr].put(
                ("new_image", output_path, output_shape, output_spacing, output_resampling)
            )

            writer_sequence_nr = (writer_sequence_nr + 1) % self._writers

        # finish consumption
        _ = consumer_queue.get()
        consumer_queue.task_done()

    def _start_reader_process(self, input_path, input_shape, mask_path, tile_queue, writer_sequence_nr, filename, cache_path=None):
        print("starting reader process for {}".format(filename))
        lookup_table_path = os.path.join(self._lookup_table_path, filename.split(".")[0] + ".tif") if self._lookup_table_path else None
        if self._custom_reader:
            reader_module_str = ".readers.{}".format(self._custom_reader)
            custom_reader= importlib.import_module(reader_module_str, package="fastinference")
            async_reader = getattr(custom_reader, self._custom_reader)
        else:
            async_reader = async_wsi_reader.async_wsi_reader
        reader = async_reader(input_wsi_path=input_path,
                              lookup_table_path=lookup_table_path,
                              output_filename=filename,
                              read_tile_queue=tile_queue,
                              network_info=self._recon_info,
                              tile_size=self._tile_size,
                              output_shape=input_shape,
                              mask_wsi_path=mask_path,
                              cache_path=cache_path,
                              spacing=self._read_spacing,
                              mask_spacing=self._mask_spacing,
                              mask_class=self._mask_class,
                              batch_size=self._batch_size,
                              preprocess_function=self._normalizer,
                              writer_sequence_nr=writer_sequence_nr,
                              verbose=self._verbose,
                              error_queue=self._error_queue)
        reader.start()

    def _init_reader_vars(self, input_path, output_path):
        input_wsi = dptimagereader.ImageReader(image_path=input_path)
        read_spacing = input_wsi.refine(spacing=self._read_spacing)
        read_level = input_wsi.level(spacing=self._read_spacing)
        write_level = input_wsi.level(spacing=self._write_spacing)
        input_shape = input_wsi.shapes[read_level]
        output_shape = input_wsi.shapes[write_level]
        output_spacing = input_wsi.refine(spacing=self._write_spacing)
        output_resampling = read_spacing / output_spacing
        input_wsi.close()
        filename = os.path.basename(output_path)
        return filename, input_shape, output_resampling, output_shape, output_spacing

    def _join_consumer_queue(self, consumer_queue):
        t1 = time.time()
        consumer_queue.join()
        print("total runtime: {}".format(time.time() - t1))

    def _start_writer_processes(self, consumer_queue, write_queue_list):
        for i in range(self._writers):
            if self._custom_writer:
                writer_module_str = ".writers.{}".format(self._custom_writer)
                custom_writer = importlib.import_module(writer_module_str, package="fastinference")
                writer_cls = getattr(custom_writer, self._custom_writer)
                writer = writer_cls(read_queue=write_queue_list[i],
                                    consumer_queue=consumer_queue,
                                    output_tile_size=self._output_tile_size,
                                    rec_info=self._recon_info,
                                    soft=self._soft,
                                    quantize=self._quantize,
                                    output_channels=self._output_channels,
                                    work_directory=self._work_directory,
                                    profiler=self._profiler,
                                    error_queue=self._error_queue)
            else:
                writer = async_wsi_writer.async_wsi_writer(read_queue=write_queue_list[i],
                                                           consumer_queue=consumer_queue,
                                                           output_tile_size=self._output_tile_size,
                                                           rec_info=self._recon_info,
                                                           soft=self._soft,
                                                           quantize=self._quantize,
                                                           output_channels=self._output_channels,
                                                           work_directory=self._work_directory,
                                                           profiler=self._profiler,
                                                           error_queue=self._error_queue,
                                                           postprocess=self.postprocess)
            writer.start()

    def _start_reader_thread(self, consumer_queue, job_list, tile_queue, write_queue_list):
        read_thread = threading.Thread(target=self._read_thread,
                                       args=(consumer_queue, tile_queue, write_queue_list, job_list))
        read_thread.start()

    def _start_gpu_processes(self, tile_queue, write_queue_list):
        for gpu_nr in range(self._gpu_count):
            if self._custom_processor:
                proc_module_str = ".processors.{}".format(self._custom_processor)
                custom_processor = importlib.import_module(proc_module_str, package="fastinference")
                processor_cls = getattr(custom_processor, self._custom_processor)
                processor = processor_cls(read_queue=tile_queue, write_queues=write_queue_list,
                                          model_path=self._model_path, augment=self._augment, soft=self._soft,
                                          batch_size=self._batch_size, ax_order=self._axes_order, gpu_device=gpu_nr,
                                          unfix_network=self._unfix_network, tile_size=self._tile_size,
                                          encoder_name=self._encoder_name, encoder_weights=self._encoder_weights,
                                          decoder_name=self._decoder_name,
                                          error_queue=self._error_queue, use_tta=self.use_tta)
            else:
                processor = async_tile_processor.async_tile_processor(read_queue=tile_queue, write_queues=write_queue_list,
                                                                      model_path=self._model_path,
                                                                      augment=self._augment, soft=self._soft,
                                                                      batch_size=self._batch_size,
                                                                      ax_order=self._axes_order, gpu_device=gpu_nr,
                                                                      unfix_network=self._unfix_network,
                                                                      tile_size=self._tile_size,
                                                                      error_queue=self._error_queue)


            processor.start()

    def _set_custom_imports(self):
        return

    def apply_network_on_joblist(self, job_list):
        self._set_custom_imports()

        print("starting queues..", flush=True)
        tile_queue = Queue(self._readers * 2)
        write_queue_list = [Queue(self._gpu_count) for _ in range(self._writers)]
        consumer_queue = JoinableQueue(self._readers)
        consumer_queue.put(None) # Poison Pill

        error_queue = Queue()
        self._error_queue = error_queue
        self._start_error_monitor(error_queue)

        print("starting writer processes..", flush=True)
        self._start_reader_thread(consumer_queue, job_list, tile_queue, write_queue_list)
        print("starting GPU processes..", flush=True)
        self._start_gpu_processes(tile_queue, write_queue_list)
        print("starting reader thread..", flush=True)
        self._start_writer_processes(consumer_queue, write_queue_list)
        print("waiting for consumer join..", flush=True)
        self._join_consumer_queue(consumer_queue)


