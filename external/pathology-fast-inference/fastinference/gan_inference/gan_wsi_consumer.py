from ..async_wsi_consumer import async_wsi_consumer
from . import gan_tile_processor, gan_wsi_reader, gan_wsi_writer


class gan_wsi_consumer(async_wsi_consumer):
    def __init__(self, **kwargs):
        async_wsi_consumer.__init__(self, **kwargs)
        self._soft = True
        self._quantize = True
        self._output_channels = [0,1,2]

    def _start_reader_process(self, input_path, input_shape, mask_path, tile_queue, writer_sequence_nr, filename, cache_path=None):
        """
        Starts a single reader process. Subclassed to make suitable for GAN procesing.
        Args:
            input_path (str): input filepath of the WSI passed to the reader.
            mask_path (str): filepath of the mask WSI passed to the reader.
            output_path (str): output path of the WSI to infer the filename from.
            tile_queue (multiprocessing.Queue): tile queue passed to the reader to pass the tile batches to.
            writer_sequence_nr (int): number for the writer to pass the processed tile batches to.

        Returns:

        """
        print("cache path: {}".format(cache_path))
        reader = gan_wsi_reader.gan_wsi_reader(input_wsi_path=input_path,
                                               output_filename=filename,
                                               read_tile_queue=tile_queue,
                                               network_info=self._recon_info,
                                               tile_size=self._tile_size,
                                               output_shape=input_shape,
                                               mask_wsi_path=mask_path,
                                               spacing=self._read_spacing,
                                               mask_spacing=self._mask_spacing,
                                               batch_size=self._batch_size,
                                               preprocess_function=self._normalizer,
                                               writer_sequence_nr=writer_sequence_nr,
                                               cache_path=cache_path)
        reader.start()

    def _start_gpu_processes(self, tile_queue, write_queue_list):
        """
        This function starts all gpu processes. One process for each gpu. Subclassed to make suitable for GAN procesing.
        Args:
            tile_queue (multiprocessinq.Queue): the tile queue given to the gpu processes to take and process tile
            batches from.
            write_queue_list ([multiprocessing.Queue]): each process will get a all writer queues to distribute the
            images to the correct writer.
        """
        for gpu_nr in range(self._gpu_count):
            processor = gan_tile_processor.gan_tile_processor(read_queue=tile_queue, write_queues=write_queue_list,
                                                              model_path=self._model_path,
                                                              augment=self._augment,
                                                              soft=self._soft,
                                                              batch_size=self._batch_size,
                                                              ax_order=self._axes_order,
                                                              gpu_device=gpu_nr,
                                                              tile_size=self._tile_size)
            processor.start()


    def _start_writer_processes(self, consumer_queue, write_queue_list):
        """
        Function that starts the writer processes. Subclassed to make suitable for GAN procesing.
        Args:
            consumer_queue (multiprocessing.Queue): the consumer queue is used to communicate from the writer to the
            consumer that an image has finished processing.
            write_queue_list ([multiprocessing.Queue]): each writer will get a single writer queue to distribute the
            images equally.
        """
        for i in range(self._writers):
            writer = gan_wsi_writer.gan_wsi_writer(read_queue=write_queue_list[i], consumer_queue=consumer_queue,
                                                   output_tile_size=self._tile_size, rec_info=self._recon_info,
                                                   soft=self._soft, quantize=self._quantize,
                                                   output_channels=self._output_channels,
                                                   work_directory=self._work_directory, profiler=self._profiler)
            writer.start()
