from ..async_tile_processor import async_tile_processor
import numpy as np



class tumorstroma(async_tile_processor):
    """
    Gan tile processor, subclassed fro the async_tile_processor.
    """
    def __init__(self, **kwargs):
        """
        """
        async_tile_processor.__init__(self, **kwargs)
        print("using custom processor tumorstroma")

    def _run_loop(self):
        while True:
            tile_info = self._fast_read_queue.get()
            writer_nr = tile_info[-1]
            if tile_info[0] == 'finish_image':
                self._write_queues[writer_nr].put(tile_info[:-1])
                continue
            output_filename, sequence_nr, tile_batch, mask_batch, info, _ = tile_info
            tile_batch, mask_batch = self._preprocess_tiles(tile_batch, mask_batch)
            result_batch = self._process_tile_batch(tile_batch, info)
            self._write_queues[writer_nr].put(
                ('write_tile', output_filename, sequence_nr, result_batch, mask_batch, info))

    def _preprocess_tiles(self, tile_batch, mask_batch):
        for c in range(tile_batch.shape[3]):
            tile_batch[:,:,:,c][mask_batch[:,:,:,0] == 2] = 0
        return (tile_batch, (mask_batch == 1).astype(np.uint8))

