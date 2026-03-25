import time

from ..async_tile_processor import async_tile_processor
import tensorflow
import segmentation_models

gpus = tensorflow.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tensorflow.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tensorflow.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)

class keras_processor(async_tile_processor):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


    def _load_network_model(self):
        network = tensorflow.keras.models.load_model(self._model_path, compile = False)
        return network
    
    def _predict_tile_batch(self, tile_batch=None, info=None):
        if self._ax_order == 'cwh':
            tile_batch = tile_batch.transpose(0, 3, 1, 2)
        result = self._model.predict(tile_batch)
        if self._ax_order == 'cwh':
            result = result.transpose(0, 2, 3, 1)
        return result

    def _send_reconstruction_info(self):
        self._write_queues[0].put(('recon_info',
                                   '',1))
