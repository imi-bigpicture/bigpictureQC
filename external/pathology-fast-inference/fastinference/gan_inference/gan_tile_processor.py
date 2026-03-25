from ..async_tile_processor import async_tile_processor
import tensorflow as tf
import numpy as np
import cyclegan.ops

class gan_tile_processor(async_tile_processor):
    """
    Gan tile processor, subclassed from the async_tile_processor.
    """
    def __init__(self, **kwargs):
        """
        See super class for initialization of other parameters.
        """
        async_tile_processor.__init__(self, **kwargs)

    def _load_network_model(self):
        """
        Load the network.
        """
        network = tf.keras.models.load_model(self._model_path,
                                             custom_objects={'ReflectionPadding2D': cyclegan.ops.ReflectionPadding2D},
                                             compile=False)
        return network

    def _predict_tile_batch(self, tile_batch=None, info=None):
        """
        Runs the tile batch through the network. Takes into account dim order. Overriden function from the super class.
        """
        if self._ax_order == 'cwh':
            tile_batch = tile_batch.transpose(0, 3, 1, 2)
        result = self._model.predict(tile_batch)
        if self._ax_order == 'cwh':
            result = result.transpose(0, 2, 3, 1)
        return result