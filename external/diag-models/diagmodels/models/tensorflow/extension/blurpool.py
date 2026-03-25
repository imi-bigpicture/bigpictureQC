"""
Implement BlurPool as in https://arxiv.org/abs/1904.11486 https://github.com/adobe/antialiased-cnns
"""

import tensorflow as tf
import numpy as np

from tensorflow.python.keras.utils import conv_utils
from tensorflow.python.framework import tensor_shape

#----------------------------------------------------------------------------------------------------

class BlurPool(tf.keras.layers.Layer):
    """BlurPool layer class."""

    def __init__(self, filter_size=5, stride=2, **kwargs):
        """
        Initialize the layer parameters.

        Args:
            filter_size (int): Filter kernel size.
            stride (int): Stride.
        """

        # TODO: Different padding
        # TODO: Consider channel order

        self.filter_size = filter_size
        self.padding = 'valid'
        self.strides = (stride, stride)
        self.kernel_size = (filter_size, filter_size)
        self.kernel = None

        if self.filter_size == 1:
            self.kernel_array = np.array([1., ])
        elif self.filter_size == 2:
            self.kernel_array = np.array([1., 1.])
        elif self.filter_size == 3:
            self.kernel_array = np.array([1., 2., 1.])
        elif self.filter_size == 4:
            self.kernel_array = np.array([1., 3., 3., 1.])
        elif self.filter_size == 5:
            self.kernel_array = np.array([1., 4., 6., 4., 1.])
        elif self.filter_size == 6:
            self.kernel_array = np.array([1., 5., 10., 10., 5., 1.])
        elif self.filter_size == 7:
            self.kernel_array = np.array([1., 6., 15., 20., 15., 6., 1.])
        else:
            raise ValueError('Invalid filter size: {size}'.format(size=filter_size))

        super(BlurPool, self).__init__(**kwargs)

    def build(self, input_shape):
        """
        Build the layer. Calculate the filter kernel.

        Args:
            input_shape (tuple): Input shape.
        """

        self.kernel = self.kernel_array[:, None] * self.kernel_array[None, :]
        self.kernel = self.kernel / np.sum(self.kernel)
        self.kernel = np.tile(self.kernel[:, :, None, None], (1, 1, input_shape[-1], 1))
        self.kernel = tf.keras.backend.constant(self.kernel, dtype=tf.keras.backend.floatx())

    def call(self, inputs, **kwargs):
        """
        Call the layer.

        Args:
            inputs (tf.Tensor): Input tensor.

        Returns:
            (tf.Tensor): Result tensor.
        """

        return tf.nn.depthwise_conv2d(inputs, self.kernel, strides=(1,) + self.strides + (1,), padding='VALID')

    def compute_output_shape(self, input_shape):
        """
        Compute the output shape for the input shape.

        Args:
            input_shape (tuple): Input shape.

        Returns:
            (tuple): Output shape.
        """

        input_shape = tensor_shape.TensorShape(input_shape).as_list()
        space = input_shape[1:-1]
        new_space = [conv_utils.conv_output_length(input_length=space[i], filter_size=self.filter_size, padding='valid', stride=self.strides[i], dilation=1) for i in range(len(space))]
        return tensor_shape.TensorShape([input_shape[0]] + new_space + [input_shape[-1]])

    def get_config(self):
        """
        Get the configuration.

        Returns:
            (dict): Configuration to restore the layer.
        """

        self_config = {'filter_size': self.filter_size, 'stride': self.strides[0]}
        base_config = super(BlurPool, self).get_config()
        return {**base_config, **self_config}
