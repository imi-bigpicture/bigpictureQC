"""
U-Net implementation using Keras.
"""

from diagmodels.models.keras import kerasmodelbase as dmskerasmodelbase
from diagmodels.errors import modelerrors as dmsmodelerrors
from diagmodels.models.keras import losses as dmskeraslosses
from diagmodels.models.keras import metrics as dmskerasmetrics

import tensorflow as tf
from keras import models, layers, optimizers, regularizers
from keras.layers import Lambda
import keras.backend as K

import numpy as np

# ----------------------------------------------------------------------------------------------------
def SubpixelConv2D(input_shape, scale=4, name="Lambda"):
    """
    Keras layer to do subpixel convolution.
    NOTE: Tensorflow backend only. Uses tf.depth_to_space
    Ref:
        [1] Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional Neural Network
            Shi et Al.
            https://arxiv.org/abs/1609.05158
    :param input_shape: tensor shape, (batch, height, width, channel)
    :param scale: upsampling scale. Default=4
    :return:
    """
    # upsample using depth_to_space
    def subpixel_shape(input_shape):
        dims = [input_shape[0],
                input_shape[1] * scale,
                input_shape[2] * scale,
                int(input_shape[3] / (scale ** 2))]
        output_shape = tuple(dims)
        return output_shape

    def subpixel(x):
        return tf.nn.depth_to_space(x, scale)

    return Lambda(subpixel, output_shape=subpixel_shape, name=name)

def BilinearResize(input_shape, scale=2.0, name="Labda"):
    """
    Keras layer to do bilinear resizing
    NOTE: Tensorflow backend only. Uses tf.depth_to_space
    :param input_shape: tensor shape, (batch, height, width, channel)
    :param scale: upsampling scale. Default=2
    :return:
    """
    # upsample using depth_to_space
    def subpixel_shape(input_shape):
        dims = [input_shape[0],
                int(input_shape[1] * scale),
                int(input_shape[2] * scale),
                input_shape[3]]
        output_shape = tuple(dims)
        return output_shape

    def subpixel(x):
        x = tf.image.resize(x, (int(input_shape[1] * scale), int(input_shape[2] * scale)), method=tf.image.ResizeMethod.BILINEAR)
        x.set_shape((None, int(input_shape[1] * scale), int(input_shape[2] * scale), None))
        return x

    return Lambda(subpixel, output_shape=subpixel_shape, name=name)

# ----------------------------------------------------------------------------------------------------

class UNet(dmskerasmodelbase.KerasModelBase):
    """U-Net model in Keras"""

    def __init__(self, name=None, description=None):
        """
        Initialize model parameters.
        Args:
            name (str): Name of the model.
            description (str): Description of the model.
        """

        # Initialize base class.
        #
        super().__init__(name, description)

        # Initialize members.
        #
        self.__input_shape = None  # Input shape: (rows, cols, channels).
        self.__output_shape = None  # Output shape: (rows, cols, channels
        self.__depth = 4  # Depth of the network
        self.__branching_factor = 0  # Filter count branching factor.
        self.__classes = 0  # Number of output classes.
        self.__batch_norm = False  # Batch normalization.
        self.__dropout_prob = 0.0  # Probability of dropout on the dropout layers.
        self.__dropout_count = 0  # Number of dropout layers.
        self.__residual = False  # Residual connections
        self.__downsampling = 'maxpool'  # Downsampling method
        self.__upsampling = 'upsampling2d'  # Upsampling method
        self.__output_offset = None  # Difference between input and output
        self.__padding = 'valid'  # Padding mode.
        self.__l2_lambda = 0.0  # L2 regularizer
        self.__loss = 'categorical_crossentropy'  # Loss function
        self.__output_offset_aux_path = None
        self.__temperature_1 = 0.2
        self.__temperature_2 = 2.0

    def configure(self,
                  input_shape,
                  depth,
                  classes,
                  branching_factor,
                  batch_norm,
                  dropout_count,
                  dropout_prob,
                  l2_lambda,
                  padding,
                  residual,
                  downsampling,
                  upsampling,
                  channels_first,
                  loss,
                  temperature_1,
                  temperature_2):
        """
        Save the network configuration parameters.
        Args:
            input_shape (tuple): Number of channels and size of the individual input images. (channels, rows, cols)
            depth (int): Depth of the U-Net.
            classes (int): Number of output classes.
            branching_factor (int): Branching factor. The number of filters on the first layer is 2 on the power of branching factor.
            batch_norm (bool): Use batch normalization.
            dropout_count (int): Number of dropout layers.
            dropout_prob (float): Dropout probability.
            l2_lambda (float): L2 loss lambda.
            padding (string): Type of padding to use.
            residual (bool): Use residual connections.
            downsampling (string): One of 'maxpool' or 'strided'
            upsampling (string): One of 'transposedconv' or 'upsampling2d'
            channels_first (bool): If true, the network expect patches shaped BCHW, otherwise BHWC.
            loss (string): 'lovasz', 'categorical_crossentropy', 'dice_cc', 'lovasz_cc' or 'dice'.
            temperature_1 (float): value for t1 if bi-tempered or focal loss is used.
            temperature_2 (float): value for t2 if bi-tempered or focal loss is used.
            loss (string): 'categorical_crossentropy'.

        Raises:
            InvalidInputShapeError: The input shape is not valid.
            InvalidModelClassCountError: The number of classes is not valid.
            InvalidBranchingFactorError: The branching factor is not valid.
            InvalidDropoutLayerCountError: The number of dropout layers is not valid.
            InvalidDropoutProbabilityError: The dropout probability is out of (0.0, 1.0) bounds.
            NotSupportedChannelOrder: The dimension order is not supported.
            NotSupportedLoss: The loss is not supported.
            InvalidNumberOfClassesForLoss: The loss function doesn't support this amount of classes.
        """

        # Check input parameters.
        #
        if len(input_shape) != 3 or min(input_shape) <= 0:
            raise dmsmodelerrors.InvalidInputShapeError(input_shape)

        if classes <= 1:
            raise dmsmodelerrors.InvalidModelClassCountError(classes)

        if branching_factor < 0:
            raise dmsmodelerrors.InvalidBranchingFactorError(branching_factor)

        if dropout_count < 0 or depth < dropout_count:
            raise dmsmodelerrors.InvalidDropoutLayerCountError(dropout_count)

        if dropout_count > 0 and (dropout_prob <= 0.0 or 1.0 <= dropout_prob):
            raise dmsmodelerrors.InvalidDropoutProbabilityError(dropout_prob)

        if padding not in ['same', 'valid']:
            raise dmsmodelerrors.NotSupportedPaddingError(padding, 'conv2d')

        if channels_first:
            raise dmsmodelerrors.NotSupportedChannelOrder('channels first')

        if loss not in ['categorical_crossentropy', 'lovasz', 'lovasz_cc', 'focal', 'focal_lovasz', 'tempered']:
            raise dmsmodelerrors.NotSupportedLoss(loss)

        # Save parameters.
        #
        self.__input_shape = input_shape
        self.__output_shape = None
        self.__depth = depth
        self.__classes = classes
        self.__branching_factor = branching_factor
        self.__batch_norm = batch_norm
        self.__dropout_count = dropout_count
        self.__dropout_prob = dropout_prob
        self.__residual = residual
        self.__downsampling = downsampling
        self.__upsampling = upsampling
        self.__padding = padding
        self.__l2_lambda = l2_lambda
        self.__loss = loss
        self.__temperature_1 = temperature_1
        self.__temperature_2 = temperature_2

    def _customdata(self):
        """
        Get a map of custom data structure to add to the dictionary of exported values.
        Returns:
            dict: Dictionary of values to add to the exported data.
        """

        # Collect custom data.
        #
        return {'depth': self.__depth,
                'classes': self.__classes,
                'branching_factor': self.__branching_factor,
                'batch_norm': self.__batch_norm,
                'dropout_count': self.__dropout_count,
                'dropout_prob': self.__dropout_prob,
                'l2_lambda': self.__l2_lambda,
                'padding': self.__padding,
                'upsampling': self.__upsampling,
                'downsampling': self.__downsampling,
                'residual': self.__residual,
                'loss': self.__loss,
                'output_shape': self.__output_shape,
                'temperature_1': self.__temperature_1,
                'temperature_2': self.__temperature_2}

    def _setcustomdata(self, data_maps):
        """
        Set the class specific data.
        Args:
            data_maps (dict): Custom data map.
        """

        # Configure custom data.
        #
        self.__depth = data_maps.get('depth', 0)
        self.__classes = data_maps.get('classes', 0)
        self.__branching_factor = data_maps.get('branching_factor', 0)
        self.__batch_norm = data_maps.get('batch_norm', False)
        self.__dropout_count = data_maps.get('dropout_count', 0)
        self.__dropout_prob = data_maps.get('dropout_prob', 0.0)
        self.__l2_lambda = data_maps.get('l2_lambda', 0.0)
        self.__padding = data_maps.get('padding', None)
        self.__downsampling = data_maps.get('downsampling', 'maxpool')
        self.__upsampling = data_maps.get('upsampling', 'upsampling2d')
        self.__loss = data_maps.get('loss', 'categorical_crossentropy')
        self.__output_shape = data_maps.get('output_shape')
        self.__temperature_1 = data_maps.get('temperature_1')
        self.__temperature_2 = data_maps.get('temperature_2')

    def build(self):
        """Build the network instance with the pre-configured parameters."""

        if 'tempered' in self.__loss:
            inputs, outputs = self.__networkdefinition_without_final()
        else:
            inputs, outputs = self.__networkdefinition()

        if "categorical_crossentropy" in self.__loss:
            self._model_instance = models.Model(inputs, outputs)

        else:
            weight_shape = (self.__output_shape[1] * self.__output_shape[2], 1)
            input_weights = layers.Input(shape=weight_shape, name='input_weights')

            self._model_instance = models.Model([inputs, input_weights], outputs)

        # Define optimizer
        #
        optimizer = optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999)

        # Compile model
        #
        if self.__loss == "categorical_crossentropy":
            self._model_instance.compile(optimizer=optimizer, loss=['categorical_crossentropy'], sample_weight_mode="temporal", metrics=['accuracy',
                                                                                                                                         dmskerasmetrics.dice_coef,
                                                                                                                                         dmskerasmetrics.iou_coef])

        else:
            # Initialize loss functions
            #
            loss_functions = dmskeraslosses.LossFunctions(classes=self.__classes,
                                                          loss=self.__loss,
                                                          weights=input_weights,
                                                          temperature_1=self.__temperature_1,
                                                          temperature_2=self.__temperature_2)

            loss_function = loss_functions.getLossFunction()


            self._model_instance.compile(optimizer=optimizer, loss=[loss_function],
                                         metrics=['accuracy', dmskerasmetrics.dice_coef, dmskerasmetrics.iou_coef])

        self._compiled = True

        # Retrieve output shape of the intermediate output layer to reshape predictions.
        # TODO: Related to issue #23. Remove commented lines once fixed.
#         self._model_instance.metrics_tensors += self._model_instance.outputs
#         self._model_instance.metrics_names += ['predictions', 'predictions_aux'] if "aux_loss" in self.__loss else ['predictions']

        self._model_predict_model = models.Model([self._model_instance.layers[0].input],
                                                 [self._model_instance.get_layer('output').output])

#         self._addcustommetric(name='errors')

    def _restoremodelparameters(self, parameters):
        """
        Custom restore function. Makes sure that the output tensors are correctly set, even after a load.
        Args:
            parameters (dict): Parameters of the model.
        """

        # Load custom loss functions and metrics, otherwise keras will crash.
        #
        input_weights = layers.Input(shape=(self.__output_shape[1] * self.__output_shape[2], 1), name='input_weights')

        from keras.utils.generic_utils import get_custom_objects
        custom_iou_coef = dmskerasmetrics.iou_coef
        custom_dice_coef = dmskerasmetrics.dice_coef

        # Initialize loss functions
        #
        if self.__loss == "categorical_crossentropy":
            get_custom_objects().update({'tf': tf,
                                         'iou_coef': custom_iou_coef,
                                         'dice_coef': custom_dice_coef})

        else:
            loss_functions = dmskeraslosses.LossFunctions(classes=self.__classes,
                                                          loss=self.__loss,
                                                          weights=input_weights,
                                                          temperature_1=self.__temperature_1,
                                                          temperature_2=self.__temperature_2)

            custom_loss_function = loss_functions.getLossFunction()

            get_custom_objects().update({'tf': tf,
                                         'iou_coef': custom_iou_coef,
                                         'dice_coef': custom_dice_coef,
                                         custom_loss_function.__name__: custom_loss_function})

        super()._restoremodelparameters(parameters)

        # Explicitly reset the train and validate functions of the internal model
        # TODO this is a workaround for https://github.com/fchollet/keras/issues/8468
        #
        self._model_instance.train_function = None
        self._model_instance.test_function = None
        
        # TODO: Related to issue #23. Remove commented lines once fixed.
        # self._model_instance.metrics_tensors += self._model_instance.outputs
        # self._model_instance.metrics_names += ['predictions', 'predictions_aux'] if "aux_loss" in self.__loss else ['predictions']

        self._model_predict_model = models.Model([self._model_instance.layers[0].input],
                                                 [self._model_instance.get_layer('output').output])


    def update(self, x, y, sample_weight=None, class_weight=None, *args, **kwargs):
        """
        Update the network.
        Args:
            x (numpy.ndarray): an array containing image data.
            y (numpy.ndarray ): an array containing label data.
            sample_weights (numpy.ndarray): an array containing weights for samples to increase their contribution to the loss
            class_weights (numpy.ndarray): an array containing weights for classes to increase their contribution to the loss
        Returns:
            dict: Output of the update function.
        """

        # Match  weights to the output of the network. Apply cropping if required. Also flatten the weights and y as Keras does not support 3D sample weight maps
        #
        sample_weight = self.__matchdatatonetwork(sample_weight)

        if sample_weight is not None:
            sample_weight = sample_weight.reshape(sample_weight.shape[0],
                                                  sample_weight.shape[1] * sample_weight.shape[2])

        # Match labels to the output of the network. Apply cropping if required.
        #
        y = self.__matchdatatonetwork(y)
        y = y.reshape(y.shape[0], y.shape[1] * y.shape[2], y.shape[3])

        # Train on batch
        #
        if self.__loss == "categorical_crossentropy":
            output = super().update(x, y, sample_weight, class_weight, *args, **kwargs)

        else:
            output = super().update([x, np.expand_dims(sample_weight, axis=-1)],
                                    y,
                                    sample_weight=None,
                                    class_weight=class_weight,
                                    *args, **kwargs)

        # calculate error
        # TODO: Related to issue #23?! Update code once fixed.
        if 'predictions' in output.keys():
            weight_sum = np.sum(sample_weight, axis=-1).reshape(sample_weight.shape[0], 1)
            error_flat = (np.max((y - output['predictions']), axis=-1) * sample_weight) / weight_sum
            output['errors'] = np.clip(np.sum(error_flat, axis=-1), 0, 1)

            # Reshape the predictions.
            #
            output['predictions'] = output['predictions'].reshape((-1,) + self.__output_shape[1:])

        return output

    def validate(self, x, y, sample_weight=None, *args, **kwargs):
        """
        Validate the network.
        Args:
            x (numpy.ndarray or list of numpy.ndarray): contains image data.
            y (numpy.ndarray or list of numpy.ndarray ): contains label data.
            sample_weights (numpy.ndarray): an array containing weights for samples to increase their contribution to the loss
        Returns:
            dict: Output of the validation function.
        """

        # Match  weights to the output of the network. Apply cropping if required.
        #
        sample_weight = self.__matchdatatonetwork(sample_weight)

        # Flatten the weights and y as Keras does not support 3D sample weight maps
        #
        if sample_weight is not None:
            sample_weight = sample_weight.reshape(sample_weight.shape[0],
                                                  sample_weight.shape[1] * sample_weight.shape[2])

        # Match labels to the output of the network. Apply cropping if required.
        #
        y = self.__matchdatatonetwork(y)
        y = y.reshape(y.shape[0], y.shape[1] * y.shape[2], y.shape[3])

        # Validate batch
        #
        if self.__loss == "categorical_crossentropy":
            output = super().validate(x, y, sample_weight, *args, **kwargs)

        else:
            output = super().validate([x, np.expand_dims(sample_weight, axis=-1)], y, sample_weight=None, *args, **kwargs)

        # Reshape the predictions.
        # TODO: Related to issue #23?! Update code once fixed.
        if 'predictions' in output.keys():
            output['predictions'] = output['predictions'].reshape((-1,) + self.__output_shape[1:])

        return output

    def predict(self, x, *args, **kwargs):
        """
        Use the network for evaluation.
        Args:
            x (numpy.ndarray or list of numpy.ndarray): contains image data.
        Returns:
            dict: Output of the evaluation function.
        """

        return {"predictions": self._model_predict_model.predict_on_batch(x)}

    @property
    def inputshape(self):
        """
        Return input shape of unet
        Returns:
            list: input shape
        """

        if not self._compiled:
            raise dmsmodelerrors.ModelNotCompiledError()

        if not self.__input_shape:
            self.__input_shape = self._model_instance.input_shape

        return self.__input_shape

    @property
    def outputshape(self):
        """
        Return output shape of unet
        Returns:
            list: Output shape
        Raises:
            ModelNotCompiledError
        """

        if not self._compiled:
            raise dmsmodelerrors.ModelNotCompiledError()

        if not self.__output_shape:
            self.__output_shape = self._model_instance.get_layer('output').output_shape

        return self.__output_shape

    def __conv_block(self, input, filters, activation, l2, dropout_enabled, name):
        """
        Build a single convolution block of the U-Net.
        Args:
            input: Network input at this point.
            filters (int): Number of filters.
            activation (string|callable): Activation function for the convolution layers.
            l2 (Regularizer): Regularizer object
            dropout_enabled (bool): Whether to add dropout in this layer
        """

        net = layers.Conv2D(filters, 3, activation=activation, padding=self.__padding, kernel_regularizer=l2, name=name + "_conv_1")(input)

        if self.__batch_norm:
            net = layers.BatchNormalization(name=name + "_batch_norm_1")(net)

        if dropout_enabled and self.__dropout_prob > 0:
            # Dropout per channel.
            #
            noise_shape = (None, 1, 1, filters)

            net = layers.Dropout(self.__dropout_prob, noise_shape=noise_shape, name=name + "_dropout")(net)

        net = layers.Conv2D(filters, 3, activation=activation, padding=self.__padding, kernel_regularizer=l2, name=name + "_conv_2")(net)

        if self.__batch_norm:
            net = layers.BatchNormalization(name=name + "_batch_norm_2")(net)

        if self.__residual:
            total_crop = K.int_shape(input)[1] - K.int_shape(net)[1]
            halve_crop = total_crop // 2

            cropped_input = layers.Cropping2D(
                ((halve_crop, total_crop - halve_crop), (halve_crop, total_crop - halve_crop)))(input)

            net = layers.Concatenate()([cropped_input, net])

        return net

    def __level_block(self, net, filters, depth, activation, l2):
        """
        Recursively build the U-Net.
        Args:
            net: Current network object.
            filters (int): Number of filters on this level.
            depth (int): Depth at this stage.
            activation (string|callable): Activation function for layers.
            l2 (Regularizer): Regularizer object
        """

        dropout_enabled = self.__dropout_count >= depth
        depth_name = self.__depth - depth + 1
        if depth > 1:
            first_block = self.__conv_block(input=net, filters=filters, activation=activation, l2=l2, dropout_enabled=dropout_enabled, name="down_{}".format(depth_name))

            if self.__downsampling == 'maxpool':
                net = layers.MaxPooling2D(pool_size=2, strides=2, name="down_MP_{}".format(depth_name))(first_block)
            elif self.__downsampling == 'bilinear':
                net = BilinearResize(K.int_shape(first_block), scale=0.5, name="bilinear_down_{}".format(self.__depth - depth))(first_block)
            else:
                net = layers.Conv2D(filters, 3, strides=2, padding=self.__padding, kernel_regularizer=l2, name="down_MP_{}".format(depth_name))(first_block)

            # Recursively go a level deeper.
            #
            net = self.__level_block(net=net, filters=2 * filters, depth=depth - 1, activation=activation, l2=l2)

            if depth == 2:
                network_shape = net._shape_tuple()[1:3]
                # self.__aux_path = self.__auxiliary_network(net, network_shape, filters, activation, l2)

            if self.__upsampling == 'transposedconv':
                net = layers.Conv2DTranspose(filters, 2, strides=2, activation=activation, padding=self.__padding, kernel_regularizer=l2, name="Upsampling_{}".format(depth_name))(net)
            elif self.__upsampling == 'subpixel':
                scale = 2
                net = layers.Conv2D(filters * scale ** 2, 3, activation=activation, padding=self.__padding)(net)
                net = SubpixelConv2D(K.int_shape(net), scale=scale, name="bilinear_up_{}".format(self.__depth - depth))(net)
                net = layers.AvgPool2D(strides=1, padding=self.__padding)(net)
            elif self.__upsampling == 'bilinear':
                net = BilinearResize(K.int_shape(net), scale=2.0, name="bilinear_up_{}".format(self.__depth - depth))(net)
            else:
                net = layers.UpSampling2D()(net)
                net = layers.Conv2D(filters, 2, activation=activation, padding=self.__padding, kernel_regularizer=l2, name="Upsampling_{}".format(depth_name))(net)

            # We use valid padding so make sure the two blocks are the same size. Cropping is done from the center with
            # any remainder (in case of uneven crop) will be cropped from the bottom and right.
            #
            total_crop = K.int_shape(first_block)[1] - K.int_shape(net)[1]
            halve_crop = total_crop // 2

            cropped_block = layers.Cropping2D(((halve_crop, total_crop - halve_crop), (halve_crop, total_crop - halve_crop)))(first_block)

            # Connect two paths of the U-Net.
            #
            net = layers.Concatenate()([cropped_block, net])

            # Add last conv block.
            #
            net = self.__conv_block(input=net, filters=filters, activation=activation, l2=l2, dropout_enabled=dropout_enabled, name="Upward_conv_block_{}".format(depth_name))

        else:
            net = self.__conv_block(input=net, filters=filters, activation=activation, l2=l2, dropout_enabled=dropout_enabled, name="Bottom_conv_block")

        return net

    def __networkdefinition(self):
        """
        Network definition.
        Returns
            Tensor: The network architecture.
        """

        input = layers.Input(shape=self.__input_shape, name='input')
        l2 = regularizers.l2(self.__l2_lambda)

        net = self.__level_block(input, 2 ** self.__branching_factor, self.__depth, 'relu', l2)

        output = layers.Conv2D(self.__classes, 1, activation='softmax', name='output', kernel_regularizer=l2)(net)

        # Flatten the output to (width * height, n_classes) as Keras does not support 3D sample weight maps
        #
        output_shape = K.int_shape(output)

        flatten = layers.Reshape((output_shape[1] * output_shape[2], output_shape[3]), name="output_flat")(output)

        self.__output_shape = output_shape

        return input, flatten

    def __matchdatatonetwork(self, data):
        """
        Slice label or weight patches based on the output size of the network. Values on the edges are discarded. Does nothing if the output size matches the label size.
        Args:
            data (np.array): A numpy array containing the labels or weights
        Returns:
            np.array: The sliced labels
        """

        # No need to reshape if the shape of the labels matches the output shape.
        #
        if self.__output_shape[1:3] == data.shape[1:3]:
            return data

        # Compute the offset based on the difference between input and output.
        #
        if self.__output_offset is None:
            # Retrieve all layers that perform an operation on the image (filter out other ones)
            #
            selected_layers = [l for l in self._model_instance.layers if "aux_" not in l.name and not any(isinstance(l, layer) for layer in [layers.InputLayer,
                                                                                                                                            layers.Dropout,
                                                                                                                                            layers.Cropping2D,
                                                                                                                                            layers.ReLU,
                                                                                                                                            layers.LeakyReLU,
                                                                                                                                            layers.PReLU,
                                                                                                                                            layers.ELU,
                                                                                                                                            layers.ThresholdedReLU,
                                                                                                                                            layers.Softmax,
                                                                                                                                            layers.BatchNormalization,
                                                                                                                                            layers.Concatenate,
                                                                                                                                            layers.Reshape,
                                                                                                                                            layers.Activation])]


            input_shape = selected_layers[0].input_shape

            if len(input_shape) == 3:
                input_shape = (1,) + tuple(input_shape)

            self.__output_offset, _, _ = self._getreconstructioninformationforlayers(input_shape, selected_layers)

        # Slice the labels based on the offset.
        #
        return data[:, self.__output_offset[0]:-self.__output_offset[1], self.__output_offset[2]:-self.__output_offset[3], :]

    def getreconstructioninformation(self, input_shape=None):
        # Retrieve all layers that perform an operation on the image (filter out other ones)
        #
        selected_layers = [l for l in self._model_instance.layers if "aux_" not in l.name and not any(isinstance(l, layer) for layer in [layers.InputLayer,
                                                                                                                                         layers.Dropout,
                                                                                                                                         layers.Cropping2D,
                                                                                                                                         layers.ReLU,
                                                                                                                                         layers.LeakyReLU,
                                                                                                                                         layers.PReLU,
                                                                                                                                         layers.ELU,
                                                                                                                                         layers.ThresholdedReLU,
                                                                                                                                         layers.Softmax,
                                                                                                                                         layers.BatchNormalization,
                                                                                                                                         layers.Concatenate,
                                                                                                                                         layers.Reshape,
                                                                                                                                         layers.Activation])]

        if input_shape is None:
            input_shape = selected_layers[0].input_shape
        if len(input_shape) == 3:
            input_shape = (1,) + tuple(input_shape)

        return self._getreconstructioninformationforlayers(input_shape, selected_layers)


    def __networkdefinition_without_final(self):
        input = layers.Input(shape=self.__input_shape, name='input')
        l2 = regularizers.l2(self.__l2_lambda)

        net = self.__level_block(input, 2 ** self.__branching_factor, self.__depth, 'relu', l2)

        output = layers.Conv2D(self.__classes, 1, activation=None, name='output', kernel_regularizer=l2)(net)

        # Flatten the output to (width * height, n_classes) as Keras does not support 3D sample weight maps
        #
        output_shape = K.int_shape(output)

        flatten = layers.Reshape((output_shape[1] * output_shape[2], output_shape[3]), name="output_flat")(output)

        self.__output_shape = output_shape

        return input, flatten

