"""
This file contains a network model builder class.
"""

from ..errors import configerrors as dmsconfigerrors

import logging

#----------------------------------------------------------------------------------------------------

class ModelBuilder(object):
    """Network model builder."""

    def __init__(self):
        """Initialize object."""

        # Initialize the base class.
        #
        super().__init__()

        # Initialize members.
        #
        self.__model_config = None  # Model configuration.
        self.__patch_shapes = {}    # Patch shapes per level.
        self.__channel_count = 0    # Number of channels in the input.
        self.__class_count = 0      # Number of output classes.

        # Initialize logging.
        #
        self.__initlogging()

    def __initlogging(self):
        """Initialize logging."""

        # Configure logging. This class relies on configured logging somewhere down on the hierarchy.
        #
        qualified_class_name = '{module_name}.{class_name}'.format(module_name=self.__class__.__module__, class_name=self.__class__.__name__)
        self.__logger = logging.getLogger(name=qualified_class_name)

    def setmodel(self, config):
        """
        Configure model parameters.

        Args:
            config (dict): Model parameters. See 'model' section of parameter configuration example.
        """

        self.__model_config = config

    def setdata(self, patch_shapes, channel_count, class_count):
        """
        Configure the data parameters.

        Args:
            patch_shapes (dict): Patch shapes per spacing.
            channel_count (int): Number of channels in the input images.
            class_count (int): Number of output classes.
        """

        self.__patch_shapes = patch_shapes
        self.__channel_count = channel_count
        self.__class_count = class_count

    def build(self):
        """
        Construct network model object.

        Returns:
            diagmodels.models.modelbase.ModelBase: Constructed model.

        Raises:
            EmptyPatchShapesError: The patch shapes map is empty.
            InvalidChannelCountError: The number of input channels is invalid.
            InvalidClassCountError: The number of output classes is invalid.
            UnknownBackendTypeError: Unknown backend type.
            InvalidBackendTypeForModelError: The backend is invalid for the given model.
            UnknownModelTypeError: Unknown model type.

            DigitalPathologyModelError: Model errors.
        """

        self.__logger.info('Configuring network model')

        # Log configuration.
        #
        self.__logger.debug('Network model configuration: {config}'.format(config=self.__model_config))
        self.__logger.debug('Input shapes: {shapes}'.format(shapes=self.__patch_shapes))
        self.__logger.debug('Input channel count: {channels}'.format(channels=self.__channel_count))
        self.__logger.debug('Output class count: {classes}'.format(classes=self.__class_count))

        # Check the data configuration.
        #
        if not self.__patch_shapes:
            raise dmsconfigerrors.EmptyPatchShapesError()

        if self.__channel_count is None or self.__channel_count < 1:
            raise dmsconfigerrors.InvalidChannelCountError(self.__channel_count)

        if self.__class_count is None or self.__class_count < 2:
            raise dmsconfigerrors.InvalidClassCountError(self.__class_count)

        # Select the patch shape.
        #
        spacing = list(self.__patch_shapes.keys())[0]

        if self.__model_config['backend'] == 'keras':
            patch_shape = tuple(self.__patch_shapes[spacing]) + (self.__channel_count,)
        elif self.__model_config['backend'] == 'tensorflow':
            patch_shape = tuple(self.__patch_shapes[spacing]) + (self.__channel_count,)
        elif self.__model_config['backend'] == 'pytorch':
            patch_shape = (self.__channel_count,) + tuple(self.__patch_shapes[spacing])
        else:
            # The given backend type is unknown.
            #
            self.__logger.error('Unknown backend type: \'{backend_type}\''.format(backend_type=self.__model_config['backend']))
            raise dmsconfigerrors.UnknownBackendTypeError(self.__model_config['backend'])

        # Check model type and construct a network accordingly.
        #
        if self.__model_config['type'] == 'naturenet':
            if self.__model_config['backend'] == 'keras':
                import diagmodels.models.keras.naturenet as dmsnaturenetmodel

            else:
                # The given backend type is invalid.
                #
                self.__logger.error('Invalid \'{backend}\' backend type for \'{name}\' model.'.format(backend=self.__model_config['backend'], name=self.__model_config['type']))
                raise dmsconfigerrors.InvalidBackendTypeForModelError(backend=self.__model_config['backend'], model=self.__model_config['type'])

            model = dmsnaturenetmodel.NatureNet(name=self.__model_config['name'], description=self.__model_config['description'])

            model.configure(input_shape=patch_shape,
                            classes=self.__class_count,
                            branching_factor=self.__model_config['branching factor'],
                            batch_norm=self.__model_config['batch norm'],
                            dropout_count=self.__model_config['dropout layers'],
                            dropout_prob=self.__model_config['dropout probability'],
                            l2_lambda=self.__model_config['L2 lambda'],
                            channels_first=self.__model_config['channels first'])

        elif self.__model_config['type'] == 'unet':
            if self.__model_config['backend'] == 'keras':

                import diagmodels.models.keras.unet as dmsunetmodel

                model = dmsunetmodel.UNet(name=self.__model_config['name'], description=self.__model_config['description'])

                model.configure(input_shape=patch_shape,
                                depth=self.__model_config['depth'],
                                classes=self.__class_count,
                                branching_factor=self.__model_config['branching factor'],
                                batch_norm=self.__model_config['batch norm'],
                                dropout_count=self.__model_config['dropout layers'],
                                dropout_prob=self.__model_config['dropout probability'],
                                l2_lambda=self.__model_config['L2 lambda'],
                                padding=self.__model_config['padding'],
                                residual=self.__model_config['residual'],
                                downsampling=self.__model_config['downsampling'],
                                upsampling=self.__model_config['upsampling'],
                                channels_first=self.__model_config['channels first'],
                                loss=self.__model_config['loss'],
                                temperature_1=self.__model_config['loss_temperature_1'],
                                temperature_2=self.__model_config['loss_temperature_2'])

            else:
                # The given backend type is invalid.
                #
                self.__logger.error('Invalid \'{backend}\' backend type for \'{name}\' model.'.format(backend=self.__model_config['backend'], name=self.__model_config['type']))
                raise dmsconfigerrors.InvalidBackendTypeForModelError(backend=self.__model_config['backend'], model=self.__model_config['type'])

        elif self.__model_config['type'] == 'deeplab':
            if self.__model_config['backend'] == 'pytorch':

                import diagmodels.models.pytorch.deeplabnet as dmsdeeplabmodel

                model = dmsdeeplabmodel.DeepLabNet(name=self.__model_config['name'], description=self.__model_config['description'])

                model.configure(input_shape=patch_shape,
                                classes=self.__class_count,
                                backbone=self.__model_config['backbone'])

            else:
                # The given backend type is invalid.
                #
                self.__logger.error('Invalid \'{backend}\' backend type for \'{name}\' model.'.format(backend=self.__model_config['backend'], name=self.__model_config['type']))
                raise dmsconfigerrors.InvalidBackendTypeForModelError(backend=self.__model_config['backend'], model=self.__model_config['type'])

        elif self.__model_config['type'] == 'inception':
            # Inception is only implemented in Keras.
            #
            if self.__model_config['backend'] == 'keras':

                import diagmodels.models.keras.inception_v3 as dmsinceptionmodel

                model = dmsinceptionmodel.InceptionV3(name=self.__model_config['name'], description=self.__model_config['description'])

                model.configure(input_shape=patch_shape,
                                classes=self.__class_count,
                                include_top=self.__model_config['include top'],
                                preload_weights=self.__model_config['preload weights'],
                                use_aux_classifier=self.__model_config['use aux classifier'],
                                depth_multiplier=self.__model_config['depth multiplier'],
                                min_depth=self.__model_config['minimum depth'],
                                channels_first=self.__model_config['channels first'])
            else:
                # The given backend type is invalid.
                #
                self.__logger.error('Invalid \'{backend}\' backend type for \'{name}\' model.'.format(backend=self.__model_config['backend'], name=self.__model_config['type']))
                raise dmsconfigerrors.InvalidBackendTypeForModelError(backend=self.__model_config['backend'], model=self.__model_config['type'])

        elif self.__model_config['type'] == 'inception_v4':
            # Inception is only implemented in Keras.
            #
            if self.__model_config['backend'] == 'keras':

                import diagmodels.models.keras.inception_v4 as dmsinceptionmodel

                model = dmsinceptionmodel.Inceptionv4(name=self.__model_config['name'], description=self.__model_config['description'])

                model.configure(input_shape=patch_shape,
                                classes=self.__class_count,
                                load_weights=self.__model_config['preload weights'],
                                dropout_prob=self.__model_config['dropout probability'],
                                l2_lambda=self.__model_config['L2 lambda'],
                                channels_first=self.__model_config['channels first'])

            else:
                # The given backend type is invalid.
                #
                self.__logger.error('Invalid \'{backend}\' backend type for \'{name}\' model.'.format(backend=self.__model_config['backend'], name=self.__model_config['type']))
                raise dmsconfigerrors.InvalidBackendTypeForModelError(backend=self.__model_config['backend'], model=self.__model_config['type'])

        elif self.__model_config['type'] == 'fcnn':
            # Inception is only implemented in Keras.
            #
            if self.__model_config['backend'] == 'keras':

                import diagmodels.models.keras.general_cnn as dmsfcnn

                model = dmsfcnn.general_FCNN(name=self.__model_config['name'], description=self.__model_config['description'])

                model.configure(input_shape=patch_shape,
                                depth=self.__model_config['depth'],
                                classes=self.__class_count,
                                branching_factor=self.__model_config['branching factor'],
                                batch_norm=self.__model_config['batch norm'],
                                dropout_count=self.__model_config['dropout layers'],
                                dropout_prob=self.__model_config['dropout probability'],
                                l2_lambda=self.__model_config['L2 lambda'],
                                channels_first=self.__model_config['channels first'])

            else:
                # The given backend type is invalid.
                #
                self.__logger.error('Invalid \'{backend}\' backend type for \'{name}\' model.'.format(backend=self.__model_config['backend'], name=self.__model_config['type']))
                raise dmsconfigerrors.InvalidBackendTypeForModelError(backend=self.__model_config['backend'], model=self.__model_config['type'])

        elif self.__model_config['type'] == 'densenet':
            # DenseNet is only implemented in TensorFlow.
            #
            if self.__model_config['backend'] == 'tensorflow':

                import diagmodels.models.tensorflow.densenet as dmsdensenet

                model = dmsdensenet.DenseNet(name=self.__model_config['name'], description=self.__model_config['description'])

                model.configure(input_shape=patch_shape,
                                valid_padding=self.__model_config['valid padding'],
                                pool_initial=self.__model_config['pool initial'],
                                pool_final=self.__model_config['pool final'],
                                blur_pool=self.__model_config['blur pool'],
                                blur_size=self.__model_config['blur size'],
                                growth_rate=self.__model_config['growth rate'],
                                classes=self.__class_count,
                                layers_per_block=self.__model_config['layers per block'],
                                bottleneck_rate=self.__model_config['bottleneck rate'],
                                weight_decay=self.__model_config['weight decay'],
                                compression=self.__model_config['compression'],
                                dropout_rate=self.__model_config['dropout rate'],
                                loss_weights=self.__model_config['loss weights'],
                                label_smoothing=self.__model_config['label smoothing'])

            else:
                # The given backend type is invalid.
                #
                self.__logger.error('Invalid \'{backend}\' backend type for \'{name}\' model.'.format(backend=self.__model_config['backend'], name=self.__model_config['type']))
                raise dmsconfigerrors.InvalidBackendTypeForModelError(backend=self.__model_config['backend'], model=self.__model_config['type'])

        else:
            # The given model type is unknown.
            #
            self.__logger.error('Unknown model type: \'{model_type}\''.format(model_type=self.__model_config['type']))
            raise dmsconfigerrors.UnknownModelTypeError(self.__model_config['type'])

        # Log the input dimension order for the model.
        #
        self.__logger.debug('Model dimension order: \'{dimension_order}\''.format(dimension_order=model.dimensionorder.upper()))

        # Return the configured network. It is still not built, just configured.
        #
        return model

    @staticmethod
    def instantiate(network_path):
        """
        Construct network model object.

        Args:
            network_path (str): Network file path.

        Returns:
            diagmodels.models.modelbase.ModelBase: Constructed model.

        Raises:
            InvalidModelDataFormatError: Invalid model data format.
        """

        import diagmodels.models.modelbase as dptmodelbase

        return dptmodelbase.ModelBase.instantiate(file=network_path)
