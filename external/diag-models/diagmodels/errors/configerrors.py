"""
Network model configuration related errors.
"""

from . import errorbase as dmserrorbase

#----------------------------------------------------------------------------------------------------

class DiagModelsConfigError(dmserrorbase.DiagModelsError):
    """Error base class for all model configuration errors."""

    def __init__(self, *args):
        """
        Initialize the object.

        Args:
            *args: Argument list.
        """

        # Initialize base class.
        #
        super().__init__(*args)

#----------------------------------------------------------------------------------------------------

class EmptyPatchShapesError(DiagModelsConfigError):
    """Raise when the configured patch shape map is empty."""

    def __init__(self):
        """Initialize the object."""

        # Initialize base class.
        #
        super().__init__('Empty patch shapes.')

#----------------------------------------------------------------------------------------------------

class InvalidChannelCountError(DiagModelsConfigError):
    """Raise when the configured input channel count is invalid."""

    def __init__(self, channel_count):
        """
        Initialize the object.

        Args:
            channel_count (int, None): Input channel count.
        """

        # Initialize base class.
        #
        super().__init__('Invalid input channel count: {count}.'.format(count=channel_count))

        # Store custom data.
        #
        self.channel_count = channel_count

#----------------------------------------------------------------------------------------------------

class InvalidClassCountError(DiagModelsConfigError):
    """Raise when the configured output class count is invalid."""

    def __init__(self, class_count):
        """
        Initialize the object.

        Args:
            class_count (int, None): Output class count.
        """

        # Initialize base class.
        #
        super().__init__('Invalid output class count: {count}.'.format(count=class_count))

        # Store custom data.
        #
        self.class_count = class_count

#----------------------------------------------------------------------------------------------------

class UnknownBackendTypeError(DiagModelsConfigError):
    """Raise when the configured backend type is unknown."""

    def __init__(self, backend):
        """
        Initialize the object.

        Args:
            backend (str): Backend name.
        """

        # Initialize base class.
        #
        super().__init__('The \'{name}\' backend is unknown.'.format(name=backend))

        # Store custom data.
        #
        self.backend = backend

#----------------------------------------------------------------------------------------------------

class InvalidBackendTypeForModelError(DiagModelsConfigError):
    """Raise when the configured backend type is invalid for the given model."""

    def __init__(self, backend, model):
        """
        Initialize the object.

        Args:
            backend (str): Backend name.
            model (str): Model name.
        """

        # Initialize base class.
        #
        super().__init__('The \'{backend_name}\' backend is invalid for \'{model_name}\' model.'.format(backend_name=backend, model_name=model))

        # Store custom data.
        #
        self.backend = backend
        self.model = model

#----------------------------------------------------------------------------------------------------

class UnknownModelTypeError(DiagModelsConfigError):
    """Raise when the configured model type is unknown."""

    def __init__(self, name):
        """
        Initialize the object.

        Args:
            name (str): Model name.
        """

        # Initialize base class.
        #
        super().__init__('The \'{name}\' model is unknown.'.format(name=name))

        # Store custom data.
        #
        self.name = name
