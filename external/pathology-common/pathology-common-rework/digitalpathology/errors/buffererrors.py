"""
Buffer related errors.
"""

from . import errorbase as dpterrorbase

#----------------------------------------------------------------------------------------------------

class DigitalPathologyBufferError(dpterrorbase.DigitalPathologyError):
    """Error base class for all buffer errors."""

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

class InvalidCacheSizeError(DigitalPathologyBufferError):
    """Raise when the number of patches to cache is invalid."""

    def __init__(self, cache_size):
        """
        Initialize the object.

        Args:
            cache_size (int): Number of patches to keep in memory.
        """

        # Initialize base class.
        #
        super().__init__('Invalid cache size: {size}.'.format(size=cache_size))

        # Store custom data.
        #
        self.cache_size = cache_size

#----------------------------------------------------------------------------------------------------

class InvalidClassificationErrorError(DigitalPathologyBufferError):
    """Raise when the classification error value is out of [0.0, 1.0] interval."""

    def __init__(self, error):
        """
        Initialize the object.

        Args:
            error (float): Classification error.
        """

        # Initialize base class.
        #
        super().__init__('Classification error is out of [0.0, 1.0] interval: {error}.'.format(error=error))

        # Store custom data.
        #
        self.error = error

#----------------------------------------------------------------------------------------------------

class BatchSizeLargerThanCacheError(DigitalPathologyBufferError):
    """Raise when the batch size is larger than the number of cached patches."""

    def __init__(self, batch_size, cache_size):
        """
        Initialize the object.

        Args:
            batch_size (int): Batch size.
            cache_size (int): Cache size.
        """

        # Initialize base class.
        #
        super().__init__('Batch size is too large: {batch} ({cache}).'.format(batch=batch_size, cache=cache_size))

        # Store custom data.
        #
        self.batch_size = batch_size
        self.cache_size = cache_size

#----------------------------------------------------------------------------------------------------

class LabelCountMismatchError(DigitalPathologyBufferError):
    """Raise when the number of pushed patches and labels are not matching on a level of pixel spacing."""

    def __init__(self, spacing, patch_count, label_count):
        """
        Initialize the object.

        Args:
            spacing (float): Pixel spacing of mismatch (micrometer).
            patch_count (int): Number of patches.
            label_count (int): Number of labels.
        """

        # Initialize base class.
        #
        super().__init__('Pushed patch - label count mismatch: {patch} - {label} at spacing {spacing} um.'.format(patch=patch_count, label=label_count, spacing=spacing))

        # Store custom data.
        #
        self.spacing = spacing
        self.patch_count = patch_count
        self.label_count = label_count

#----------------------------------------------------------------------------------------------------

class BufferOverflowError(DigitalPathologyBufferError):
    """Raise when the number of patches to write to the buffer is larger than the size of the buffer on a level of pixel spacing."""

    def __init__(self, spacing, buffer_size, batch_size):
        """
        Initialize the object.

        Args:
            spacing (float): Pixel spacing of mismatch (micrometer).
            buffer_size (int): Size of the buffer.
            batch_size (int): Number of patches to push to the buffer.
        """

        # Initialize base class.
        #
        super().__init__('Cannot push {batch} item to a buffer of size {buffer} on spacing {spacing} um.'.format(batch=batch_size, buffer=buffer_size, spacing=spacing))

        # Store custom data.
        #
        self.spacing = spacing
        self.buffer_size = buffer_size
        self.batch_size = batch_size

#----------------------------------------------------------------------------------------------------

class SpacingMismatchError(DigitalPathologyBufferError):
    """Raise when pushed pixel spacings do not match the buffered pixel spacings."""

    def __init__(self, pushed_spacings, buffered_spacings):
        """
        Initialize the object.

        Args:
            pushed_spacings (tuple): List of pushed pixel spacings.
            buffered_spacings (tuple): List of buffered pixel spacings.
        """

        # Initialize base class.
        #
        super().__init__('Pushed {pushed} pixel spacings do not match the buffered {buffered} pixel spacings.'.format(pushed=pushed_spacings, buffered=buffered_spacings))

        # Store custom data.
        #
        self.pushed_spacings = pushed_spacings
        self.buffered_spacings = buffered_spacings

#----------------------------------------------------------------------------------------------------

class InvalidBufferChunkSizeError(DigitalPathologyBufferError):
    """Raise when the buffer chunk size is invalid."""

    def __init__(self, chunk_size):
        """
        Initialize the object.

        Args:
            chunk_size (int): Number of patches to transfer at once.
        """

        # Initialize base class.
        #
        super().__init__('Invalid buffer chunk size: {size}.'.format(size=chunk_size))

        # Store custom data.
        #
        self.chunk_size = chunk_size

#----------------------------------------------------------------------------------------------------

class InvalidDataTypeError(DigitalPathologyBufferError):
    """Raise when the data type of the patches to buffer is invalid."""

    def __init__(self, data_type):
        """
        Initialize the object.

        Args:
            data_type (type): Data type.
        """

        # Initialize base class.
        #
        super().__init__('Invalid patch data type: {data_type}.'.format(data_type=data_type))

        # Store custom data.
        #
        self.data_type = data_type

#----------------------------------------------------------------------------------------------------

class MissingWeightsError(DigitalPathologyBufferError):
    """Raise when the buffer is configured to store weights too but the pushed batch does not contain weights."""

    def __init__(self, spacing):
        """
        Initialize the object.

        Args:
            spacing (float): First level of pixel spacing where the weights are missing (micrometer).
        """

        # Initialize base class.
        #
        super().__init__('Buffer configured to store weights but is missing from the input at spacing: {spacing} um.'.format(spacing=spacing))

        # Store custom data.
        #
        self.spacing = spacing

#----------------------------------------------------------------------------------------------------

class WeightCountMismatchError(DigitalPathologyBufferError):
    """Raise when the number of pushed patches and weights are not matching on a level of pixel spacing."""

    def __init__(self, spacing, patch_count, weight_count):
        """
        Initialize the object.

        Args:
            spacing (float): Pixel spacing of mismatch (micrometer).
            patch_count (int): Number of patches.
            weight_count (int): Number of labels.
        """

        # Initialize base class.
        #
        super().__init__('Pushed patch - weight count mismatch: {patch} - {weight} at spacing {spacing} um.'.format(patch=patch_count, weight=weight_count, spacing=spacing))

        # Store custom data.
        #
        self.spacing = spacing
        self.patch_count = patch_count
        self.weight_count = weight_count

#----------------------------------------------------------------------------------------------------

class PatchDataTypeMismatchError(DigitalPathologyBufferError):
    """Raise when patch data type of the source buffer does not match the target buffer."""

    def __init__(self, target_data_type, source_data_type):
        """
        Initialize the object.

        Args:
            target_data_type (type): Patch data type of the target buffer.
            source_data_type (type): Patch data type of the source buffer.
        """

        # Initialize base class.
        #
        super().__init__('The transferred source {src} patch data type does not match the {dst} target.'.format(src=source_data_type, dst=target_data_type))

        # Store custom data.
        #
        self.target_data_type = target_data_type
        self.source_data_type = source_data_type

#----------------------------------------------------------------------------------------------------

class LabelOneHotMismatchError(DigitalPathologyBufferError):
    """Raise when one-hot configuration of the source buffer does not match the target buffer."""

    def __init__(self, target_one_hot, source_one_hot):
        """
        Initialize the object.

        Args:
            target_one_hot (type): One-hot configuration of the target buffer.
            source_one_hot (type): One-hot configuration of the source buffer.
        """

        # Initialize base class.
        #
        super().__init__('The transferred source \'{src}\' one-hot configuration does not match the \'{dst}\' target.'.format(src=source_one_hot, dst=target_one_hot))

        # Store custom data.
        #
        self.target_one_hot = target_one_hot
        self.source_one_hot = source_one_hot

#----------------------------------------------------------------------------------------------------

class WeightMapMismatchError(DigitalPathologyBufferError):
    """Raise when weight map configuration of the source buffer does not match the target buffer."""

    def __init__(self, target_weight_map, source_weight_map):
        """
        Initialize the object.

        Args:
            target_weight_map (type): Weight map configuration of the target buffer.
            source_weight_map (type): Weight map configuration of the source buffer.
        """

        # Initialize base class.
        #
        super().__init__('The transferred source \'{src}\' weight map configuration does not match the \'{dst}\' target.'.format(src=source_weight_map, dst=target_weight_map))

        # Store custom data.
        #
        self.target_weight_map = target_weight_map
        self.source_weight_map = source_weight_map
