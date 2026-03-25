from digitalpathology.adapters.augmenters.noise import noiseaugmenterbase as dptnoiseaugmenterbase
from digitalpathology.errors import augmentationerrors as dptaugmentationerrors
import numpy as np

#----------------------------------------------------------------------------------------------------

class BlackoutAugmenter(dptnoiseaugmenterbase.NoiseAugmenterBase):
    """Apply Gaussian blur on the patch."""

    def __init__(self, blackout_class):
        """
        Initialize the object.
        Args:
            blackout_class (list): values in the mask to blackout.
            value_range (tuple): Minimum and maximum intensity for the input patch. For example (0, 255).
        Raises:
            InvalidBlurSigmaIntervalError: The sigma interval for Gaussian blur is not valid.
        """

        # Initialize base class.
        #
        super().__init__(keyword='Blackout')

        # Initialize members.
        #
        self.__blackout_class = None
        self.__blackout_signatures = []
        self.__setparams(blackout_class)
        self.__signature_choice = 0

    def __setparams(self, blackout_class):
        """
        Set the sigma interval.
        Args:
            blackout_class (): Interval for sigma selection for Gaussian blur.
        Raises:
            InvalidBlurSigmaIntervalError: The sigma interval for Gaussian blur is not valid.
        """

        # Check the interval.
        #

        self.__blackout_class = blackout_class

    def transform(self, patch, label=None):
        """
        Zero out the listed classes in the patch.
        Args:
            patch (np.ndarray): Patch to transform.
            label (np.ndarray):
        Returns:
            np.ndarray: Transformed patch.
        """
        blackout_label = 2
        if np.any(label == self.__blackout_class) and np.count_nonzero(label == blackout_label) > 100:
            signature = np.zeros(label.shape)
            signature[label == blackout_label] = 1
            if len(self.__blackout_signatures) < 10:
                self.__blackout_signatures.extend(signature)
            else:
                self.__blackout_signatures[np.random.randint(0,10)] = signature
        if len(self.__blackout_signatures) > 0 and not np.any(label == self.__blackout_class) and np.random.rand() > 0.1:
            label[self.__blackout_signatures[self.__map_choice] == 1] = 0
            for channel in range(patch.shape[2]):
                patch[:,:,channel][self.__blackout_signatures[self.__map_choice] == 1] = 0

        for channel in range(patch.shape[2]):
            patch[:,:,channel][label == blackout_label] = 0
        label[label == blackout_label] = 0

        return patch, label

    def randomize(self):
        """Randomize the parameters of the augmenter."""
        if len(self.__blackout_signatures) > 1:
            self.__map_choice = np.random.randint(0, len(self.__blackout_signatures) - 1)