"""
This file contains the base-class for extracting patches from a collection of whole slide images.
"""
# ----------------------------------------------------------------------------------------------------

class BaseBatchSampler(object):
    """This class is a batch sampler class that extracts patches from a collection of whole slide images."""

    def __init__(self):
        pass

    @property
    def workers(self):
        raise ImportError("Function from base-class not implemented.")

    def stop(self):
        """Terminate and join all sampler processes."""
        raise ImportError("Function from base-class not implemented.")

    def error(self, message):
        """
        Log an error that occurred outside the BatchSampler to save the reason of imminent shutdown.
        """
        raise ImportError("Function from base-class not implemented.")

    def ping(self):
        """
        Check if all worker processes are alive and responding.
        """
        raise ImportError("Function from base-class not implemented.")

    def step(self):
        """
        Clear the unnecessary patch samplers and create new ones with randomized sources.
        """
        raise ImportError("Function from base-class not implemented.")

    def batch(self, batch_size):
        """
        Collect a batch of patches.
        """
        raise ImportError("Function from base-class not implemented.")