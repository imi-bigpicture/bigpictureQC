"""
Base for the PatchSampler classes.
"""

class BasePatchSampler(object):
    def __init__(self):
        pass

    def sample(self, counts, shapes):
        raise ImportError("Function from base-class not implemented.")

    def close(self):
        raise ImportError("Function from base-class not implemented.")

