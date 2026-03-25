"""
Regularizer object for Elastic Weight Consolidation (EWC).
"""

import tensorflow as tf
import numpy as np

#----------------------------------------------------------------------------------------------------

class EwcRegularizer(tf.keras.regularizers.Regularizer):
    """Regularizer object for EWC."""

    def __init__(self, fisher_information, prior_weights, lambda_param=0.1, **kwargs):
        """
        Initialize regularizer parameters.

        Args:
            fisher_information (numpy.ndarray): Fisher information matrix.
            prior_weights (numpy.ndarray): Reference weights.
            lambda_param (float): Regularization ratio.
        """

        # Initialize base class.
        #
        super().__init__(**kwargs)

        # Initialize members.
        #
        self.fisher_information = np.asarray(fisher_information, dtype=np.float32)
        self.prior_weights = np.asarray(prior_weights, dtype=np.float32)
        self.lambda_param = lambda_param

    def __call__(self, x):
        """
        Calculate regularization loss.

        Return:
            float: Regularization loss.
        """

        return self.lambda_param * tf.keras.backend.sum(self.fisher_information * tf.keras.backend.square(x - self.prior_weights))

    def get_config(self):
        """
        Get configuration.

        Returns:
            dict: Configuration to instantiate a new regularizer object.
        """

        return {'fisher_information': self.fisher_information, 'prior_weights': self.prior_weights, 'lambda_param': self.lambda_param}
