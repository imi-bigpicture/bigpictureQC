import tensorflow as tf
import numpy as np
from functools import partial
import keras
from diagmodels.errors import modelerrors as dmsmodelerrors


class LossFunctions():
    """Loss Functions for keras and tensorflow models"""

    def __init__(self, classes=0, loss='categorical_crossentropy', weights=None, temperature_1=0.2, temperature_2=2.0):
        """
        Initialize loss parameters.
        Args:
            classes (int): Number of classes present.
            loss (str): loss function to use
            weights (tensor): input tensor for weights)
            bitempered_t1 (float): value for t1 bitempered
            bitempered_t2 (float): value for t2 bitempered

        return loss function
        """

        # Initialize members.
        #
        self.__classes = classes
        self.__loss = loss
        self.__input_weights = weights
        self.__temperature_1 = temperature_1
        self.__temperature_2 = temperature_2

    def getLossFunction(self):

        if self.__loss == 'lovasz':
            loss_function = partial(self.__keras_lovasz_softmax, weights=self.__input_weights)
            loss_function.__name__ = 'lovasz_loss'

        elif self.__loss == 'focal':
            loss_function = partial(self.__focal_loss, weights=self.__input_weights)
            loss_function.__name__ = 'focal_loss'

        elif self.__loss == 'focal_lovasz':
            loss_function = partial(self.__focal_lovasz, weights=self.__input_weights)
            loss_function.__name__ = 'focal_lovasz_loss'

        elif self.__loss == 'lovasz_cc':
            loss_function = partial(self.__lovasz_cc_loss, weights=self.__input_weights)
            loss_function.__name__ = 'lovasz_cc_loss'

        elif self.__loss == 'tempered':
            loss_function = partial(self.__tempered_loss, weights=self.__input_weights)
            loss_function.__name__ = 'tempered_loss'

        else:
            raise dmsmodelerrors.NotSupportedLoss(self.__loss)

        return loss_function

    # ----------------------------------------------------------------------------------------------------
    # Lovasz loss
    # ----------------------------------------------------------------------------------------------------
    def __cumsum(self, x, axis=None):
        return np.cumsum(x, axis=axis)

    def __lovasz_grad(self, gt_sorted):
        """
        Computes gradient of the Lovasz extension w.r.t sorted errors
        See Alg. 1 in paper
        """
        gts = tf.reduce_sum(gt_sorted)

        intersection = gts - tf.py_function(self.__cumsum, [gt_sorted], tf.float32)
        union = gts + tf.py_function(self.__cumsum, [1. - gt_sorted], tf.float32)
        jaccard = 1. - intersection / union
        jaccard = tf.concat((jaccard[0:1], jaccard[1:] - jaccard[:-1]), 0)

        return jaccard

    def __lovasz_softmax(self, y_true, y_pred, weights):
        """
        Multi-class Lovasz-Softmax loss
        probas: [B, H, W, C] Tensor, class probabilities at each prediction (between 0 and 1)
        labels: [B, H, W] Tensor, ground truth labels (between 0 and C - 1)
        weights: [B, H, W]

        returns lovasz loss
        """

        # Reshape the inputs
        #
        probas = tf.reshape(y_pred, (-1, self.__classes))  # (B*H*W, classes)
        labels = tf.reshape(y_true, (-1,))  # (B*H*W,)
        weights = tf.reshape(weights, (-1,))

        loss = self.__lovasz_softmax_flat(probas, labels, weights, classes='present')

        return loss

    def __lovasz_softmax_flat(self, probas, labels, weights, classes='all'):
        """
        Multi-class Lovasz-Softmax loss
          probas: [P, C] Variable, class probabilities at each prediction (between 0 and 1)
          labels: [P] Tensor, ground truth labels (between 0 and C - 1)
          weights: [P] Tensor, weight map for pixel wise weighting.
          classes: 'all' for all, 'present' for classes present in labels, or a list of classes to average.
        """
        C = self.__classes  # (classes)
        losses = []
        present = []

        # TODO: Check if we ever want to have the option to calculate the loss per patch.
        #
        class_to_sum = list(range(C)) if classes in ['all', 'present'] else classes  # [0,1,2,3 etc] -> len == classes
        for c in class_to_sum:
            fg = tf.cast(tf.equal(labels, c), probas.dtype)  # foreground for class c (B*H*W,)
            if classes == 'present':
                present.append(tf.reduce_sum(fg) > 0)
            errors = tf.abs(fg - probas[:, c])  # (B*H*W,)
            errors_sorted, perm = tf.nn.top_k(errors, k=tf.shape(errors)[0], name="descending_sort_{}".format(c))  # (B*H*W,) (decending)
            fg_sorted = tf.gather(fg, perm)

            weights_sorted = tf.gather(weights, perm)
            grad = self.__lovasz_grad(fg_sorted)

            weighted_grads = tf.stop_gradient(tf.multiply(grad, weights_sorted))

            class_loss = tf.tensordot(errors_sorted, weighted_grads, 1, name="loss_class_{}".format(c))
            losses.append(class_loss)

        if len(class_to_sum) == 1:  # short-circuit mean when only one class
            return losses[0]
        losses_tensor = tf.stack(losses)
        if classes == 'present':
            present = tf.stack(present)
            losses_tensor = tf.boolean_mask(losses_tensor, present)
        loss = tf.reduce_mean(losses_tensor)
        return loss

    def __keras_lovasz_softmax(self, y_true, y_pred, weights):
        ''''
        Multi class Lovasz loss to minimize.

        Code taken from https://github.com/bermanmaxim/LovaszSoftmax.
        Adapted to work with Keras and our library
        '''

        y_arg = tf.keras.backend.expand_dims(tf.argmax(y_true, axis=-1), axis=-1)

        loss = self.__lovasz_softmax(y_arg, y_pred, weights)

        return loss

    # ----------------------------------------------------------------------------------------------------
    # Lovasz loss + Categorical crossentropy
    # ----------------------------------------------------------------------------------------------------

    def __lovasz_cc_loss(self, y_true, y_pred, weights):
        '''
        Combination of lovasz loss and categorical crossentropy
        '''

        y_arg = tf.keras.backend.expand_dims(tf.argmax(y_true, axis=-1), axis=-1)
        lovasz_loss = self.__lovasz_softmax(y_arg, y_pred, weights)

        cross_entr_loss = tf.keras.backend.expand_dims(keras.losses.categorical_crossentropy(y_true, y_pred), axis=-1)
        cross_entr_loss = tf.multiply(cross_entr_loss, weights)
        cross_entr_loss = tf.reduce_mean(cross_entr_loss)

        return lovasz_loss + cross_entr_loss

    # ----------------------------------------------------------------------------------------------------
    # Focal loss
    # ----------------------------------------------------------------------------------------------------

    def __focal_loss(self, y_true, y_pred, weights):
        """
        implementation from: https://github.com/umbertogriffo/focal-loss-keras/blob/master/losses.py

        Softmax version of focal loss.
               m
          FL = ∑  -alpha * (1 - p_o,c)^gamma * y_o,c * log(p_o,c)
              c=1
          where m = number of classes, c = class and o = observation

          gamma -- 2.0 as mentioned in the paper
          alpha -- 0.25 as mentioned in the paper

        References:
            Official paper: https://arxiv.org/pdf/1708.02002.pdf
            https://www.tensorflow.org/api_docs/python/tf/keras/backend/categorical_crossentropy

        Arguments:
            y_true {tensor} -- ground truth labels, shape of [B, H*W, num_cls]
            y_pred {tensor} -- model's output, shape of [B, H*W, num_cls]
            weights {tensor} -- weights [B, H*W, 1]

        Returns:
            [tensor] -- loss.
        """

        alpha = self.__temperature_1  # 0.25
        gamma = self.__temperature_2  # 2.0

        y_pred = tf.add(y_pred, tf.keras.backend.epsilon())

        categorical_entropy = tf.keras.backend.expand_dims(keras.losses.categorical_crossentropy(y_true, y_pred), axis=-1)
        focal_weight = tf.multiply(y_true, tf.pow(tf.subtract(1., y_pred), gamma))

        focal_loss = tf.multiply(alpha, tf.multiply(focal_weight, categorical_entropy))

        focal_loss_weighted = tf.multiply(focal_loss, weights)
        reduced_focal_loss_weighted = tf.reduce_max(focal_loss_weighted, axis=-1)
        reduced_focal_loss_weighted = tf.reduce_mean(reduced_focal_loss_weighted)

        return reduced_focal_loss_weighted

    # ----------------------------------------------------------------------------------------------------
    # Lovasz loss + Focal loss
    # ----------------------------------------------------------------------------------------------------

    def __focal_lovasz(self, y_true, y_pred, weights):
        """
        Combination of focal loss and lovasz loss
        """
        focal_loss = self.__focal_loss(y_true, y_pred, weights)
        lovasz_loss = self.__keras_lovasz_softmax(y_true, y_pred, weights)

        return focal_loss + lovasz_loss

    # ----------------------------------------------------------------------------------------------------
    # bi-tempered loss
    # ----------------------------------------------------------------------------------------------------

    def __tempered_loss(self, y_true, y_pred, weights):
        """
        Implementation of the bi-tempered loss:
        Blogpost: https://ai.googleblog.com/2019/08/bi-tempered-logistic-loss-for-training.html
        Paper: https://arxiv.org/pdf/1906.03361.pdf

        y_true: [B, HxW, C] Tensor, class probabilities at each prediction (between 0 and 1)
        y_pred: [B, HxW, C] Tensor, ground truth labels (between 0 and C - 1)
        weights: [B, HxW, 1]

        returns bi-tempered loss
        """

        t1 = self.__temperature_1  # 0.2
        t2 = self.__temperature_2  # 4.0

        loss = self.__bi_tempered_logistic_loss(y_pred, y_true, weights, t1, t2)

        return loss

    def __log_t(self, u, t):
        """Compute log_t for `u`."""

        def _internal_log_t(u, t):
            return (u ** (1.0 - t) - 1.0) / (1.0 - t)

        return tf.cond(
            tf.equal(t, 1.0), lambda: tf.math.log(u),
            partial(_internal_log_t, u, t))

    def __exp_t(self, u, t):
        """Compute exp_t for `u`."""

        def _internal_exp_t(u, t):
            return tf.nn.relu(1.0 + (1.0 - t) * u) ** (1.0 / (1.0 - t))

        return tf.cond(
            tf.equal(t, 1.0), lambda: tf.exp(u),
            partial(_internal_exp_t, u, t))

    def __compute_normalization_fixed_point(self, activations, t, num_iters=5):
        """Returns the normalization value for each example (t > 1.0).
        Args:
          activations: A multi-dimensional tensor with last dimension `num_classes`.
          t: Temperature 2 (> 1.0 for tail heaviness).
          num_iters: Number of iterations to run the method.
        Return: A tensor of same rank as activation with the last dimension being 1.
        """

        mu = tf.reduce_max(activations, -1, keepdims=True)
        normalized_activations_step_0 = activations - mu
        shape_normalized_activations = tf.shape(normalized_activations_step_0)

        def iter_condition(i, unused_normalized_activations):
            return i < num_iters

        def iter_body(i, normalized_activations):
            logt_partition = tf.math.reduce_sum(
                self.__exp_t(normalized_activations, t), -1, keepdims=True)
            normalized_activations_t = tf.reshape(
                normalized_activations_step_0 * tf.pow(logt_partition, 1.0 - t),
                shape_normalized_activations)
            return [i + 1, normalized_activations_t]

        _, normalized_activations_t = tf.while_loop(
            iter_condition,
            iter_body, [0, normalized_activations_step_0],
            maximum_iterations=num_iters)
        logt_partition = tf.math.reduce_sum(
            self.__exp_t(normalized_activations_t, t), -1, keepdims=True)
        return -self.__log_t(1.0 / logt_partition, t) + mu

    def __compute_normalization_binary_search(self, activations, t, num_iters=10):
        """Returns the normalization value for each example (t < 1.0).
        Args:
          activations: A multi-dimensional tensor with last dimension `num_classes`.
          t: Temperature 2 (< 1.0 for finite support).
          num_iters: Number of iterations to run the method.
        Return: A tensor of same rank as activation with the last dimension being 1.
        """
        mu = tf.math.reduce_max(activations, -1, keepdims=True)
        normalized_activations = activations - mu
        shape_activations = tf.shape(activations)
        effective_dim = tf.cast(
            tf.reduce_sum(
                tf.cast(
                    tf.greater(normalized_activations, -1.0 / (1.0 - t)), tf.int32),
                -1,
                keepdims=True), tf.float32)
        shape_partition = tf.concat([shape_activations[:-1], [1]], 0)
        lower = tf.zeros(shape_partition)
        upper = -self.__log_t(1.0 / effective_dim, t) * tf.ones(shape_partition)

        def iter_condition(i, unused_lower, unused_upper):
            return i < num_iters

        def iter_body(i, lower, upper):
            logt_partition = (upper + lower) / 2.0
            sum_probs = tf.reduce_sum(self.__exp_t(
                normalized_activations - logt_partition, t), -1, keepdims=True)
            update = tf.cast(tf.less(sum_probs, 1.0), tf.float32)
            lower = tf.reshape(lower * update + (1.0 - update) * logt_partition,
                               shape_partition)
            upper = tf.reshape(upper * (1.0 - update) + update * logt_partition,
                               shape_partition)
            return [i + 1, lower, upper]

        _, lower, upper = tf.while_loop(
            iter_condition,
            iter_body, [0, lower, upper],
            maximum_iterations=num_iters)
        logt_partition = (upper + lower) / 2.0
        return logt_partition + mu

    def __compute_normalization(self, activations, t, num_iters=5):
        """Returns the normalization value for each example.
        Args:
          activations: A multi-dimensional tensor with last dimension `num_classes`.
          t: Temperature 2 (< 1.0 for finite support, > 1.0 for tail heaviness).
          num_iters: Number of iterations to run the method.
        Return: A tensor of same rank as activation with the last dimension being 1.
        """
        return tf.cond(
            tf.less(t, 1.0),
            partial(self.__compute_normalization_binary_search, activations, t,
                    num_iters),
            partial(self.__compute_normalization_fixed_point, activations, t,
                    num_iters))

    def _internal_bi_tempered_logistic_loss(self, activations, labels, t1, t2):
        """Computes the Bi-Tempered logistic loss.
        Args:
          activations: A multi-dimensional tensor with last dimension `num_classes`.
          labels: batch_size
          t1: Temperature 1 (< 1.0 for boundedness).
          t2: Temperature 2 (> 1.0 for tail heaviness).
        Returns:
          A loss tensor for robust loss.
        """
        if t2 == 1.0:
            normalization_constants = tf.math.log(
                tf.reduce_sum(tf.exp(activations), -1, keepdims=True))
            if t1 == 1.0:
                return normalization_constants + tf.reduce_sum(
                    tf.multiply(labels, tf.math.log(labels + 1e-10) - activations), -1,
                    keepdims=True)
            else:
                shifted_activations = tf.exp(activations - normalization_constants)
                one_minus_t2 = 1.0
        else:
            one_minus_t1 = (1.0 - t1)
            one_minus_t2 = (1.0 - t2)
            normalization_constants = self.__compute_normalization(
                activations, t2, num_iters=5)
            shifted_activations = tf.nn.relu(1.0 + one_minus_t2 *
                                             (activations - normalization_constants))

        if t1 == 1.0:
            return tf.reduce_sum(
                tf.multiply(
                    tf.math.log(labels + 1e-10) -
                    tf.math.log(tf.pow(shifted_activations, 1.0 / one_minus_t2)), labels),
                -1,
                keepdims=True)
        else:
            beta = 1.0 + one_minus_t1
            logt_probs = (tf.pow(shifted_activations, one_minus_t1 / one_minus_t2) -
                          1.0) / one_minus_t1
            return tf.reduce_sum(
                tf.multiply(self.__log_t(labels, t1) - logt_probs, labels) - 1.0 / beta *
                (tf.pow(labels, beta) -
                 tf.pow(shifted_activations, beta / one_minus_t2)), -1)

    def __tempered_sigmoid(self, activations, t, num_iters=5):
        """Tempered sigmoid function.
        Args:
          activations: Activations for the positive class for binary classification.
          t: Temperature tensor > 0.0.
          num_iters: Number of iterations to run the method.
        Returns:
          A probabilities tensor.
        """
        t = tf.convert_to_tensor(t)
        input_shape = tf.shape(activations)
        activations_2d = tf.reshape(activations, [-1, 1])
        internal_activations = tf.concat(
            [tf.zeros_like(activations_2d), activations_2d], 1)
        normalization_constants = tf.cond(
            # pylint: disable=g-long-lambda
            tf.equal(t, 1.0),
            lambda: tf.math.log(
                tf.reduce_sum(tf.exp(internal_activations), -1, keepdims=True)),
            partial(self.__compute_normalization, internal_activations, t,
                    num_iters))
        internal_probabilities = self.__exp_t(internal_activations - normalization_constants,
                                              t)
        one_class_probabilities = tf.split(internal_probabilities, 2, axis=1)[1]
        return tf.reshape(one_class_probabilities, input_shape)

    def __tempered_softmax(self, activations, t, num_iters=5):
        """Tempered softmax function.
        Args:
          activations: A multi-dimensional tensor with last dimension `num_classes`.
          t: Temperature tensor > 0.0.
          num_iters: Number of iterations to run the method.
        Returns:
          A probabilities tensor.
        """
        t = tf.convert_to_tensor(t)
        normalization_constants = tf.cond(
            tf.equal(t, 1.0),
            lambda: tf.math.log(tf.reduce_sum(tf.exp(activations), -1, keepdims=True)),
            partial(self.__compute_normalization, activations, t, num_iters))
        return self.__exp_t(activations - normalization_constants, t)

    def __bi_tempered_binary_logistic_loss(self,
                                           activations,
                                           labels,
                                           t1,
                                           t2,
                                           label_smoothing=0.0,
                                           num_iters=5):
        """Bi-Tempered binary logistic loss.
        Args:
          activations: A tensor containing activations for class 1.
          labels: A tensor with shape and dtype as activations.
          t1: Temperature 1 (< 1.0 for boundedness).
          t2: Temperature 2 (> 1.0 for tail heaviness, < 1.0 for finite support).
          label_smoothing: Label smoothing
          num_iters: Number of iterations to run the method.
        Returns:
          A loss tensor.
        """
        with tf.name_scope('binary_bitempered_logistic'):
            t1 = tf.convert_to_tensor(t1)
            t2 = tf.convert_to_tensor(t2)
            out_shape = tf.shape(labels)
            labels_2d = tf.reshape(labels, [-1, 1])
            activations_2d = tf.reshape(activations, [-1, 1])
            internal_labels = tf.concat([1.0 - labels_2d, labels_2d], 1)
            internal_logits = tf.concat([tf.zeros_like(activations_2d), activations_2d],
                                        1)
            losses = self.__bi_tempered_logistic_loss(internal_logits, internal_labels, t1, t2,
                                                      label_smoothing, num_iters)
            return tf.reshape(losses, out_shape)

    def __bi_tempered_logistic_loss(self,
                                    activations,
                                    labels,
                                    weights,
                                    t1,
                                    t2,
                                    label_smoothing=0.0,
                                    num_iters=5):
        """Bi-Tempered Logistic Loss with custom gradient.
        Args:
          activations: A multi-dimensional tensor with last dimension `num_classes`.
          labels: A tensor with shape and dtype as activations.
          t1: Temperature 1 (< 1.0 for boundedness).
          t2: Temperature 2 (> 1.0 for tail heaviness, < 1.0 for finite support).
          label_smoothing: Label smoothing parameter between [0, 1).
          num_iters: Number of iterations to run the method.
        Returns:
          A loss tensor.
        """
        with tf.name_scope('bitempered_logistic'):
            t1 = tf.convert_to_tensor(t1)
            t2 = tf.convert_to_tensor(t2)
            if label_smoothing > 0.0:
                num_classes = tf.cast(tf.shape(labels)[-1], tf.float32)
                labels = (
                                 1 - num_classes /
                                 (num_classes - 1) * label_smoothing) * labels + label_smoothing / (
                                 num_classes - 1)

            @tf.custom_gradient
            def _custom_gradient_bi_tempered_logistic_loss(activations):
                """Bi-Tempered Logistic Loss with custom gradient.
                Args:
                  activations: A multi-dimensional tensor with last dim `num_classes`.
                Returns:
                  A loss tensor, grad.
                """
                with tf.name_scope('gradient_bitempered_logistic'):
                    probabilities = self.__tempered_softmax(activations, t2, num_iters)
                    loss_values = tf.multiply(
                        labels,
                        self.__log_t(labels + 1e-10, t1) -
                        self.__log_t(probabilities, t1)) - 1.0 / (2.0 - t1) * (
                                          tf.pow(labels, 2.0 - t1) - tf.pow(probabilities, 2.0 - t1))

                    def grad(d_loss):
                        """Explicit gradient calculation.
                        Args:
                          d_loss: Infinitesimal change in the loss value.
                        Returns: Loss gradient.
                        """
                        delta_probs = probabilities - labels
                        forget_factor = tf.pow(probabilities, t2 - t1)
                        delta_probs_times_forget_factor = tf.multiply(delta_probs,
                                                                      forget_factor)
                        delta_forget_sum = tf.math.reduce_sum(
                            delta_probs_times_forget_factor, -1, keepdims=True)
                        escorts = tf.pow(probabilities, t2)
                        escorts = escorts / tf.math.reduce_sum(escorts, -1, keepdims=True)
                        derivative = delta_probs_times_forget_factor - tf.multiply(
                            escorts, delta_forget_sum)
                        return tf.multiply(d_loss, derivative)

                    return loss_values, grad

            loss_values = _custom_gradient_bi_tempered_logistic_loss(activations)
            loss_values_weighted = tf.multiply(loss_values, weights)
            return tf.math.reduce_sum(loss_values_weighted, -1)

    def __sparse_bi_tempered_logistic_loss(self,
                                           activations,
                                           labels,
                                           t1,
                                           t2,
                                           num_iters=5):
        """Sparse Bi-Tempered Logistic Loss with custom gradient.
        Args:
          activations: A multi-dimensional tensor with last dimension `num_classes`.
          labels: A tensor with dtype of int32.
          t1: Temperature 1 (< 1.0 for boundedness).
          t2: Temperature 2 (> 1.0 for tail heaviness, < 1.0 for finite support).
          num_iters: Number of iterations to run the method.
        Returns:
          A loss tensor.
        """
        with tf.name_scope('sparse_bitempered_logistic'):
            t1 = tf.convert_to_tensor(t1)
            t2 = tf.convert_to_tensor(t2)
            num_classes = tf.shape(activations)[-1]

            @tf.custom_gradient
            def _custom_gradient_sparse_bi_tempered_logistic_loss(activations):
                """Sparse Bi-Tempered Logistic Loss with custom gradient.
                Args:
                  activations: A multi-dimensional tensor with last dim `num_classes`.
                Returns:
                  A loss tensor, grad.
                """
                with tf.name_scope('gradient_sparse_bitempered_logistic'):
                    probabilities = self.__tempered_softmax(activations, t2, num_iters)
                    loss_values = -self.__log_t(
                        tf.reshape(
                            tf.gather_nd(probabilities,
                                         tf.where(tf.one_hot(labels, num_classes))),
                            tf.shape(activations)[:-1]), t1) - 1.0 / (2.0 - t1) * (
                                          1.0 - tf.reduce_sum(tf.pow(probabilities, 2.0 - t1), -1))

                    def grad(d_loss):
                        """Explicit gradient calculation.
                        Args:
                          d_loss: Infinitesimal change in the loss value.
                        Returns: Loss gradient.
                        """
                        delta_probs = probabilities - tf.one_hot(labels, num_classes)
                        forget_factor = tf.pow(probabilities, t2 - t1)
                        delta_probs_times_forget_factor = tf.multiply(delta_probs,
                                                                      forget_factor)
                        delta_forget_sum = tf.math.reduce_sum(
                            delta_probs_times_forget_factor, -1, keepdims=True)
                        escorts = tf.pow(probabilities, t2)
                        escorts = escorts / tf.math.reduce_sum(escorts, -1, keepdims=True)
                        derivative = delta_probs_times_forget_factor - tf.multiply(
                            escorts, delta_forget_sum)
                        return tf.multiply(d_loss, derivative)

                    return loss_values, grad

            loss_values = _custom_gradient_sparse_bi_tempered_logistic_loss(activations)
            return loss_values
