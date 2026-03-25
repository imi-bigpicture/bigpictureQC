import digitalpathology.generator.batch.simplesampler as sampler
import digitalpathology.generator.tf_data_generator as data_generator

import argparse
import os
import sys
import yaml
import tensorflow as tf
import segmentation_models as sm
import albumentations

from albumentations.core.transforms_interface import ImageOnlyTransform
import skimage.color
import numpy as np

from segmentation_models.base import Loss 
from segmentation_models.base import functional as F 

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)

class LovaszLoss(Loss):
    def __init__(self, num_classes=2, class_weights=None, class_indexes=None):
        super().__init__(name="lovasz_loss")
        self.class_weights = class_weights if class_weights is not None else 1
        self.class_indexes = class_indexes
        self.num_classes = num_classes
        
    def __call__(self, gt, pr):
        
        return self._lovasz_softmax_flat(probas=pr,
                                         labels=gt,
                                         classes='present',
                                         num_classes=self.num_classes)

    def __cumsum(self, x, axis=None):
        return np.cumsum(x, axis=axis)

    def _lovasz_grad(self, gt_sorted):
        """
        Computes gradient of the Lovasz extension w.r.t sorted errors
        See Alg. 1 in paper
        """
        gts = tf.reduce_sum(gt_sorted)

        intersection = gts - tf.py_function(self.__cumsum, [gt_sorted], tf.float32)
        union = gts + tf.py_function(self.__cumsum, [1. - gt_sorted], tf.float32)
        jaccard = 1. - intersection / (union + 1e-5)
        jaccard = tf.concat((jaccard[0:1], jaccard[1:] - jaccard[:-1]), 0)

        return jaccard
    
    @tf.autograph.experimental.do_not_convert
    def _lovasz_softmax_flat(self, probas, labels, classes='all', num_classes=2):
        """
        Multi-class Lovasz-Softmax loss
          probas: [P, C] Variable, class probabilities at each prediction (between 0 and 1)
          labels: [P] Tensor, ground truth labels (between 0 and C - 1)
          weights: [P] Tensor, weight map for pixel wise weighting.
          classes: 'all' for all, 'present' for classes present in labels, or a list of classes to average.
        """
        C = num_classes  # (classes)
        probas = tf.reshape(probas, (-1, C))  # (B*H*W, classes)
        labels = tf.keras.backend.expand_dims(tf.argmax(labels, axis=-1), axis=-1)
        labels = tf.reshape(labels, (-1,)) # (B*H*W,)
        
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
            errors_sorted, perm = tf.math.top_k(errors, k=tf.shape(errors)[0], name="descending_sort_{}".format(c))  # (B*H*W,) (decending)
            fg_sorted = tf.gather(fg, perm)

            grad = self._lovasz_grad(fg_sorted)
            
            class_loss = tf.tensordot(errors_sorted, grad, 1, name="loss_class_{}".format(c))
            losses.append(class_loss)

        if len(class_to_sum) == 1:  # short-circuit mean when only one class
            return losses[0]
        losses_tensor = tf.stack(losses)
        if classes == 'present':
            present = tf.stack(present)
            losses_tensor = tf.boolean_mask(losses_tensor, present)
        loss = tf.reduce_mean(losses_tensor)
        return loss        

class HED_normalize(ImageOnlyTransform):
    def __init__(self, sigmas, biases, **params):
        super(HED_normalize, self).__init__(**params)
        self.biases = biases
        self.sigmas = sigmas

    def color_norm_hed(self, img, sigmas, biases):
        patch = img.astype(np.float32) / 255.0
        patch_hed = skimage.color.rgb2hed(rgb=patch)

        # Augment the Haematoxylin channel.
        patch_hed[:, :, 0] *= 1.0 + sigmas[0]
        patch_hed[:, :, 0] += biases[0]

        # Augment the Eosin channel.
        patch_hed[:, :, 1] *= 1.0 + sigmas[1]
        patch_hed[:, :, 1] += biases[1]

        patch_hed[:, :, 2] *= 1.0 + sigmas[2]
        patch_hed[:, :, 2] += biases[2]

        # Convert back to RGB color coding.
        patch_rgb = skimage.color.hed2rgb(hed=patch_hed)
        patch_transformed = np.clip(a=patch_rgb, a_min=0.0, a_max=1.0)
        return np.asarray(patch_transformed * 255, dtype=np.uint8)

    def apply(self, img, sigmas, biases, **params):
        return self.color_norm_hed(img, sigmas, biases)

    def get_params_dependent_on_targets(self, params):
        sigmas = [np.random.uniform(-self.sigmas[i], self.sigmas[i]) for i in range(len(self.sigmas))]
        biases = [np.random.uniform(-self.biases[i], self.biases[i]) for i in range(len(self.biases))]
        return {'sigmas': sigmas, 'biases': biases}

    def get_transform_init_args_names(self):
        return ("sigmas", "biases")

    @property
    def targets_as_params(self):
        return ["image"]

class InferenceModelSave(tf.keras.callbacks.Callback):
    def __init__(self, output_path=None):
        super().__init__()
        self._filepath = output_path

    def on_epoch_end(self, epoch, logs=None):
        print("saving inference model")
        filepath = self._filepath
        inference_model = tf.keras.models.Model(self.model.inputs, self.model.layers[-2].output)
        inference_model.save(filepath, overwrite=True)

def collect_arguments():
    """
    Collect command line arguments.
    """
    argument_parser = argparse.ArgumentParser(description='Check dataset configuration.')
    argument_parser.add_argument('-r', '--run_name', required=False, type=str, default="model",help='input data file')
    argument_parser.add_argument('-d', '--data_path', required=True, type=str, help='input data file')
    argument_parser.add_argument('-c', '--config_path', required=True, type=str, help='input')
    argument_parser.add_argument('-a', '--alb_config_path', required=False, type=str, default="", help='albumentations')
    argument_parser.add_argument('-o', '--output_path', required=True, type=str, help='output')
    arguments = vars(argument_parser.parse_args())

    return arguments["run_name"], arguments["data_path"], arguments["config_path"], \
           arguments["alb_config_path"], arguments["output_path"]

#----------------------------------------------------------------------------------------------------

class WeightedDice(tf.keras.metrics.Mean):
    def __init__(self, name="dice_coe", **kwargs):
        super(WeightedDice, self).__init__(name=name, **kwargs)
        self.smooth = 1e-10

    def update_state(self, y_true, y_pred, sample_weight=None):
        output = tf.dtypes.cast(tf.math.argmax(y_pred, axis=-1), tf.float32)
        target = tf.dtypes.cast(tf.math.argmax(y_true, axis=-1), tf.float32)
        inse = tf.reduce_sum(tf.cast((output == target), tf.float32) * sample_weight, axis=-1)
        dice = (2. * inse + self.smooth) / (tf.reduce_sum(sample_weight, axis=-1)*2 + self.smooth)
        return super(WeightedDice, self).update_state(dice)

def dice_coe(output, target, axis = None, smooth=1e-10):
    output = tf.dtypes.cast( tf.math.greater(output, 0.5), tf. float32 )
    target = tf.dtypes.cast( tf.math.greater(target, 0.5), tf. float32 )
    inse = tf.reduce_sum(output * target, axis=axis)
    l = tf.reduce_sum(output, axis=axis)
    r = tf.reduce_sum(target, axis=axis)

    dice = (2. * inse + smooth) / (l + r + smooth)
    dice = tf.reduce_mean(dice, name='dice_coe')
    return dice

def get_config_from_yaml(config_path):
    with open(file=config_path, mode='r') as param_file:
        parameters = yaml.load(stream=param_file, Loader=yaml.SafeLoader)
    return parameters['model'], parameters['sampler'], parameters['training']

def init_model(model_param, n_classes):
    if model_param['activation'] == 'sigmoid':
        model = sm.Unet(backbone_name=model_param['backbone'],
                        encoder_weights=model_param['encoder_weights'],
                        activation=model_param['activation'])
        loss = tf.keras.losses.BinaryCrossentropy()
        dice = dice_coe
    elif model_param['activation'] == 'softmax':
        model = sm.Unet(backbone_name=model_param['backbone'],
                        encoder_weights=model_param['encoder_weights'],
                        activation=model_param['activation'],
                        classes=n_classes)
        # loss = tf.keras.losses.CategoricalCrossentropy()
        loss = LovaszLoss(num_classes=n_classes) if model_param['loss'] == 'lovasz' else tf.keras.losses.CategoricalCrossentropy()
        flat = tf.keras.layers.Reshape((-1, n_classes), name="output_flat")(model.output)
        model = tf.keras.models.Model(model.inputs, flat)
        dice = WeightedDice()
    else:
        model = sm.Unet(backbone_name=model_param['backbone'],
                        encoder_weights=model_param['encoder_weights'],
                        activation=model_param['activation'],
                        classes=n_classes)
        loss = tf.keras.losses.CategoricalCrossentropy()
        dice = WeightedDice()
    if model_param['activation'] == 'softmax':
        model.compile(tf.keras.optimizers.Adam(model_param['learning_rate']),
                      loss=loss,
                      weighted_metrics=[dice, 'accuracy'])
    else:
        model.compile(tf.keras.optimizers.Adam(model_param['learning_rate']),
                      loss=loss,
                      metrics=[dice, 'accuracy'])
    return model


def main():
    run_name, data_path, config_path, albumentations_path, output_path = collect_arguments()

    model_param, sampler_param, training_param = get_config_from_yaml(config_path)

    if albumentations_path and os.path.exists(albumentations_path):
        print("Transform file found, applying transforms")
        transforms = albumentations.load(albumentations_path, data_format='yaml')
    else:
        print("No transform file found!!!!!")
        transforms = None
    n_classes = len(np.unique(list(sampler_param['training']['label_map'].values())))
    print(n_classes)
    to_categorical = True if model_param['activation'] == 'softmax' else False

    training_sampler = sampler.SimpleSampler(patch_source_filepath=data_path,
                                             **sampler_param['training'])
    val_sampler = sampler.SimpleSampler(patch_source_filepath=data_path,
                                        **sampler_param['validation'],
                                        partition='validation')

    train_datagen = data_generator.TFDataGenerator(training_sampler,
                                                   augmentations_pipeline=transforms,
                                                   batch_size=training_param['training_batch_size'],
                                                   n_classes=n_classes,
                                                   to_categorical=to_categorical)

    val_datagen = data_generator.TFDataGenerator(val_sampler,
                                                 batch_size=training_param['validation_batch_size'],
                                                 reset_samplers=False,
                                                 n_classes=n_classes,
                                                 to_categorical=to_categorical)

    print("{} images in the training sampler".format(len(training_sampler._patch_samplers.items())))
    print("{} images in the validation sampler".format(len(val_sampler._patch_samplers.items())))

    print("creating model for {} classes".format(n_classes))
    model = init_model(model_param, n_classes)

    

    model_output_path = os.path.join(output_path, run_name + ".h5")
    checkpoint = tf.keras.callbacks.ModelCheckpoint(model_output_path, monitor='val_dice_coe', mode='max',
                                                    save_best_only=False)
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_dice_coe', mode='max',
                                                  patience=training_param['stop_plateau'],
                                                  restore_best_weights=False)
    reduce = tf.keras.callbacks.ReduceLROnPlateau(factor=training_param['lr_reduction_factor'],
                                                  patience=training_param['lr_plateau'])

    callbacks = [checkpoint, reduce, early_stop]
    if model_param['activation'] == 'softmax':
        print("add inference model saving callback")
        model_output_inference_path = os.path.join(output_path, run_name + "_inference.h5")
        inference_model = InferenceModelSave(model_output_inference_path)
        callbacks.append(inference_model)

    model.fit(train_datagen,
              epochs=training_param['epochs'],
              callbacks=callbacks,
              validation_data=val_datagen,
              use_multiprocessing=False,
              max_queue_size=training_param['workers']*2,
              workers=training_param['workers'],
			  verbose=2)

if __name__ == '__main__':
    sys.exit(main())
