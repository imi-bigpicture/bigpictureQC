import tensorflow as tf

def iou_coef(y_true, y_pred):
    '''
    IOU coefficient metric

    y_true (tensor): tensor containing all true labels (B, H*W, C)
    y_pred (): tensor containing predictions (B, H*W, C)
    '''

    intersection = tf.multiply(y_true, y_pred)
    notTrue = tf.subtract(1., y_true)

    union = y_true + tf.multiply(notTrue, y_pred)

    iou = (tf.keras.backend.sum(intersection, axis=-1) + tf.keras.backend.epsilon()) / (tf.keras.backend.sum(union, axis=-1) + tf.keras.backend.epsilon())

    return iou


def dice_coef(y_true, y_pred):
    '''
    Dice coefficient metric

    y_true (tensor): tensor containing all true labels (B, H*W, C)
    y_pred (): tensor containing predictions (B, H*W, C)
    '''

    smooth = 1.

    intersection = tf.keras.backend.sum(y_true * y_pred)
    score = (2. * intersection + smooth) / (tf.keras.backend.sum(y_true) + tf.keras.backend.sum(y_pred) + smooth)

    return score
