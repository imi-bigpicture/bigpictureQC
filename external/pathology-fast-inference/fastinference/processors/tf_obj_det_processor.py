import tensorflow as tf 
import numpy as np 
import cv2 

from ..async_tile_processor import async_tile_processor

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


class tf_obj_det_processor(async_tile_processor): 

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


    def _load_network_model(self):
        network = tf.saved_model.load(self._model_path)
        return network

    def _get_result_patch(self, detections, img_shape):

        min_score=0.6
        thickness = -1 
        
        bboxes = detections['detection_boxes'][0].numpy()
        classes = detections['detection_classes'][0].numpy().astype(np.uint32) + 1 
        scores = detections['detection_scores'][0].numpy()
        soft_scores = detections['detection_multiclass_scores'][0].numpy()
        
        result_patch = np.ones((img_shape + (3,)), dtype=np.float32)

        for indx in range(scores.shape[0]):
            if scores[indx] >= min_score: 
                bbox = (bboxes[indx] * img_shape[0]).astype(int)
                center_y = (bbox[0] + bbox[2]) // 2
                center_x = (bbox[1] + bbox[3]) // 2
                result_patch[center_y - 12:center_y + 12,center_x - 12:center_x + 12] = soft_scores[indx]
                # result_patch = cv2.circle(result_patch, (center_x, center_y), 3, soft_scores[indx], -1)

        return result_patch[None]

    def _predict_tile_batch(self, tile_batch=None, info=None):
        assert tile_batch.shape[0] == 1 
        
        if self._ax_order == 'cwh':
            tile_batch = tile_batch.transpose(0, 3, 1, 2)
        detections = self._model(tile_batch)

        result = self._get_result_patch(detections, tile_batch.shape[1:3])

        if self._ax_order == 'cwh':
            result = result.transpose(0, 2, 3, 1)
        return result

    def _send_reconstruction_info(self):
        self._write_queues[0].put(('recon_info',
                                   '',1))
