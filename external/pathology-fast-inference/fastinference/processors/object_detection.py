from ..async_tile_processor import async_tile_processor

import torch
from yolov5.utils.torch_utils import select_device
from yolov5.models.experimental import attempt_load
from yolov5.utils.general import non_max_suppression, xyxy2xywh, scale_coords


class object_detection(async_tile_processor):
    """
    Subclassed from async_tile_processor. Uses YOLOv5 to do
    object detection on WSIs.
    """
    def __init__(self, **kwargs):
        async_tile_processor.__init__(self, **kwargs)
        self._gpu_device = select_device(str(kwargs['gpu_device']))
        print("Using custom processor: object_detection")

    def _load_network_model(self):
        model = attempt_load(self._model_path, map_location=self._gpu_device)
        print("Succesfully loaded network weights")
        return model

    @torch.no_grad()
    def _detect_objects(self, tile_batch, info):
        if self._ax_order == 'cwh':
            tile_batch = tile_batch.transpose(0, 3, 1, 2)
        tile_batch = torch.from_numpy(tile_batch).to(self._gpu_device)
        preds = self._model(tile_batch)[0]     # Output shape is [batchsize, n_predictions, 6]
        preds = self._postprocess_predictions(preds, tile_batch.shape[2:])
        return preds

    def _postprocess_predictions(self, preds, im_shape):
        """
        Apply non-maximum suppression of predictions (NMS) and center
        predictions from left upper corner to center of bounding box prediction.

        Returns list of detections, (n,6) torch.Tensors per image [xywh, conf, class]
        """

        # NMS
        preds = non_max_suppression(preds,
                                    conf_thres=0.25,     # TODO: Make this variable
                                    iou_thres=0.45,      # TODO: Make this variable
                                    classes=None,        # Filter specific classes for NMS
                                    agnostic=False,  # If true, apply NMS on all predictions together instead of per class
                                    max_det=1000)        # Maximum number of predictions

        # Center predictions
        if len(preds):
            for i, pred in enumerate(preds):
                centered_pred = torch.zeros_like(pred)
                pred[:, :4] = scale_coords(im_shape, pred[:, :4], im_shape)
                for j, (*xyxy, conf, cls) in reversed(list(enumerate(pred))):
                    centered_pred[j, :] = torch.tensor([*xyxy2xywh(torch.tensor(xyxy).view(1, 4)).view(-1), conf, cls])
                preds[i] = centered_pred.cpu().numpy()

        return preds

    def _run_loop(self):
        while True:
            tile_info = self._fast_read_queue.get()
            writer_nr = tile_info[-1]
            if tile_info[0] == 'finish_image':
                self._write_queues[writer_nr].put(tile_info[:-1])
                continue
            output_filename, sequence_nr, tile_batch, mask_batch, info, _ = tile_info
            result_batch = self._detect_objects(tile_batch, info)
            self._write_queues[writer_nr].put(('write_tile', output_filename, sequence_nr, result_batch, mask_batch, info))

    def run(self):
        """
        Called when the process is started. This runs the main steps of the tile processor.
        """
        self._model = self._load_network_model()
        self._initiate_fast_queue()
        self._run_loop()
