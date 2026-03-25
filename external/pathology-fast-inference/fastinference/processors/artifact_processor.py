from ..async_tile_processor import async_tile_processor
import os
import numpy as np
import torch
import time, os

class artifact_processor(async_tile_processor):
    def __init__(self, encoder_name='tu-convnext_base', encoder_weights='imagenet', decoder_name='deeplabv3plus', use_tta=False, **kwargs):
        super().__init__(**kwargs)

        self.encoder_name = encoder_name
        self.encoder_weights = None
        self.decoder_name = decoder_name
        self.use_tta = use_tta

    def _load_network_model(self):

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        aux_params = dict(
            pooling='avg',  # one of 'avg', 'max'
            dropout=0.4,  # dropout ratio, default is None
            activation='softmax2d',  # activation function, default is None
            classes=7)

        if self.decoder_name == 'deeplabv3plus':
            from segmentation_models_pytorch import DeepLabV3Plus
            self._model = DeepLabV3Plus(
                encoder_name=self.encoder_name,
                encoder_weights=self.encoder_weights,
                classes=7,
                activation='softmax2d',
                aux_params=aux_params)
        elif self.decoder_name == "unetplusplus":
            from segmentation_models_pytorch import UnetPlusPlus
            self._model = UnetPlusPlus(
                encoder_name=self.encoder_name,
                encoder_weights=self.encoder_weights,
                decoder_use_batchnorm=True,
                classes=7,
                activation='softmax2d',
                aux_params=aux_params)
        
        self._model = self._model.to(device)

        print("Loading model from: ", self._model_path)
        t0 = time.time()
        checkpoint = torch.load(self._model_path, map_location=torch.device(device))
        if 'model_state_dict' in checkpoint.keys():
            checkpoint = checkpoint['model_state_dict']
        
        self._model.load_state_dict(checkpoint)
        self._model.to(torch.device(device))
        self._model.eval()
        return self._model
    
    def augment_image(self, image):
        """Augment the image with the 8 rotation/mirroring configurations."""

        batch_image = np.empty(shape=(8,) + image.shape, dtype=image.dtype)
        batch_image[0] = image
        batch_image[1] = np.rot90(m=image, k=1, axes=(0, 1))
        batch_image[2] = np.rot90(m=image, k=2, axes=(0, 1))
        batch_image[3] = np.rot90(m=image, k=3, axes=(0, 1))
        batch_image[4] = np.fliplr(m=image)
        batch_image[5] = np.rot90(m=batch_image[4], k=1, axes=(0, 1))
        batch_image[6] = np.rot90(m=batch_image[4], k=2, axes=(0, 1))
        batch_image[7] = np.rot90(m=batch_image[4], k=3, axes=(0, 1))

        return batch_image
    
    def sum_augmentations(self, batch_result):
        """Sum the result of augmented inference."""

        batch_result[1] = np.rot90(m=batch_result[1], k=-1, axes=(0, 1))
        batch_result[2] = np.rot90(m=batch_result[2], k=-2, axes=(0, 1))
        batch_result[3] = np.rot90(m=batch_result[3], k=-3, axes=(0, 1))
        batch_result[4] = np.fliplr(m=batch_result[4])
        batch_result[5] = np.fliplr(m=np.rot90(m=batch_result[5], k=-1, axes=(0, 1)))
        batch_result[6] = np.fliplr(m=np.rot90(m=batch_result[6], k=-2, axes=(0, 1)))
        batch_result[7] = np.fliplr(m=np.rot90(m=batch_result[7], k=-3, axes=(0, 1)))

        return np.mean(a=batch_result, axis=0)

    def _predict_tile_batch(self, tile_batch=None, info=None):
        """
        Input: tile_batch of shape (N, H, W, C)
        Output: (N, H, W, C_out)
        """
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Ensure numpy
        if isinstance(tile_batch, torch.Tensor):
            tile_batch = tile_batch.detach().cpu().numpy()

        tile_batch = tile_batch.astype(np.float32)

        if not self.use_tta:
            # ------------------
            # no TTA
            # ------------------
            x = torch.from_numpy(tile_batch).permute(0, 3, 1, 2).to(device)

            with torch.no_grad():
                out = self._model(x)
                if isinstance(out, (tuple, list)):
                    out = out[0]

            return out.detach().cpu().numpy().transpose(0, 2, 3, 1)

        # ------------------
        # TTA PATH slow
        # ------------------
        results = []
        for tile in tile_batch:
            augmented = self.augment_image(tile)
            tta_tensor = torch.from_numpy(
                augmented.transpose(0, 3, 1, 2)
            ).float().to(device)

            with torch.no_grad():
                out = self._model(tta_tensor)
                if isinstance(out, (tuple, list)):
                    out = out[0]

                out = out.cpu().numpy().transpose(0, 2, 3, 1)
                merged = self.sum_augmentations(out).astype(np.float32)

            results.append(merged)

        return np.stack(results, axis=0)

    def _send_reconstruction_info(self):
        self._write_queues[0].put(('recon_info',
                                   '',1))
        
