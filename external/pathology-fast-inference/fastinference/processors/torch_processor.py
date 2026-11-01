import os
import numpy as np
import torch
import torch.nn as nn
from importlib import import_module
import torch.utils.data as data
import segmentation_models_pytorch as smp
import yaml
from ..async_tile_processor import async_tile_processor

# fastinference/processors/torch_processor.py
class torch_processor(async_tile_processor): 
    def __init__(self, **kwargs):
        self.encoder_name  = kwargs.pop("encoder_name", None)
        self.decoder_name  = kwargs.pop("decoder_name", None)
        self.num_classes   = kwargs.pop("num_classes", None)
        self.model_name    = kwargs.pop("model_name", None)
        self.checkpoint    = kwargs.pop("checkpoint", None)
        self.encoder_weights = kwargs.pop("encoder_weights", None)

        super().__init__(**kwargs)  # now only args that async_tile_processor accepts
        self.softmax_fn = nn.LogSoftmax(dim=1)

    def get_config_from_yaml(self, config_path: str) -> dict:
        with open(file=config_path, mode='r') as param_file:
            parameters = yaml.load(stream=param_file, Loader=yaml.SafeLoader)
        return parameters['model'], parameters['sampler'], parameters['training']          

    def _load_network_model(self):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # The model configuration YAML sits next to the weights and shares their base name,
        # e.g. tissue_network_unet_mobilenetv2.pth -> tissue_network_unet_mobilenetv2.yaml.
        # Suffixes such as "_best_model" or "_epoch<N>" are stripped for older checkpoint names.
        base_path = os.path.splitext(self._model_path)[0]
        config_path = base_path.split("_epoch")[0].split("_best")[0] + '.yaml'
        print("Device: ", device)
        model_parameters, sampler_parameters, training_parameters = self.get_config_from_yaml(config_path)
        n_classes = len(np.unique(list(sampler_parameters['training']['label_map'].values())))
    
        model = smp.Unet(encoder_name=model_parameters['backbone'], 
                classes=n_classes, 
                encoder_weights=model_parameters['encoder_weights'])

        state_dict = torch.load(self._model_path, map_location=torch.device(device))
        model.load_state_dict(state_dict)
        print("Model succesfully loaded!")
        model.to(device) 

        return model           

    def _predict_tile_batch(self, tile_batch=None, info=None):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if self._ax_order == 'cwh':
            tile_batch = tile_batch.transpose(0, 3, 1, 2)
        
        tile_batch = torch.from_numpy(tile_batch).to(device)
        
        with torch.cuda.amp.autocast():   
            result = self._model.predict(tile_batch)
            
        result = self.softmax_fn(result)
        result = result.detach().cpu().numpy()

        if self._ax_order == 'cwh':
            result = result.transpose(0, 2, 3, 1)
        
        return result

    def _send_reconstruction_info(self):
        self._write_queues[0].put(('recon_info',
                                   '',1))
