import json
from typing import List

import torch.nn as nn
from mmengine.dist import get_dist_info
from mmengine.logging import MMLogger
from mmengine.optim import DefaultOptimWrapperConstructor

from mmseg.registry import OPTIM_WRAPPER_CONSTRUCTORS


@OPTIM_WRAPPER_CONSTRUCTORS.register_module()
class GeoMoELayerDecayOptimizerConstructor(DefaultOptimWrapperConstructor):
    """
    Custom Optimizer Constructor with Layer-wise Learning Rate Decay.

    This constructor implements a specific layer-wise learning rate decay
    strategy based on the provided parameter naming convention.
    """

    def add_params(self, params: List[dict], module: nn.Module,
                   **kwargs) -> None:
        """
        Add parameters with layer-wise learning rate decay to the optimizer.
        """
        logger = MMLogger.get_current_instance()
        
        layer_decay_rate = self.paramwise_cfg.get('layer_decay_rate', 0.75)
        
        base_lr = self.base_lr
        weight_decay = self.base_wd


        block_len = [len(module.backbone.blocks1), len(module.backbone.blocks2), len(module.backbone.blocks3)]

        num_layers = sum(block_len) + 4 + 1
        layer_scales = [layer_decay_rate**(num_layers - i - 1) for i in range(num_layers)]
        
        logger.info(f"Block lengths: {block_len}, Total layers for LRD: {num_layers}, Decay rate: {layer_decay_rate}")

        parameter_groups = {}
        for name, param in module.named_parameters():
            if not param.requires_grad:
                continue 

            if name.startswith("backbone.blocks1"):
                layer_id = int(name.split(".")[2]) + 1
            elif name.startswith("backbone.blocks2"):
                layer_id = int(name.split(".")[2]) + 2 + block_len[0]
            elif name.startswith("backbone.blocks3"):
                layer_id = int(name.split(".")[2]) + 4 + block_len[0] + block_len[1]
            elif name.startswith("backbone.patch_embed1") or name.startswith("backbone.pos_embed"):
                layer_id = 0
            elif name.startswith("backbone.patch_embed2"):
                layer_id = 1 + block_len[0]
            elif name.startswith("backbone.patch_embed3"):
                layer_id = 2 + block_len[0] + block_len[1]
            elif name.startswith("backbone.patch_embed4"):
                layer_id = 3 + block_len[0] + block_len[1]
            elif name.startswith("backbone.stage"):
                layer_id = num_layers - 2
            else:
                layer_id = num_layers - 1

            if param.ndim == 1 or name.endswith('.bias'):
                group_name = "no_decay"
                this_weight_decay = 0.
            else:
                group_name = "decay"
                this_weight_decay = weight_decay

            group_key = f"layer_{layer_id}_{group_name}"
            if group_key not in parameter_groups:
                parameter_groups[group_key] = {
                    "params": [],
                    "param_names": [],
                    "weight_decay": this_weight_decay,
                    "lr": layer_scales[layer_id] * base_lr,
                    "lr_scale": layer_scales[layer_id],
                    "group_name": group_key
                }
            
            parameter_groups[group_key]["params"].append(param)
            parameter_groups[group_key]["param_names"].append(name)


        params.extend(parameter_groups.values())
        rank, _ = get_dist_info()
        if rank == 0:
            max_lr = 0
            for value in parameter_groups.values():
                max_lr = max(max_lr, value['lr'])
            logger.info('*'*10+f'Max lr: {max_lr}, base_lr: {base_lr}'+'*'*10)

        '''
        rank, _ = get_dist_info()
        if rank == 0:
            to_display = {}
            for key in sorted(parameter_groups.keys(), key=lambda x: int(x.split('_')[1])):
                pg = parameter_groups[key]
                to_display[key] = {
                    'param_names': pg['param_names'],
                    'lr': pg['lr'],
                    'lr_scale': pg['lr_scale'],
                    'weight_decay': pg['weight_decay'],
                }
            logger.info(f'Parameter Groups = {json.dumps(to_display, indent=2)}')
        '''

@OPTIM_WRAPPER_CONSTRUCTORS.register_module()
class MoELayerDecayOptimizerConstructor(DefaultOptimWrapperConstructor):
    """
    Custom Optimizer Constructor with Layer-wise Learning Rate Decay.

    This constructor implements a specific layer-wise learning rate decay
    strategy based on the provided parameter naming convention.
    """

    def add_params(self, params: List[dict], module: nn.Module,
                   **kwargs) -> None:
        """
        Add parameters with layer-wise learning rate decay to the optimizer.
        """
        logger = MMLogger.get_current_instance()
        logger.info(f'Using CustomLayerDecayOptimizerConstructor.')
        
        layer_decay_rate = self.paramwise_cfg.get('layer_decay_rate', 0.75)
        
        base_lr = self.base_lr
        weight_decay = self.base_wd


        try:
            block_len = len(module.backbone.blocks)
        except AttributeError as e:
            logger.error(f"The model structure does not match the expected format. Error: {e}")
            raise

        num_layers = block_len + 1 + 1
        layer_scales = [layer_decay_rate**(num_layers - i - 1) for i in range(num_layers)]
        
        logger.info(f"Block lengths: {block_len}, Total layers for LRD: {num_layers}, Decay rate: {layer_decay_rate}")

        parameter_groups = {}
        for name, param in module.named_parameters():
            if not param.requires_grad:
                continue  

            if name.startswith("backbone.blocks"):
                layer_id = int(name.split(".")[2]) + 1
            elif name.startswith("backbone.patch_embed") or name.startswith("backbone.pos_embed"):
                layer_id = 0
            else:
                layer_id = num_layers - 1


            if param.ndim == 1 or name.endswith('.bias'):
                group_name = "no_decay"
                this_weight_decay = 0.
            else:
                group_name = "decay"
                this_weight_decay = weight_decay


            group_key = f"layer_{layer_id}_{group_name}"
            if group_key not in parameter_groups:
                parameter_groups[group_key] = {
                    "params": [],
                    "param_names": [],
                    "weight_decay": this_weight_decay,
                    "lr": layer_scales[layer_id] * base_lr,
                    "lr_scale": layer_scales[layer_id],
                    "group_name": group_key
                }
            
            parameter_groups[group_key]["params"].append(param)
            parameter_groups[group_key]["param_names"].append(name)

      
        params.extend(parameter_groups.values())
        

        '''
        rank, _ = get_dist_info()
        if rank == 0:
            to_display = {}
            for key in sorted(parameter_groups.keys(), key=lambda x: int(x.split('_')[1])):
                pg = parameter_groups[key]
                to_display[key] = {
                    'param_names': pg['param_names'],
                    'lr': pg['lr'],
                    'lr_scale': pg['lr_scale'],
                    'weight_decay': pg['weight_decay'],
                }
            logger.info(f'Parameter Groups = {json.dumps(to_display, indent=2)}')
        '''