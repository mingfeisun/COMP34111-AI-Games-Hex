#!/usr/bin/env python3

import torch

from hexhex.model import hexconvolution
from hexhex.model import hexconvolutionunet
from hexhex.model import hexconvolutionattention
from hexhex.utils.logger import logger

import torch.nn as nn
from collections import OrderedDict
from typing import Optional

from enum import Enum
from typing import Dict, Any

class ModelType(Enum):
    STANDARD = "standard"
    ATTENTION = "attention" 
    UNET = "unet"

MODEL_CONFIGS: Dict[ModelType, Dict[str, Any]] = {
    ModelType.STANDARD: {
        "module": hexconvolution,
        "class": hexconvolution.Conv,
        "name": "Convolution, Standard"
    },
    ModelType.ATTENTION: {
        "module": hexconvolutionattention,
        "class": hexconvolutionattention.Conv,
        "name": "Attention"
    },
    ModelType.UNET: {
        "module": hexconvolutionunet,
        "class": hexconvolutionunet.UNetWithAttention,
        "name": "UNet with Attention"
    }
}

def transfer_model_weights(
    source_model: nn.Module, 
    target_model: nn.Module,
    save_path: Optional[str] = None,
    config: Optional[dict] = None
) -> nn.Module:
    """
    Transfer weights from source model to target model where layer shapes match.
    
    Args:
        source_model: Model to copy weights from
        target_model: Model to copy weights to
        save_path: Optional path to save updated model
        config: Optional config to save with model
    
    Returns:
        Updated target model
    """
    # Get state dictionaries
    source_state_dict = source_model.state_dict()
    target_state_dict = target_model.state_dict()

    # Create new state dict with matching weights
    new_state_dict = OrderedDict()
    for key in source_state_dict:
        if (key in target_state_dict and 
            source_state_dict[key].shape == target_state_dict[key].shape):
            new_state_dict[key] = source_state_dict[key]

    # Update target model weights
    target_state_dict.update(new_state_dict)
    target_model.load_state_dict(target_state_dict)

    # Optionally save updated model
    if save_path:
        save_dict = {'model_state_dict': target_model.state_dict()}
        if config:
            save_dict['config'] = config
        torch.save(save_dict, save_path)

    return target_model

def create_model(
    config, 
    export_mode: bool = False, 
    model_type: ModelType = ModelType.ATTENTION
) -> nn.Module:
    """Creates a model based on the specified type and configuration."""
    board_size = config.getint('board_size')
    switch_model = config.getboolean('switch_model')
    rotation_model = config.getboolean('rotation_model')

    model_config = MODEL_CONFIGS[model_type]
    model_module = model_config["module"]
    model_class = model_config["class"]

    model = model_class(
        board_size=board_size,
        layers=config.getint('layers'),
        intermediate_channels=config.getint('intermediate_channels'),
        reach=config.getint('reach'),
        export_mode=export_mode
    )

    if not switch_model:
        model = model_module.NoSwitchWrapperModel(model)

    if rotation_model:
        model = model_module.RotationWrapperModel(model, export_mode)

    return model

def create_and_store_model(config, name, from_pretrained=False):
    new_model = create_model(config)
    if from_pretrained:
        checkpoint = torch.load("data/azagent_0391.pt")
        trained_model = create_model(checkpoint['config'], model_type=ModelType.ATTENTION)
        trained_model.load_state_dict(checkpoint['model_state_dict'])
        trained_model.eval()
        torch.no_grad()
        new_model = transfer_model_weights(trained_model, new_model)
    model_file = f'models/{name}.pt'
    torch.save({
        'model_state_dict': new_model.state_dict(),
        'config': config
        }, model_file)
    logger.info(f'wrote {model_file}\n')
