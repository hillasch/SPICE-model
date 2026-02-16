

from torch import get_device


def get_frozen_model(device=None):
    """Load and freeze DINOv2 base model for feature extraction."""
    import torch
    from transformers import AutoImageProcessor, AutoModel

    device = device or get_device()

    model_name = "facebook/dinov2-base"
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)

    for param in model.parameters():
        param.requires_grad = False
    model.eval()
    return model, processor
