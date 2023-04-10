import timm
import logging
from models.wideresnet import WideResNet

logger = logging.getLogger(__name__)


def create_model(
    model_name, num_classes=10, pretrained: bool = False, dropout: float = 0.2, **kwargs
):
    if model_name == "WideResNet":
        logger.info(f"{WideResNet}")
        return WideResNet(
            num_classes=num_classes, depth=28, widen_factor=2, dropout=dropout
        )
    logger.info(f"timm model : {model_name}")
    return timm.create_model(model_name, num_classes=num_classes, pretrained=pretrained)


__all__ = ["WideResNet", "create_model"]
