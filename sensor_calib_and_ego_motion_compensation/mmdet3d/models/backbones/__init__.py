from mmdet.models.backbones import SSDVGG, HRNet, ResNet, ResNetV1d, ResNeXt

from .resnet import *
from .second import *
from .sparse_encoder import *
from .pillar_encoder import *
from .vovnet import *
from .dla import *
try:
    from .radar_encoder import *
except ModuleNotFoundError as exc:
    # Radar models are optional for LiDAR and camera-LiDAR inference.
    if exc.name != "flash_attn" and not (exc.name or "").startswith("flash_attn."):
        raise
