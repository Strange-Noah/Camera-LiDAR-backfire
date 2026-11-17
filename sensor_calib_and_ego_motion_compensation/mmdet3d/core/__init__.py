"""Core utilities for the legacy BEVFusion model, with lazy-safe exports."""

from .bbox import *  # noqa: F401,F403
from .bbox.coders import *  # noqa: F401,F403
from .points import *  # noqa: F401,F403
from .post_processing.box3d_nms import circle_nms
from .utils.gaussian import draw_heatmap_gaussian, gaussian_radius
from .voxel import *  # noqa: F401,F403

# MMDetection 3 moved task utilities out of mmdet.core.
from mmdet.models.task_modules.samplers import PseudoSampler
