from mmdet.registry import MODELS

# MMDetection 3 uses one model registry for backbones, necks, heads, and losses.
BACKBONES = HEADS = LOSSES = NECKS = MODELS
FUSIONMODELS = VTRANSFORMS = FUSERS = MODELS


def build_backbone(cfg):
    return BACKBONES.build(cfg)


def build_neck(cfg):
    return NECKS.build(cfg)


def build_vtransform(cfg):
    return VTRANSFORMS.build(cfg)


def build_fuser(cfg):
    return FUSERS.build(cfg)


def build_head(cfg):
    return HEADS.build(cfg)


def build_loss(cfg):
    return LOSSES.build(cfg)


def build_fusion_model(cfg, train_cfg=None, test_cfg=None):
    return FUSIONMODELS.build(
        cfg, default_args=dict(train_cfg=train_cfg, test_cfg=test_cfg)
    )


def build_model(cfg, train_cfg=None, test_cfg=None):
    return build_fusion_model(cfg, train_cfg=train_cfg, test_cfg=test_cfg)
