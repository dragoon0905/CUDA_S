from .deeplab_multi import DeeplabMulti


def get_model(cfg):
    if cfg.model.backbone == "deeplabv2_multi":
        model = DeeplabMulti(tm=cfg.tm, num_target=cfg.num_target, eval_target=cfg.eval_target, num_classes=cfg.data.num_classes, init=cfg.model.imagenet_pretrained)
        params = model.optim_parameters(lr=cfg.opt.lr, tm=cfg.tm)
    else:
        raise NotImplementedError()
    return model, params
