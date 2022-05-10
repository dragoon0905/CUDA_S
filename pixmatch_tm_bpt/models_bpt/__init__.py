from .deeplab_multi import DeeplabMulti


def get_model(cfg):
    if cfg.model.backbone == "deeplabv2_multi":
        tm = True
        num_target = cfg.num_target
        eval_target =cfg.num_target
        target_list=[]
        target_list.append(num_target)
        model = DeeplabMulti(tm, num_target, eval_target,num_classes=cfg.data.num_classes, init=cfg.model.imagenet_pretrained)
        params = model.optim_parameters(cfg,target_list)
    else:
        raise NotImplementedError()
    return model, params
