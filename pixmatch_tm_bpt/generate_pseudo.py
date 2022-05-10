import os
import random
import logging
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm, trange
import numpy as np

import hydra
from omegaconf import OmegaConf, DictConfig
from torch.utils.tensorboard import SummaryWriter

from datasets.cityscapes_Dataset import City_Dataset, inv_preprocess, decode_labels
from datasets.gta5_Dataset import GTA5_Dataset
from datasets.synthia_Dataset import SYNTHIA_Dataset
from datasets.idd_dataset import IDDDataSet
from datasets.vistas_dataset import MapillaryDataSet
from perturbations.augmentations import augment, get_augmentation
from perturbations.fourier import fourier_mix
from perturbations.cutmix import cutmix_combine
from models_bpt import get_model
from models_bpt.ema import EMA
from utils.eval import Eval, synthia_set_16, synthia_set_13
import copy
from PIL import Image


palette = [128, 64, 128, 244, 35, 232, 70, 70, 70, 102, 102, 156, 190, 153, 153, 153, 153, 153, 250, 170, 30,
           220, 220, 0, 107, 142, 35, 70, 130, 180, 220, 20, 60, 255, 0, 0, 0, 0, 142, 0, 0, 70,
           0, 60, 100, 0, 80, 100, 0, 0, 230, 119, 11, 32]
zero_pad = 256 * 3 - len(palette)
for i in range(zero_pad):
    palette.append(0)

def colorize_mask(mask):
    # mask: numpy array of the mask
    new_mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
    new_mask.putpalette(palette)
    return new_mask

class Trainer():
    def __init__(self, cfg, logger, writer):

        # Args
        self.cfg = cfg
        self.device = torch.device('cuda')
        self.logger = logger
        self.writer = writer

        # Counters
        self.epoch = 0
        self.iter = 0
        self.current_MIoU = 0
        self.best_MIou = 0
        self.best_source_MIou = 0

        # Metrics
        self.evaluator = Eval(self.cfg.data.num_classes)

        # Loss
        self.ignore_index = -1
        self.loss = nn.CrossEntropyLoss(ignore_index=self.ignore_index)

        # Model
        self.model, params = get_model(self.cfg)
        # self.model = nn.DataParallel(self.model, device_ids=[0])  # TODO: test multi-gpu
        self.model.to(self.device)
        #num_target
        self.num_target=self.cfg.num_target
        print(self.num_target)
        for k, v in self.model.named_parameters():
            if 'TM_1' in k:
                v.requires_grad = False
                print(k)


        # EMA
        self.ema = EMA(self.model, self.cfg.ema_decay)

        # Optimizer
        self.optimizer = torch.optim.SGD(params,
                            lr=cfg.opt.learning_rate, momentum=cfg.opt.momentum, weight_decay=cfg.opt.weight_decay)
        # if self.cfg.opt.kind == "SGD":
        #     self.optimizer = torch.optim.SGD(
        #         params, momentum=self.cfg.opt.momentum, weight_decay=self.cfg.opt.weight_decay)

        # elif self.cfg.opt.kind == "Adam":
        #     self.optimizer = torch.optim.Adam(params, betas=(
        #         0.9, 0.99), weight_decay=self.cfg.opt.weight_decay)
        # else:
        #     raise NotImplementedError()
        self.lr_factor = 10


        # Source
        if self.cfg.data.source.dataset == 'synthia':
            source_train_dataset = SYNTHIA_Dataset(split='train', **self.cfg.data.source.kwargs)
            source_val_dataset = SYNTHIA_Dataset(split='val', **self.cfg.data.source.kwargs)
        elif self.cfg.data.source.dataset == 'gta5':
            source_train_dataset = GTA5_Dataset(split='train', **self.cfg.data.source.kwargs)
            source_val_dataset = GTA5_Dataset(split='val', **self.cfg.data.source.kwargs)
        else:
            raise NotImplementedError()
        self.source_dataloader = DataLoader(
            source_train_dataset, shuffle=True, drop_last=True, **self.cfg.data.loader.kwargs)
        self.source_val_dataloader = DataLoader(
            source_val_dataset, shuffle=False, drop_last=False, **self.cfg.data.loader.kwargs)

        # Target
        if self.cfg.data.target.dataset == 'cityscapes':
            target_train_dataset = City_Dataset(split='train', **self.cfg.data.target.kwargs)
            target_val_dataset = City_Dataset(split='val', **self.cfg.data.target.kwargs)
            target_memory_dataset=City_Dataset(split='memory', **self.cfg.data.target.kwargs)
        elif self.cfg.data.target.dataset == 'IDD':
            target_train_dataset = IDDDataSet(split='train', **self.cfg.data.target.kwargs)
            target_val_dataset = IDDDataSet(split='val', **self.cfg.data.target.kwargs)
        elif self.cfg.data.target.dataset == 'MapillaryVistas':
            target_train_dataset = MapillaryDataSet(split='train', **self.cfg.data.target.kwargs)
            target_val_dataset = MapillaryDataSet(split='val', **self.cfg.data.target.kwargs)
        else:
            raise NotImplementedError()
        self.target_dataloader = DataLoader(
            target_train_dataset, shuffle=True, drop_last=True, **self.cfg.data.loader.kwargs)
        self.target_val_dataloader = DataLoader(
            target_val_dataset, shuffle=False, drop_last=False, **self.cfg.data.loader.kwargs)
        self.target_memory_dataloader = DataLoader(
            target_memory_dataset, shuffle=False, drop_last=False, **self.cfg.data.loader.kwargs)

        # Perturbations
        if self.cfg.lam_aug > 0:
            self.aug = get_augmentation()

    def train(self):
        self.train_one_epoch()

    def train_one_epoch(self):


        # Helper
        def unpack(x):
            return (x[0], x[1]) if isinstance(x, tuple) else (x, None)

        # Training loop
        total = min(len(self.source_dataloader), len(self.target_memory_dataloader))
        for batch_idx, (batch_s, batch_t) in enumerate(tqdm(
            zip(self.source_dataloader, self.target_memory_dataloader),
            total=total, desc=f"Epoch {self.epoch + 1}"
        )):



            # Load and reset
            self.model.train()
            self.evaluator.reset()


            ######################
            # Target Pseudolabel #
            ######################
            x, gt, name = batch_t
            x = x.to(self.device)
            with torch.no_grad():
                # Substep 1: forward pass
                pred = self.model(x.to(self.device),forward_target=self.num_target)
                pred_1, pred_2 = unpack(pred)

                # Substep 2: convert soft predictions to hard predictions
                pred_P_1 = F.softmax(pred_1, dim=1)
                label_1 = torch.argmax(pred_P_1.detach(), dim=1)
                maxpred_1, argpred_1 = torch.max(pred_P_1.detach(), dim=1)
                T = self.cfg.pseudolabel_threshold
                mask_1 = (maxpred_1 > T)
                ignore_tensor = torch.ones(1).to(
                    self.device, dtype=torch.long) * self.ignore_index
                label_1 = torch.where(mask_1, label_1, ignore_tensor)
                if self.cfg.aux:
                    pred_P_2 = F.softmax(pred_2, dim=1)
                    maxpred_2, argpred_2 = torch.max(pred_P_2.detach(), dim=1)
                    pred_c = (pred_P_1 + pred_P_2) / 2
                    maxpred_c, argpred_c = torch.max(pred_c, dim=1)
                    mask = (maxpred_1 > T) | (maxpred_2 > T)
                    label_2 = torch.where(mask, argpred_c, ignore_tensor)
                ###### save psuedo label
                #label=gt.squeeze(dim=1).cpu.numpy()
                #self.evaluator.add_batch(label, label_1)
                #MIoU = self.evaluator.Mean_Intersection_over_Union()
                #self.logger.info('MIoU:{:.3f}'.format(MIoU))
                pred_1=pred_1.detach().cpu().numpy()
                np.save('/root/test/Methods/CUDA/pixmatch_tm/Pseudo/pseudo_city/%s.npy' % (name[0]),pred_1)
                pseudo_ = label_1
                print(label_1)
                pseudo_ = pseudo_.detach().cpu().numpy()
                pseudo_ = np.asarray(pseudo_, dtype=np.uint8).squeeze(0)
                
                output=Image.fromarray(pseudo_)
                output.save('/root/test/Methods/CUDA/pixmatch_tm/Pseudo/pseudo_city/%s.png' % (name[0]))
                
                output_col = colorize_mask(pseudo_)
                output_col.save('/root/test/Methods/CUDA/pixmatch_tm/Pseudo/pseudo_city/%s_color.png' % (name[0]))
                
            

    def load_checkpoint(self, filename):
        checkpoint = torch.load(filename, map_location='cpu')

        # Get model state dict
        if not self.cfg.train and 'shadow' in checkpoint:
            state_dict = checkpoint['shadow']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint

        # Remove DP/DDP if it exists
        state_dict = {k.replace('module.', ''): v for k,
                      v in state_dict.items()}

        # Load state dict
        if hasattr(self.model, 'module'):
            self.model.module.load_state_dict(state_dict, strict=False)
        else:
            self.model.load_state_dict(state_dict, strict=False)
        self.logger.info(f"Model loaded successfully from {filename}")

        # Load optimizer and epoch
        if self.cfg.train and self.cfg.model.resume_from_checkpoint:
            if 'optimizer' in checkpoint:
                self.optimizer.load_state_dict(checkpoint['optimizer'])
                self.logger.info(f"Optimizer loaded successfully from {filename}")
            if 'epoch' in checkpoint and 'iter' in checkpoint:
                 self.epoch = checkpoint['epoch']
                 self.iter = checkpoint['iter'] if 'iter' in checkpoint else checkpoint['iteration']
                 self.logger.info(f"Resuming training from epoch {self.epoch} iter {self.iter}")
        else:
            self.logger.info(f"Did not resume optimizer")

@hydra.main(config_path='configs', config_name='gta5')
def main(cfg: DictConfig):

    # Seeds
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.random.manual_seed(cfg.seed)

    # Logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    # Monitoring
    if cfg.wandb:
        import wandb
        wandb.init(project='pixmatch', name=cfg.name, config=cfg, sync_tensorboard=True)
    writer = SummaryWriter(cfg.name)

    # Trainer
    trainer = Trainer(cfg=cfg, logger=logger, writer=writer)

    # Load pretrained checkpoint
    if cfg.model.checkpoint:
        assert Path(cfg.model.checkpoint).is_file(), f'not a file: {cfg.model.checkpoint}'
        trainer.load_checkpoint(cfg.model.checkpoint)

    # Print configuration
    logger.info('\n' + OmegaConf.to_yaml(cfg))

    # Train
    if cfg.train:
        trainer.train()

    # Evaluate
    else:
        trainer.validate()
        trainer.evaluator.Print_Every_class_Eval(
            out_16_13=(int(cfg.data.num_classes) in [16, 13]))


if __name__ == '__main__':
    main()
