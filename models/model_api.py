import os
import sys
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from torch.optim.lr_scheduler import *
from models import *
from utils.utils import *
import torch
import pytorch_lightning as pl

from timm.data import Mixup
from timm.models import create_model
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.scheduler import create_scheduler
from timm.optim import create_optimizer
from timm.utils import accuracy
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from utils.loss import DistillationLoss


def build_criterion(args):
    if args.mixup > 0.:
        # smoothing is handled with mixup label transform
        criterion = SoftTargetCrossEntropy()
    elif args.smoothing:
        criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        criterion = torch.nn.CrossEntropyLoss()
    return criterion


def build_mixup_fn(args, num_classes):
    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    if mixup_active:
        mixup_fn = Mixup(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
            label_smoothing=args.smoothing, num_classes=num_classes)
    return mixup_fn


class LitModel(pl.LightningModule):
    def __init__(self, num_classes, hparams):
        super().__init__()

        self.save_hyperparameters(hparams)
        if 'fasternet' in hparams.model_name:
            self.model = create_model(
                hparams.model_name,
                mlp_ratio=hparams.mlp_ratio,
                embed_dim=hparams.embed_dim,
                depths=hparams.depths,
                pretrained=hparams.pretrained,
                n_div=hparams.n_div,
                feature_dim=hparams.feature_dim,
                patch_size=hparams.patch_size,
                patch_stride=hparams.patch_stride,
                patch_size2=hparams.patch_size2,
                patch_stride2=hparams.patch_stride2,
                num_classes=num_classes,
                layer_scale_init_value=hparams.layer_scale_init_value,
                drop_path_rate=hparams.drop_path_rate,
                norm_layer=hparams.norm_layer,
                act_layer=hparams.act_layer,
                pconv_fw_type=hparams.pconv_fw_type
            )
        else:
            self.model = create_model(
                hparams.model_name,
                pretrained=hparams.pretrained,
                num_classes=num_classes
            )

        base_criterion = build_criterion(hparams)
        self.distillation_type = hparams.distillation_type
        if hparams.distillation_type == 'none':
            self.criterion = base_criterion
        else:
            # assert hparams.teacher_path, 'need to specify teacher-path when using distillation'
            print(f"Creating teacher model: {hparams.teacher_model}")
            teacher_model = create_model(
                hparams.teacher_model,
                pretrained=True,
                num_classes=num_classes,
                global_pool='avg',
            )
            for param in teacher_model.parameters():
                param.requires_grad = False
            teacher_model.eval()
            self.criterion = DistillationLoss(base_criterion,
                                              teacher_model,
                                              hparams.distillation_type,
                                              hparams.distillation_alpha,
                                              hparams.distillation_tau
                                              )
        self.criterion_eva = torch.nn.CrossEntropyLoss()
        self.mixup_fn = build_mixup_fn(hparams, num_classes)

    def forward(self, x):
        return self.model(x)

    def on_train_epoch_start(self):
        if self.hparams.multi_scale is not None:
            if self.current_epoch == int(self.hparams.multi_scale.split('_')[1]):
                # image_size = self.hparams.image_size
                self.trainer.reset_train_dataloader(self)

    def _calculate_loss(self, batch, mode="train"):
        imgs, labels = batch
        if mode == "train" and self.mixup_fn is not None:
            imgs, labels = self.mixup_fn(imgs, labels)
        preds = self.model(imgs)

        if mode == "train":
            if self.distillation_type == 'none':
                loss = self.criterion(preds, labels)
            else:
                loss = self.criterion(imgs, preds, labels)
            self.log("%s_loss" % mode, loss)
        else:
            loss = self.criterion_eva(preds, labels)
            acc1, acc5 = accuracy(preds, labels, topk=(1, 5))
            sync_dist = True if torch.cuda.device_count() > 1 else False
            self.log("%s_loss" % mode, loss, sync_dist=sync_dist)
            self.log("%s_acc1" % mode, acc1, sync_dist=sync_dist)
            self.log("%s_acc5" % mode, acc5, sync_dist=sync_dist)

        return loss

    def training_step(self, batch, batch_idx):
        loss = self._calculate_loss(batch, mode="train")
        return loss

    def validation_step(self, batch, batch_idx):
        self._calculate_loss(batch, mode="val")

    def test_step(self, batch, batch_idx):
        self._calculate_loss(batch, mode="test")

    def configure_optimizers(self):
        optimizer = create_optimizer(self.hparams, self.parameters())
        if self.hparams.sched == 'cosine':
            scheduler = LinearWarmupCosineAnnealingLR(optimizer,
                            warmup_epochs=self.hparams.warmup_epochs,
                            max_epochs=self.hparams.epochs,
                            warmup_start_lr=self.hparams.warmup_lr,
                            eta_min=self.hparams.min_lr
                        )
        else:
            # scheduler, _ = create_scheduler(self.hparams, optimizer)
            raise NotImplementedError

        return [optimizer], [scheduler]

