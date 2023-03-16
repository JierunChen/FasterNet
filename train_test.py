import os
import sys
import torch
from torch import nn
from argparse import ArgumentParser
import wandb

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar, StochasticWeightAveraging
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import seed_everything
from pytorch_lightning.plugins import DDPPlugin

from utils.utils import *
from utils.fuse_conv_bn import fuse_conv_bn
from data.data_api import LitDataModule
from models.model_api import LitModel


def main(args):
    # Init data pipeline
    dm, _ = LitDataModule(hparams=args)

    # Init LitModel
    if args.checkpoint_path is not None:
        PATH = args.checkpoint_path
        if PATH[-5:]=='.ckpt':
            model = LitModel.load_from_checkpoint(PATH, map_location='cpu', num_classes=dm.num_classes, hparams=args)
            print('Successfully load the pl checkpoint file.')
            if args.pl_ckpt_2_torch_pth:
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                model = model.model.to(device)
                torch.save(model.state_dict(), PATH[:-5]+'.pth')
                exit()
        elif PATH[-4:] == '.pth':
            model = LitModel(num_classes=dm.num_classes, hparams=args)
            missing_keys, unexpected_keys = model.model.load_state_dict(torch.load(PATH), False)
            # show for debug
            print('missing_keys: ', missing_keys)
            print('unexpected_keys: ', unexpected_keys)
        else:
            raise TypeError
    else:
        model = LitModel(num_classes=dm.num_classes, hparams=args)

    flops, params = get_flops_params(model.model, args.image_size)

    if args.fuse_conv_bn:
        fuse_conv_bn(model.model)

    if args.measure_latency:
        dm.prepare_data()
        dm.setup(stage="test")
        for idx, (images, _) in enumerate(dm.test_dataloader()):
            model = model.model.eval()
            throughput, latency = measure_latency(images[:1, :, :, :], model, GPU=False, num_threads=1)
            if torch.cuda.is_available():
                throughput, latency = measure_latency(images, model, GPU=True)
            exit()

    print_model(model)

    # Callbacks
    MONITOR = 'val_acc1'
    checkpoint_callback = ModelCheckpoint(
        monitor=MONITOR,
        dirpath=args.model_ckpt_dir,
        filename=args.model_name+'-{epoch}-{val_acc1:.4f}',
        save_top_k=1,
        save_last=True,
        mode='max' if 'acc' in MONITOR else 'min'
    )
    refresh_callback = TQDMProgressBar(refresh_rate=20)
    callbacks = [
        checkpoint_callback,
        refresh_callback
    ]

    # Initialize wandb logger
    WANDB_ON = True if args.dev+args.test_phase==0 else False
    if WANDB_ON:
        wandb_logger = WandbLogger(
            project=args.wandb_project_name,
            save_dir=args.wandb_save_dir,
            offline=args.wandb_offline,
            log_model=False,
            job_type='train')
        wandb_logger.log_hyperparams(args)
        wandb_logger.log_hyperparams({"flops": flops, "params": params})

    # Initialize a trainer
    find_unused_para = False if args.distillation_type == 'none' else True
    trainer = pl.Trainer(
        fast_dev_run=args.dev,
        logger=wandb_logger if WANDB_ON else None,
        max_epochs=args.epochs,
        gpus=args.gpus,
        accelerator="gpu",
        sync_batchnorm=args.sync_batchnorm,
        num_nodes=args.num_nodes,
        gradient_clip_val=args.clip_grad,
        strategy=DDPPlugin(find_unused_parameters=find_unused_para) if args.strategy == 'ddp' else args.strategy,
        callbacks=callbacks,
        precision=args.precision,
        benchmark=args.benchmark
    )

    if bool(args.test_phase):
        trainer.test(model, datamodule=dm)
    else:
        trainer.fit(model, dm)
        if args.dev==0:
            trainer.test(ckpt_path="best", datamodule=dm)

    # Close wandb run
    if WANDB_ON:
        wandb.finish()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('-c', '--cfg', type=str, default='cfg/fasternet_t0.yaml')
    parser.add_argument('-g', "--gpus", type=str, default=None,
                        help="Number of GPUs to train on (int) or which GPUs to train on (list or str) applied per node.")
    parser.add_argument('-d', "--dev", type=int, default=0, help='fast_dev_run for debug')
    parser.add_argument("--num_nodes", type=int, default=1)
    parser.add_argument('-n', "--num_workers", type=int, default=4)
    parser.add_argument('-b', "--batch_size", type=int, default=2048)
    parser.add_argument('-e', "--batch_size_eva", type=int, default=1000, help='batch_size for evaluation')
    parser.add_argument("--model_ckpt_dir", type=str, default="./model_ckpt/")
    parser.add_argument("--data_dir", type=str, default="../../data/imagenet")
    parser.add_argument('--pin_memory', action='store_true')
    parser.add_argument("--checkpoint_path", type=str, default=None)
    parser.add_argument("--pconv_fw_type", type=str, default='split_cat',
                        help="use 'split_cat' for training/inference and 'slicing' only for inference")
    parser.add_argument('--measure_latency', action='store_true', help='measure latency or throughput')
    parser.add_argument('--test_phase', action='store_true')
    parser.add_argument('--fuse_conv_bn', action='store_true')
    parser.add_argument("--wandb_project_name", type=str, default="fasternet")
    parser.add_argument('--wandb_offline', action='store_true')
    parser.add_argument('--wandb_save_dir', type=str, default='./')
    parser.add_argument('--pl_ckpt_2_torch_pth', action='store_true',
                        help='convert pl .ckpt file to torch .pth file, and then exit')

    args = parser.parse_args()
    cfg = load_cfg(args.cfg)
    args = merge_args_cfg(args, cfg)

    # please change {WANDB_API_KEY} to your personal api_key before using wandb
    # os.environ["WANDB_API_KEY"] = "{WANDB_API_KEY}"

    main(args)
