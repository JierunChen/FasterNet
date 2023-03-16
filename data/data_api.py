import os
import sys
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from utils.utils import str2list
from data.custom_imagenet_data import custom_ImagenetDataModule
from data.custom_imagenet_data import build_imagenet_transform


__all__ = ['LitDataModule']


def LitDataModule(hparams):
    dm =None
    CLASS_NAMES = None
    drop_last = True
    bs = hparams.batch_size
    bs_eva = hparams.batch_size_eva
    n_gpus = len(str2list(hparams.gpus)) if hparams.gpus is not None else hparams.devices
    n_nodes = hparams.num_nodes
    batch_size = int(1.0*bs / n_gpus / n_nodes) if hparams.strategy == 'ddp' else bs
    batch_size_eva = int(1.0*bs_eva / n_gpus / n_nodes) if hparams.strategy == 'ddp' else bs_eva

    if hparams.dataset_name == "imagenet":
        dm = custom_ImagenetDataModule(
            image_size=hparams.image_size,
            data_dir=hparams.data_dir,
            train_transforms=build_imagenet_transform(is_train=True, args=hparams, image_size=hparams.image_size),
            train_transforms_multi_scale=None if hparams.multi_scale is None else build_imagenet_transform(
                is_train=True, args=hparams, image_size=int(hparams.multi_scale.split('_')[0])),
            scaling_epoch=None if hparams.multi_scale is None else int(hparams.multi_scale.split('_')[1]),
            val_transforms=build_imagenet_transform(is_train=False, args=hparams, image_size=hparams.image_size),
            num_workers=hparams.num_workers,
            pin_memory=hparams.pin_memory,
            # dist_eval= True if len(str2list(hparams.gpus))>1 else False,
            batch_size=batch_size,
            batch_size_eva=batch_size_eva,
            drop_last=drop_last
        )
    else:
        print("Invalid dataset name, exiting...")
        exit()

    return dm, CLASS_NAMES