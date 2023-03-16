# type: ignore[override]
import os
from typing import Any, Callable, Optional

import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from torchvision import datasets
from utils.utils import *

from pl_bolts.datasets import UnlabeledImagenet
from pl_bolts.transforms.dataset_normalizations import imagenet_normalization
from pl_bolts.utils import _TORCHVISION_AVAILABLE
from pl_bolts.utils.warnings import warn_missing_pkg

from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data import create_transform

if _TORCHVISION_AVAILABLE:
    from torchvision import transforms
else:  # pragma: no cover
    warn_missing_pkg("torchvision")


class custom_ImagenetDataModule(LightningDataModule):
    """
    The train set is the imagenet train.
    The val/test set are the official imagenet validation set.

    """

    name = "imagenet"

    def __init__(
        self,
        data_dir: str,
        meta_dir: Optional[str] = None,
        image_size: int = 224,
        num_workers: int = 0,
        batch_size: int = 32,
        batch_size_eva: int = 32,
        # dist_eval: bool = True,
        pin_memory: bool = True,
        drop_last: bool = False,
        train_transforms_multi_scale = None,
        scaling_epoch = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """
        Args:
            data_dir: path to the imagenet dataset file
            meta_dir: path to meta.bin file
            image_size: final image size
            num_workers: how many data workers
            batch_size: batch_size
            pin_memory: If true, the data loader will copy Tensors into CUDA pinned memory before
                        returning them
            drop_last: If true drops the last incomplete batch
        """
        super().__init__(*args, **kwargs)

        if not _TORCHVISION_AVAILABLE:  # pragma: no cover
            raise ModuleNotFoundError(
                "You want to use ImageNet dataset loaded from `torchvision` which is not installed yet."
            )

        self.image_size = image_size
        self.dims = (3, self.image_size, self.image_size)
        self.data_dir = data_dir
        self.num_workers = num_workers
        self.meta_dir = meta_dir
        self.batch_size = batch_size
        self.batch_size_eva = batch_size_eva
        self.pin_memory = pin_memory
        self.drop_last = drop_last
        self.num_samples = 1281167
        self.num_tasks = get_world_size()
        self.global_rank = get_rank()
        self.train_transforms_multi_scale = train_transforms_multi_scale
        self.scaling_epoch = scaling_epoch
        # self.dist_eval = dist_eval

    @property
    def num_classes(self) -> int:
        return 1000

    def _verify_splits(self, data_dir: str, split: str) -> None:
        dirs = os.listdir(data_dir)

        if split not in dirs:
            raise FileNotFoundError(
                f"a {split} Imagenet split was not found in {data_dir},"
                f" make sure the folder contains a subfolder named {split}"
            )

    def prepare_data(self) -> None:
        """This method already assumes you have imagenet2012 downloaded. It validates the data using the meta.bin.

        .. warning:: Please download imagenet on your own first.
        To get imagenet:
        1. download yourself from http://www.image-net.org/challenges/LSVRC/2012/downloads
        2. download the devkit (ILSVRC2012_devkit_t12.tar.gz)
        """
        self._verify_splits(self.data_dir, "train")
        self._verify_splits(self.data_dir, "val")

    def setup(self, stage: Optional[str] = None) -> None:
        """Creates train, val, and test dataset."""
        if stage == "fit" or stage is None:
            train_transforms = self.train_transform() if self.train_transforms is None else self.train_transforms
            val_transforms = self.val_transform() if self.val_transforms is None else self.val_transforms

            self.dataset_train = datasets.ImageFolder(os.path.join(self.data_dir, 'train'), transform=train_transforms)
            self.dataset_val = datasets.ImageFolder(os.path.join(self.data_dir, 'val'), transform=val_transforms)

            if self.train_transforms_multi_scale is not None:
                self.dataset_train_multi_scale = datasets.ImageFolder(os.path.join(self.data_dir, 'train'),
                                                          transform=self.train_transforms_multi_scale)
            else:
                self.dataset_train_multi_scale = None

        if stage == "test" or stage is None:
            val_transforms = self.val_transform() if self.val_transforms is None else self.val_transforms
            self.dataset_test = datasets.ImageFolder(os.path.join(self.data_dir, 'val'), transform=val_transforms)

    def train_dataloader(self) -> DataLoader:
        if self.dataset_train_multi_scale is not None and \
                self.trainer.current_epoch < self.scaling_epoch:
            dataset = self.dataset_train_multi_scale
            print("load dataset_train_multi_scale")
        else:
            dataset = self.dataset_train
            print("load dataset_train")

        loader: DataLoader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            drop_last=self.drop_last,
            pin_memory=self.pin_memory
        )
        return loader

    def val_dataloader(self) -> DataLoader:
        loader: DataLoader = DataLoader(
            self.dataset_val,
            batch_size=self.batch_size_eva,
            # persistent_workers=True,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=False,
            pin_memory=self.pin_memory
        )
        return loader

    def test_dataloader(self) -> DataLoader:
        """Uses the validation split of imagenet2012 for testing."""
        loader: DataLoader = DataLoader(
            self.dataset_test,
            batch_size=self.batch_size_eva,
            # persistent_workers=True,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=False,
            pin_memory=self.pin_memory
        )
        return loader

    def train_transform(self) -> Callable:
        preprocessing = transforms.Compose(
            [
                transforms.RandomResizedCrop(self.image_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                imagenet_normalization(),
            ]
        )

        return preprocessing

    def val_transform(self) -> Callable:

        preprocessing = transforms.Compose(
            [
                transforms.Resize(self.image_size + 32),
                transforms.CenterCrop(self.image_size),
                transforms.ToTensor(),
                imagenet_normalization(),
            ]
        )
        return preprocessing


def build_imagenet_transform(is_train, args, image_size):
    resize_im = image_size > 32
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=image_size,
            is_training=True,
            # use_prefetcher=args.use_prefetcher,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation=args.train_interpolation,
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
        )
        if not resize_im:
            # replace RandomResizedCropAndInterpolation with
            # RandomCrop
            transform.transforms[0] = transforms.RandomCrop(
                image_size, padding=4)
        return transform

    t = []
    if resize_im:
        # warping (no cropping) when evaluated at 384 or larger
        if image_size >= 384:
            t.append(
                transforms.Resize((image_size, image_size),
                                  interpolation=transforms.InterpolationMode.BICUBIC),
            )
            print(f"Warping {image_size} size input images...")
        else:
            # size = int((256 / 224) * image_size)
            size = int(1.0*image_size/args.test_crop_ratio)
            t.append(
                transforms.Resize(size, interpolation=transforms.InterpolationMode.BICUBIC),  # to maintain same ratio w.r.t. 224 images
            )
            t.append(transforms.CenterCrop(image_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
    return transforms.Compose(t)