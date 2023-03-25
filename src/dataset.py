from pathlib import Path
import torch
import pytorch_lightning as pl

from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision.transforms import transforms, AutoAugment, AutoAugmentPolicy

BASE_DIR = Path(__file__).parent.parent


class PatchifyTransform:

    def __init__(self, patch_size):
        """Custom transform that patchifies image on
        patch_size x patch_size flattened patches.

        Args:
            patch_size: the size of patch
        """
        self.patch_size = patch_size

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        """Use torch.Tensor.unfold method to add patches
        to new dimension. Flatten new and color dimensions.

        Args:
            img: image tensor to patchify

        Returns:
            Patchified tensor
        """
        res = img.unfold(1, self.patch_size, self.patch_size)  # 3 x 8 x 32 x 4
        res = res.unfold(2, self.patch_size, self.patch_size)  # 3 x 8 x 8 x 4 x 4

        return res.reshape(-1, self.patch_size * self.patch_size * 3)  # -1 x 48 == 64 x 48


class CIFAR10DataModule(pl.LightningDataModule):

    def __init__(self, batch_size: int, patch_size: int = 4, val_batch_size: int = 16,
                 im_size: int = 32, rotation_degrees: (int, int) = (-30, 30)):
        """Lightning Data Module that manages CIFAR10 dataset and its data loaders

        Args:
            batch_size: the size of training batch
            patch_size: the size of image patch
            val_batch_size: the size of validation batch
        """
        super().__init__()

        self.batch_size = batch_size
        self.val_batch_size = val_batch_size
        self.train_transform = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomResizedCrop(size=(im_size, im_size)),
                transforms.RandomRotation(degrees=rotation_degrees),
                AutoAugment(AutoAugmentPolicy.CIFAR10),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
                PatchifyTransform(patch_size)
            ]
        )
        self.val_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
                PatchifyTransform(patch_size)
            ]
        )
        self.patch_size = patch_size
        self.ds_train = None
        self.ds_val = None

    def prepare_data(self) -> None:
        """Download CIFAR10 dataset into local directory"""
        CIFAR10(BASE_DIR.joinpath('data/cifar'), train=True, transform=self.train_transform, download=True)
        CIFAR10(BASE_DIR.joinpath('data/cifar'), train=False, transform=self.val_transform, download=True)

    def setup(self, stage: str) -> None:
        """Initialize train and validation datasets"""
        self.ds_train = CIFAR10(BASE_DIR.joinpath('data/cifar'), train=True, transform=self.train_transform)
        self.ds_val = CIFAR10(BASE_DIR.joinpath('data/cifar'), train=False, transform=self.val_transform)

    def train_dataloader(self):
        """Create dataloader for training"""
        # Due to small dataset we don't need to use multiprocessing
        return DataLoader(self.ds_train, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        """Create dataloader for validation"""
        return DataLoader(self.ds_val, batch_size=self.val_batch_size)

    @property
    def classes(self):
        """Returns the amount of CIFAR10 classes"""
        return 10  # CIFAR10 has 10 possible classes
