"""
Based on template from https://github.com/PyTorchLightning/pytorch-lightning/blob/master/pl_examples/basic_examples/lightning_module_template.py

Adapted by Gerard
"""
import logging as log
import os
from argparse import ArgumentParser
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision.datasets import MNIST

from pytorch_lightning.core import LightningModule
from pytorch_lightning.core import data_loader

# import mlflow
# import mlflow.pytorch

import input_target_transforms as TT
import distributed_utils

# from faim_mlflow import get_run_manager
from ml_args import parse_args
from evaluation import evaluate
from models import get_model

from datasets import GameImagesDataset, GameFoldersDataset, OverfitDataset, get_dataset


class UnetLightning(LightningModule):
    """
    Unet Auto Encoder models with pytorch lightning
    """

    def __init__(self, hparams):
        """
        Pass in parsed HyperOptArgumentParser to the model
        :param hparams:
        """
        # init superclass
        super(LightningTemplateModel, self).__init__()
        self.hparams = hparams

        self.batch_size = hparams.batch_size

        # if you specify an example input, the summary will show input/output for each layer
        self.example_input_array = torch.rand((3, 224, 256))
        self.dataset = None
        # build model
        self.net = get_model(hparams.model_name)
    # ---------------------
    # TRAINING
    # ---------------------

    def forward(self, x):
        """
        Based on model choice
        :param x:
        :return:
        """
        return self.net.forward(x)

    def loss(self, targets, segmentations):
        """
        Mean Squared Error. Basic loss function for difference in image arrays
        :param targets: ground truth image or segmentation map
        :param segmentations: output of the autoencoder network
        """
        mse = F.mse_loss(segmentations, targets)
        return mse

    def training_step(self, batch, batch_idx):
        """
        Lightning calls this inside the training loop
        :param batch:
        :return:
        """
        # forward pass
        images, targets = batch
        predictions = self.forward(images)
        # calculate loss
        loss_val = self.loss(targets, predictions)
        return {'train_loss': loss_val}

    def validation_step(self, batch, batch_idx):
        """
        Called in validation loop with model in eval mode
        """
        images, targets = batch
        predictions = self.forward(images)
        loss_val = self.loss(targets, predictions)
        return {'val_loss': loss_val}

    def validation_end(self, outputs):
        val_losses = torch.stack([x['val_loss'] for x in outputs])
        avg_loss = val_losses.mean()
        max_loss = val_losses.max()
        return {'avg_val_loss': avg_loss,
                'max_val_loss': max_loss}

    # ---------------------
    # TRAINING SETUP
    # ---------------------
    def configure_optimizers(self):
        """
        REQUIRED
        can return multiple optimizers and learning_rate schedulers
        Use lists to return lr scheduler, else can just return optimizer
        :return: list of optimizers
        """
        args = self.hparams
        optimizer = torch.optim.SGD(
            self.net.parameters(),
            lr=args.lr,
            momentum=args.momentum, weight_decay=args.weight_decay
        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=10)
        # scheduler = torch.optim.lr_scheduler.LambdaLR(
        #     optimizer,
        #     lambda x: (1 - x / (len(self.train_dataloader()) * args.epochs)) ** 0.9)
        return [optimizer], [scheduler]

    def load_full_dataset(self, split):
        if self.dataset is None:
            self.dataset = get_dataset(
                self.hparams.dataset, 'train', transform=None)
            num_samples = len(self.dataset)
            num_train = int(0.8 * num_samples)
            log.info(f'Full Dataset loaded: len: {num_samples}')
            self.train_split, self.val_split = torch.utils.data.random_split(
                self.dataset, (num_train, num_samples - num_train))
            log.info(
                f'Train {len(self.train_split)} and Val {len(self.val_split)} splits made')

            self.train_split.transform = TT.get_transform(
                train=True, inpaint=False, noise=False)
            self.val_split.transform = TT.get_transform(
                train=False, inpaint=False, noise=False)

    @pl.data_loader
    def train_dataloader(self):
        """
        Required
        """
        log.info('Training data loader called.')
        self.load_full_dataset()
        return torch.utils.data.DataLoader(self.train_split, batch_size=self.hparams.batch_size, drop_last=True)

    @pl.data_loader
    def val_dataloader(self):
        log.info('Validation data loader called.')
        self.load_full_dataset()
        return torch.utils.data.DataLoader(self.val_split, batch_size=self.hparams.batch_size, drop_last=False)

    # @pl.data_loader
    # def test_dataloader(self):
    #     log.info('Test data loader called.')
    #     return self.__dataloader(train=False)


def main(args):
    # init module
    model = UnetLightning(args)

    # most basic trainer, uses good defaults
    trainer = Trainer(
        max_nb_epochs=args.epochs,
        gpus=args.gpus,
        nb_gpu_nodes=args.world_size,
    )
    trainer.fit(model)


if __name__ == '__main__':
    args = parse_args()
    main(args)
