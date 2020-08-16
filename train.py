"""
This file defines the core research contribution   
"""
import os
import sys
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, Sequence
import random
import copy
from PIL import Image
import numpy as np
import glob
import warnings
import json
import collections
import contextlib
import subprocess
import socket
import argparse

from easydict import EasyDict as ezdict
from tqdm.auto import tqdm
from tqdm.contrib import DummyTqdmFile
import torch
import torchvision
import torchvision.transforms as transforms
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.optim.optimizer import Optimizer

import pytorch_lightning as pl
from conf_eval.training import change_last_layer, model_load, MultiClassErrorMeter, MultiLabelErrorMeter, get_calibration, bce_ignore_with_logits

import conf_eval.utils as MU
import evaluate
from conf_eval import datasets as mydatasets
from conf_eval.paths import paths


lookup = ezdict(
    resnet18 = dict(
        lfwp_gender  = dict(
            # Number of class
            num_outputs = 2,
            # Dataset class / function handles
            DS          = 'LFWpGenderFamUnfDataset',
            # learning rate
            lr          = 0.0080,
            # num of epochs
            max_epochs  = 24,
            # batch size
            batch_size  = 128,
            # label style
            label_style = 'MC',
        ),
        cat_vs_dog   = dict(
            num_outputs = 2,
            DS          = 'PetsFamUnfDataset',
            lr          = 0.0600,
            max_epochs  = 128,
            batch_size  = 128,
            label_style = 'MC',
        ),
        imnet_animal = dict(
            num_outputs = 4,
            DS          = 'ImageNetFamUnfDataset',
            lr          = 0.0030,
            max_epochs  = 64,
            batch_size  = 128,
            label_style = 'MC',
        ),
        voc_to_coco  = dict(
            num_outputs = 20,
            DS          = 'VOCCocoFamUnfDataset',
            lr          = 0.0300,
            max_epochs  = 32,
            batch_size  = 128,
            label_style = 'ML',
        ),
    ),
    densenet161 = dict(
        lfwp_gender  = dict(
            num_outputs = 2,
            DS          = 'LFWpGenderFamUnfDataset',
            lr          = 0.0040,
            max_epochs  = 96,
            batch_size  = 32,
            label_style = 'MC',
        ),
    ),
)
        

class plModelWrapped(pl.LightningModule):
    '''Pytorch Lightning module to train, validate, and test on familiar splits.'''

    def backbone_function(self, backbone):
        '''Get backbone model by name'''
        assert hasattr(torchvision.models, backbone), 'Backbone %s not found in torchvision.models' % backbone
        return getattr(torchvision.models, backbone)

    def __init__(self, hparams):
        # save hparams
        super().__init__()
        self.hparams = hparams
        # model
        self.model = self.backbone_function(hparams.backbone)(pretrained=False, progress=True)
        lln = 'classifier' if hparams.backbone.startswith('densenet') else 'fc' # last layer's name
        # label style
        assert hparams.label_style in ['MC', 'ML']
        if hparams.label_style == 'MC':
            self.Meter = MultiClassErrorMeter
            self.loss = F.cross_entropy
        elif hparams.label_style == 'ML':
            self.Meter = MultiLabelErrorMeter
            self.loss = bce_ignore_with_logits
        else:
            raise ValueError('label_style ("ML" or "MC") not recognized: %s' % hparams.label_style)
        # load the models from downloaded version
        assert hparams.pretrained == 'Places365'
        change_last_layer( self.model, lln, num_outputs=365 )
        model_load( paths.prePlaces365[ hparams.backbone ], self.model )
        change_last_layer( self.model, lln, num_outputs=hparams.num_outputs )
        # store lr to log
        self.lr = hparams.lr
        self.lr_now = None
        # performance meters. Lightning interlaces train/eval, so need to use two meters
        self.train_error_meter = self.Meter()
        self.eval_error_meter = self.Meter()
        # data transforms
        self.transform_train = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.transform_eval = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        # forward pass
        x, y = batch
        y_hat = self.forward(x)
        # NaN alert
        assert torch.all(y_hat == y_hat)
        # loss and performance
        res = dict(
            loss = self.loss(y_hat, y),
        )
        self.train_error_meter.add(y_hat, y)
        # logs and progress bar
        if self.lr_now is not None:
            res['lr_batch'] = torch.tensor(self.lr_now)
        pbar = dict(lr=self.lr_now if self.lr_now is not None else '')
        return dict(log=res, progress_bar=pbar, **res)

    def training_epoch_end(self, outputs):
        # for logging
        res = dict(
            train_error = self.train_error_meter.value(),
            step = self.current_epoch, # tell Lightning logger to use epoch as step (x axis)
            epoch_monitor = torch.tensor(self.current_epoch), # needed to save only last epoch checkpoint
        )
        # reset meter
        self.train_error_meter.reset()
        return dict(**res, log=res)

    def validation_step(self, batch, batch_idx):
        # sanity check
        assert self.hparams.split == 'val'
        # forward pass
        x, y = batch
        y_hat = self.forward(x)
        # loss and performance
        res = dict(
            val_loss_batches = self.loss(y_hat, y),
        )
        self.eval_error_meter.add(y_hat, y)
        return dict(log=res, **res)

    def validation_epoch_end(self, outputs):
        # for logging
        avg_loss = torch.stack([x['val_loss_batches'] for x in outputs]).mean()
        res = dict(
            val_loss = avg_loss,
            val_error = self.eval_error_meter.value(),
            epoch_monitor = torch.tensor(self.current_epoch), # needed to save only last epoch checkpoint
            step = self.current_epoch # tell Lightning logger to use epoch as step (x axis)
        )
        if self.lr_now is not None:
            res['lr_epoch'] = torch.tensor(self.lr_now) # this lags one iteration
        self.eval_error_meter.reset()

        return dict(**res, log=res)

    def test_step(self, batch, batch_idx):
        # sanity check
        assert self.hparams.split == 'test'
        # forward pass
        x, y = batch
        y_hat = self(x)
        # performance
        self.eval_error_meter.add(y_hat, y)
        return {}

    def test_epoch_end(self, outputs):
        # for logging
        test_error = self.eval_error_meter.value()
        return {'test_error': test_error}

    def configure_optimizers(self):
        # return optimizers and learning_rate schedulers
        # SGD with momentum and wdecay
        optimizer = torch.optim.SGD(self.parameters(), 
                lr=self.hparams.lr, 
                momentum=self.hparams.momentum,
                weight_decay=self.hparams.wdecay,
        )

        # step function that degrades lr at 75% max_epochs
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer, milestones=[self.hparams.max_epochs//4 * 3], gamma=0.1
        )
        return [optimizer], [scheduler]

    def optimizer_step(
            self,
            epoch: int,
            batch_idx: int,
            optimizer: Optimizer,
            optimizer_idx: int,
            second_order_closure: Optional[Callable] = None,
            on_tpu: bool = False,
            using_native_amp: bool = False,
            using_lbfgs: bool = False,
    ) -> None:
        '''Add storing last lr to model'''
        super().optimizer_step(
            epoch=epoch,
            batch_idx=batch_idx,
            optimizer=optimizer,
            optimizer_idx=optimizer_idx,
            second_order_closure=second_order_closure,
            on_tpu = on_tpu,
            using_native_amp = using_native_amp,
            using_lbfgs = using_lbfgs,
        )
        assert optimizer_idx == 0
        # used to record learning rate in tensorboard
        self.lr_now = max([group['lr'] for group in optimizer.param_groups])

    def train_dataloader(self):
        # get training dataloader for familiar split
        ds = getattr(mydatasets, self.hparams.DS)(paths=None, train=True, split=self.hparams.split,
            fam_mode='familiar', transform=self.transform_train, target_transform=None,
        )
        return DataLoader(ds, 
            batch_size=self.hparams.batch_size, num_workers=self.hparams.num_workers, 
            shuffle=True,
        )

    def val_dataloader(self):
        # get validation dataloader for familiar split
        ds = getattr(mydatasets, self.hparams.DS)(paths=None, train=False, split=self.hparams.split,
            fam_mode='familiar', transform=self.transform_eval, target_transform=None,
        )
        return DataLoader(ds, 
            batch_size=self.hparams.batch_size, num_workers=self.hparams.num_workers, 
            shuffle=False,
        )

    def test_dataloader(self):
        # OPTIONAL
        ds = getattr(mydatasets, self.hparams.DS)(paths=None, train=False, split=self.hparams.split,
            fam_mode='familiar', transform=self.transform_eval, target_transform=None,
        )
        return DataLoader(ds, 
            batch_size=self.hparams.batch_size, num_workers=self.hparams.num_workers, 
            shuffle=False,
        )


# hooks to tweak behavior of Pytorch Lightning
class RefreshPbarAtEpochEnd(pl.Callback):
    def on_epoch_start(self, trainer, pl_module):
        for x in getattr(tqdm, '_instances', []):
            x.refresh()

# hooks to tweak behavior of Pytorch Lightning
class ModelCheckpointAtEpochEnd(pl.Callback):
    def on_epoch_end(self, trainer, pl_module):
        if trainer.disable_validation:
            trainer.checkpoint_callback.on_validation_end(trainer, pl_module)


def train_once(hparams):
    '''Train one model given hyperparameter'''
    tqdm.write('Hostname: %s' % socket.gethostname())
    hparams = copy.deepcopy(hparams)

    # turn off validation for test (validation split used in training)
    if hparams.split == 'val':
        val_percent_check = 1.
    elif hparams.split == 'test':
        val_percent_check = 0.

    # checkpoints
    cwd = os.path.join(os.getcwd(), hparams.dir_format.format(**hparams))
    tqdm.write('Check for existing checkpoints ...')
    # checkpoint saving behavior
    if hparams.split == 'test':
        checkpoint_callback_options = dict(
            monitor='epoch_monitor', mode='max', save_top_k=1,
        )
    else:
        checkpoint_callback_options = dict(
            monitor='lr_epoch', mode='min', save_top_k=1,
        )
    checkpoint_callback = pl.callbacks.model_checkpoint.ModelCheckpoint(
        filepath=os.path.join(cwd, 'checkpoint_{epoch:06d}'),
        **checkpoint_callback_options
    )
    # check checkpoints for resuming
    ckpt_pattern = os.path.join(cwd, 'checkpoint_epoch=*.ckpt')
    existing_ckpt = sorted(glob.glob(ckpt_pattern))
    existing_ckpt = existing_ckpt[-1] if existing_ckpt else None
    if existing_ckpt:
        tqdm.write('Loaded checkpoint from %s' % existing_ckpt)
    else:
        tqdm.write('No checkpoint found at %s' % ckpt_pattern)
    
    # model and trainer
    tqdm.write('Create model ...')
    model = plModelWrapped(hparams)
    tqdm.write('Create trainer ...')
    torch.set_num_threads(hparams.cpus)
    trainer = pl.Trainer(
        max_epochs=hparams.max_epochs,
        gpus=hparams.gpus,
        num_nodes=1,
        val_percent_check = val_percent_check,
        early_stop_callback = False,
        checkpoint_callback = checkpoint_callback,
        callbacks = [ModelCheckpointAtEpochEnd(), RefreshPbarAtEpochEnd() ],
        resume_from_checkpoint = existing_ckpt,
        logger = pl.loggers.TensorBoardLogger( save_dir=cwd, version=0, name='lightning_logs' ),
        progress_bar_refresh_rate = 10 if hparams.pbar else 0,
        weights_summary='top',
        default_root_dir=cwd,
    )

    # training
    tqdm.write('Trainer fitting ...')
    trainer.fit(model)

    if hparams.split == 'val':
        # save the end state to avoid re-running!
        filepath = trainer.checkpoint_callback.format_checkpoint_name(
                trainer.current_epoch, trainer.callback_metrics)
        trainer.checkpoint_callback._save_model(filepath)

    # fix tensorboard permission stuff
    try:
        subprocess.check_call(['chmod', '-R', 'g+rw', trainer.logger.root_dir])
    except Exception as e:
        tqdm.write('Exception while changing tensorboard folder permission:\n%s' % e)
    
    return trainer, model

        

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train models and evaluate using several metrics'
                                                 ' using familiar and unfamiliar dataset splits')
    parser.add_argument('dset', type=str, choices=[ 'lfwp_gender', 'cat_vs_dog', 
                                                    'imnet_animal', 'voc_to_coco' ],
                        help='Experiment dataset to run')
    parser.add_argument('split', type=str, choices=[ 'val', 'test' ],
                        help='Mini-val (val) training or validation (test) training')
    parser.add_argument('--backbone', type=str, default='resnet18', choices=['resnet18', 'densenet161'],
                        help='Backbone network. For example, resnet18 or densenet161.')
    parser.add_argument('--i_runs', type=int, nargs='+', default=[0],
                        help='Indices of runs. For example, average results from 10 runs using `--i_runs {0..9}`')
    parser.add_argument('--ensemble', action='store_true',
                        help='Treat multiple i_runs as part of an ensemble. Mean of their outputs will be used to calculate the performance. If unset, they will be treated as individual models and the mean of their performances will be reported.')
    parser.add_argument('--get_calibration', action='store_true',
                        help='Get a calibration temperature by calibrating the model with the validation set.')
    parser.add_argument('--calibrate', type=float, default=None, 
                        help='Calibrate the model with the provided temperature.')
    args = parser.parse_args()

    # sanity check
    if args.get_calibration:
        assert args.calibrate is None, 'Do not provide calibration when calibrating'
        assert args.split == 'val', 'Must calibrate on the validation set'
        if len(args.i_runs) > 1:
            warnings.warn('Calibration was performed on only one run in the original experiments.')
            import pdb; pdb.set_trace()

    # hyper params
    hparams = ezdict(
        # dataset name (lfwp_gender, cat_vs_dog, etc.)
        dset_name = args.dset,
        # backbone name (resnet18 or densenet161)
        backbone = args.backbone,
        # pretrained on Places365 (only option)
        pretrained = 'Places365',
        # validation ('val') or test ('test') split
        split = args.split,
        # SGD momentum
        momentum = 0.9,
        # SGD weight decay
        wdecay = 0,
        # data loader number of workers
        num_workers = 8,
        cpus = 8,
        # number of gpus
        gpus = 1,
        # saving directory format
        dir_format = 'trained_models/{backbone}_{dset_name}_{split}_{i_run}',
        # progress bar display
        pbar = True,

        # below: to fill within for loop
        # index of run
        i_run = None,

        # below: to fill with lookup depending on other options
        # number of outputs (i.e. number of classes)
        num_outputs = None,
        # number of epochs to train
        max_epoch = None,
        # learning rate
        lr = None,
        # dataset class name
        DS = None,
        # batch size
        batch_size = None,
        # label style: multiclass (cross-entropy + integer labels + accuracy)
        #              or multilabel (binary cross-entropy + binary vector labels + mAP)
        label_style = None,
    )
    hparams.update(lookup[args.backbone][args.dset])
    SafeProbClass = evaluate.lookup.SafeProbClass[args.dset] # multiclass or multilabel?

    # potentially multiple runs
    fam_safe_pred_allruns, unf_safe_pred_allruns = [], []
    for i_run in args.i_runs:
        hparams.i_run = i_run

        # train
        trainer, model = train_once(hparams)
        del trainer
        model.zero_grad()

        # Evaluate on familiar split
        fam_logits, fam_labels = evaluate.eval_dset( model, args.dset, familiar=True )
        # Calibrate or not?
        temperature = args.calibrate
        if args.get_calibration:
            temperature = get_calibration( fam_logits, fam_labels, 
                SafeProbClass=SafeProbClass, grid=np.linspace(0.02,3,150).tolist() )
            print('*'*80)
            print('Calibration: temperature = %.2f' % temperature)
            print('*'*80)
        if temperature is not None: 
            # temperature from FAMILIAR only
            fam_logits = fam_logits/temperature
        # Use numerically safe probability wrapper
        fam_safe_pred = SafeProbClass.from_logits(fam_logits.numpy())

        # Evaluate on unfamiliar split
        unf_logits, unf_labels = evaluate.eval_dset( model, args.dset, familiar=False )
        # Calibrate or not?
        if temperature is not None: 
            # temperature from FAMILIAR only
            unf_logits = unf_logits/temperature
        # Use numerically safe probability wrapper
        unf_safe_pred = SafeProbClass.from_logits(unf_logits.numpy())

        fam_safe_pred_allruns.append(fam_safe_pred)
        unf_safe_pred_allruns.append(unf_safe_pred)

    # aggregate outputs and compute performance
    if args.ensemble:
        # Evaluate a number of criteria on mean prediction
        fam_res = evaluate.ensemble_runs_evaulate( fam_safe_pred_allruns, fam_labels )
        unf_res = evaluate.ensemble_runs_evaulate( unf_safe_pred_allruns, unf_labels )
    else:
        # Evaluate a number of criteria on each model
        fam_res = evaluate.multiple_runs_evaulate( fam_safe_pred_allruns, fam_labels )
        unf_res = evaluate.multiple_runs_evaulate( unf_safe_pred_allruns, unf_labels )

    # print mean of resulting criteria values
    print('Runs: %s %s performance' % (args.i_runs, 'ensemble' if args.ensemble else 'mean'))
    print('%s:' % args.dset)
    print('    familiar:')
    for k in fam_res:
        print('        %s: %.6f' % (k, fam_res[k]))
    print('    unfamiliar:')
    for k in unf_res:
        print('        %s: %.6f' % (k, unf_res[k]))
