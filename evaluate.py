import os
import sys
import scipy
import itertools
import collections
import pickle
import glob
import argparse
import contextlib
try:
    from tqdm.auto import tqdm
except:
    from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets, models, transforms
from torch.autograd import Variable
from easydict import EasyDict as ezdict

from conf_eval.utils import SafeProbsMC,SafeProbsML
from conf_eval import datasets as mydatasets
from conf_eval.paths import paths
from conf_eval.training import change_last_layer


# Meta information on four datasets
lookup = ezdict(
    # Number of class
    num_outputs = dict(
        lfwp_gender  = 2,
        cat_vs_dog   = 2,
        imnet_animal = 4,
        voc_to_coco  = 20,
    ),
    # Release model files
    release_model_file = dict(
        lfwp_gender  = 'release_models/lfwp_gender/net.pkl',
        cat_vs_dog   = 'release_models/cat_vs_dog/net.pkl',
        imnet_animal = 'release_models/imnet_animal/net.pkl',
        voc_to_coco  = 'release_models/voc_to_coco/net.pkl',
    ),
    # Numerically-safe probability computation class
    # MC for softmax logit in multi-class, ML for sigmoid logit in multi-label
    SafeProbClass = dict(
        lfwp_gender  = SafeProbsMC,
        cat_vs_dog   = SafeProbsMC,
        imnet_animal = SafeProbsMC,
        voc_to_coco  = SafeProbsML,
    ),
    # Dataset class / function handles
    dataset = dict(
        lfwp_gender  = mydatasets.LFWpGenderFamUnfDataset,
        cat_vs_dog   = mydatasets.PetsFamUnfDataset,
        imnet_animal = mydatasets.ImageNetFamUnfDataset,
        voc_to_coco  = mydatasets.VOCCocoFamUnfDataset, 
    ),
    # Calibration temperature for baseline ResNet18 models
    # (obtained on a familiar data validation split)
    calibrate_T_baseline_resnet18 = dict(
        lfwp_gender  = 1.60,
        cat_vs_dog   = 1.90,
        imnet_animal = 1.68,
        voc_to_coco  = 1.02,
    )
)

def load_model(dset_name, save_file):
    '''Load a (ResNet18) model given save file and dataset name (for knowing #output channels)'''
    model = models.resnet18(pretrained=False)
    lln = 'fc'
    num_outputs = lookup.num_outputs[dset_name]
    # Forcefully redo last layer before actually loading weights
    change_last_layer( model, lln, num_outputs=num_outputs )

    if not os.path.isfile(save_file):
        raise RuntimeError('Checkpoint does not exist: {}'.format(save_file))
    nload = torch.load( save_file )
    model.load_state_dict(nload['model'])
    # Also in the released models:
    # optimizer_state = nload['state']
    # train_log = nload['other']
    return model

def eval_dset( model, dset_name, familiar ):
    '''Given a dataset (by name), evaluate a model and obtain numpy predictions and ground truths.
    Input: familiar: True for evaluating on the familiar split, and False for unfamiliar
    '''
    # Common ImageNet evaluation processing
    tf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Dataset
    DS = lookup.dataset[dset_name]
    ds = DS(paths=paths, train=False, split='test',
        fam_mode='familiar' if familiar else 'unfamiliar', 
        transform=tf, target_transform=None,
    )
    # Data loader
    dl = torch.utils.data.DataLoader(ds, 
        batch_size=32,
        shuffle=False, 
        num_workers=8)
    return eval_dsetloader( model, dl )

def eval_dsetloader( model, dset_loader ):
    '''Evaluate a model using data from a given dataset loader and obtain numpy predictions and ground truths.'''
    model.cuda()
    model.eval()

    # First: collect all the logits and labels for the validation set
    logits_list = []
    labels_list = []
    # Compatibility for "no_grad" (on fail, use dummy context)
    _compat = not hasattr(torch, 'no_grad')
    with contextlib.suppress() if _compat else torch.no_grad():
        for input, label in tqdm(dset_loader, desc='Eval batches'):
            input, label = [ Variable(x, volatile=(_compat)).cuda() for x in [input, label] ]
            # Evaluate and add result to list
            logits = model(input)
            assert logits.data.ndimension() == 2
            logits_list.append(logits.detach().data.cpu())
            labels_list.append(label.detach().data.cpu())
    # Result: N_samples x N_classes torch cpu tensors
    logits = torch.cat(logits_list)
    labels = torch.cat(labels_list)
    return logits, labels

def multiple_runs_evaulate( safe_pred_allruns, labels ):
    '''Evaluate logits from multiple runs and averaging the performance.'''
    res_allruns = [ x.dict_performance(labels.numpy()) for x in safe_pred_allruns ]
    res = ezdict()
    for k in res_allruns[0].keys():
        res[k] = np.mean([ x[k] for x in res_allruns ])
    return res

def ensemble_runs_evaulate( safe_pred_allruns, labels ):
    '''Evaluate ensemble performance by averaging probabilities from multiple ensemble members.'''
    safe_pred_mean = type(safe_pred_allruns[0]).stack(safe_pred_allruns, axis=0).mean(axis=0)
    return safe_pred_mean.dict_performance(labels.numpy())

def main(dset_name, calibrate=False, model=None):
    '''Script to evaluate a released model on a dataset familiar / unfamiliar pair.
    calibrate: to use temperature scaling or not.
    '''
    # Load model
    if model is None:
        save_file = lookup.release_model_file[dset_name] # todo: change to own
        model = load_model(dset_name, save_file)

    # Evaluate on familiar split
    logits, labels = eval_dset( model, dset_name, familiar=True )
    # Calibrate or not?
    logits = logits/lookup.calibrate_T_baseline_resnet18[dset_name] if calibrate else logits
    # Use numerically safe probability wrapper
    safe_pred = lookup.SafeProbClass[dset_name].from_logits(logits.numpy())
    # Evaluate a number of criteria 
    fam_res = safe_pred.dict_performance(labels.numpy())

    # Evaluate on unfamiliar split
    logits, labels = eval_dset( model, dset_name, familiar=False )
    # Calibrate or not?
    logits = logits/lookup.calibrate_T_baseline_resnet18[dset_name] if calibrate else logits
    # Use numerically safe probability wrapper
    safe_pred = lookup.SafeProbClass[dset_name].from_logits(logits.numpy())
    # Evaluate a number of criteria 
    unf_res = safe_pred.dict_performance(labels.numpy())

    # Return criteria values
    return fam_res, unf_res


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate released models using several metrics'
                                                 ' using familiar and unfamiliar dataset splits')
    parser.add_argument('dset', type=str, choices=[ 'lfwp_gender', 'cat_vs_dog', 
                                                    'imnet_animal', 'voc_to_coco' ],
                        help='Experiment dataset to run')
    parser.add_argument('--calibrate', action='store_true',
                        help='Calibrate the model or not')
    args = parser.parse_args()

    # run evaluation
    fam_res, unf_res = main(args.dset, calibrate=args.calibrate)

    # print resulting criteria values
    print('%s:' % args.dset)
    print('    familiar:')
    for k in fam_res:
        print('        %s: %.6f' % (k, fam_res[k]))
    print('    unfamiliar:')
    for k in unf_res:
        print('        %s: %.6f' % (k, unf_res[k]))
