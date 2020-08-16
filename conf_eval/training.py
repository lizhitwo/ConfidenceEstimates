import torch
import torch.nn as nn
from torch.nn import functional as F

def change_last_layer( model, last_layer_name, num_outputs, init=True):
    '''Helper function to replace last layer to match num of output classes 
    before training or loading weights.

    last_layer_name: name of last layer. "fc" for ResNet, "classifier" for DenseNet, etc.
    num_outputs: number of output classes, e.g. 20 for VOC.
    init: initialize the new last layer using glorot initialization'''
    lln = last_layer_name
    num_ftrs = getattr(model, lln).in_features
    setattr(model, lln, nn.Linear(num_ftrs, num_outputs))
    if init:
        try:
            nn.init.xavier_normal_(getattr(model, lln).weight)
        except:
            nn.init.xavier_normal(getattr(model, lln).weight)

def model_load( model_file, model ):
    '''Load a Places365 model from file'''
    model_wrapped = torch.nn.Module()
    model_wrapped.module = model
    model_dict = torch.load(model_file, map_location=lambda storage, loc: storage)
    # loaded state dict has "module." prefixes
    model_wrapped.load_state_dict(model_dict['state_dict'])

class MultiClassErrorMeter(object):
    '''Error rate meter for multi-class classification'''
    def __init__(self):
        self.reset()
    def reset(self):
        self.errors = []
    def add(self, y_hat, y):
        self.errors.append(y_hat.detach().argmax(dim=1) == y)
    def value(self):
        return 1-torch.cat(self.errors, dim=0).float().mean()

class MultiLabelErrorMeter(object):
    '''Error rate meter for multi-label classification'''
    def __init__(self):
        self.reset()
    def reset(self):
        self.errors = []
    def add(self, y_hat, y):
        mask = y != -100
        self.errors.append((y[mask]>0.5) == (y_hat[mask]>0))
    def value(self):
        return 1-torch.cat(self.errors, dim=0).float().mean()

def bce_ignore_with_logits( input, target, reduction='mean', ignore_index=-100 ):
    '''Same as F.binary_cross_entropy_with_logits, but with ignore labels (1/0/-100).'''
    mask = (target != ignore_index).type_as(input)
    return F.binary_cross_entropy_with_logits( input, target, weight=mask, reduction=reduction)

def get_calibration( fam_logits, fam_labels, SafeProbClass, grid ):
    '''Calibrate Guo et al. temperature scaling method. Dense grid search for the temperature.'''
    nlls = []
    for T in grid:
        # raise temperature
        fam_safe_pred = SafeProbClass.from_logits((fam_logits/T).numpy())
        # get negative log-likelihood
        nll = fam_safe_pred.NLL(fam_labels.numpy())
        nlls.append((nll, T))
    # get temperature that yields lowest NLL
    return min(nlls)[1]

