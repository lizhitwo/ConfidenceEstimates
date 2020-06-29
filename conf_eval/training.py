import torch
import torch.nn as nn

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

