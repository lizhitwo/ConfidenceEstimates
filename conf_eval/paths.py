import os.path
from easydict import EasyDict as ezdict
import socket


class Paths(object):

    def __init__(self, root=None):
        # decide root based on input and machine name
        if root is None:
            hostname = socket.gethostname()
            if hostname == 'cashew':
                self.dsetroot = '/home/zhizhong/tmp_dset/'
            elif hostname.startswith('vision-'):
                self.dsetroot = '/home/zli115/tmp_dset/'
            else:
                raise Exception('Please configure your dataset root and paths in this code file')
        else:
            self.dsetroot = root
        
        # dataset root folders
        self.ImageNetroot = os.path.join(self.dsetroot, 'ILSVRC2012')
        # self.Places365root = os.path.join(self.dsetroot, 'Places365')
        self.VOCroot = os.path.join(self.dsetroot, 'VOC2012')
        self.Cocoroot = os.path.join(self.dsetroot, 'MSCOCO')
        self.OIIITPetsroot = os.path.join(self.dsetroot, 'Oxford_IIIT_Pets')
        self.LFWproot = os.path.join(self.dsetroot, 'LFW+_Release')

        # pretrained models from https://github.com/CSAILVision/places365 for training
        self.pretrainedroot = os.path.join(self.dsetroot, 'pretrained')
        self.prePlaces365 = ezdict()
        for x in ['resnet18', 'resnet50', 'densenet161']:
            self.prePlaces365[x] = os.path.join(
                self.pretrainedroot, 
                'whole_%s_places365_python36.pth.tar' % (x.lower()),
            )


paths = Paths()
