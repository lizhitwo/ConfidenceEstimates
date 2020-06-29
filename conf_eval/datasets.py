import torch.utils.data as data
from torchvision.datasets.folder import ImageFolder, default_loader
from torchvision.datasets.coco import CocoCaptions, CocoDetection
from torchvision import transforms
from easydict import EasyDict as ezdict

import os
import os.path
import re
import glob
import xml.etree.ElementTree as ET
import random
import math
import scipy.io as sio
import pickle
import numpy as np
import collections
import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import h5py

from .paths import paths as default_paths
from . import utils as MU






_num_notalldogs = 20
_num_notallcats = 9

voc2coco_cls = [4, 1, 14, 8, 39, 5, 2, 15, 56, 19, 60, 16, 17, 3, 0, 58, 18, 57, 6, 62]

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def split_imgs( orig_imgs, split=0.8, seed=None, by_class=True ):
    '''Split imgs = [(path, cls), ... ] into train/val'''

    if by_class:
        n_classes = max([ x[1] for x in orig_imgs ])+1
        imgs = { k: [] for k in range(n_classes) } # [ [] for x in range(n_classes) ]
        imgs[-100] = []
        for i, img in enumerate(orig_imgs):
            assert not hasattr(img[1], '__len__')
            imgs[ img[1] ].append( (img, i) )
    else:
        imgs = {None: [ (img, i) for i, img in enumerate(orig_imgs) ]}

    # split each class into train/test
    rng = random.Random()
    rng.seed(seed)
    trainimgs = []
    valimgs = []
    for i in sorted(imgs.keys()): # range(len(imgs)):
        clsimg = imgs[i]
        random.shuffle( clsimg, random=rng.random )
        n_train = math.ceil(len(clsimg) * split)
        trainimgs.extend( clsimg[:n_train] )
        valimgs.extend( clsimg[n_train:] )

    # shuffle one last time
    random.shuffle( trainimgs, random=rng.random )
    random.shuffle( valimgs, random=rng.random )

    # get indices and imgs
    trainimgs, trainidxs = [ list(x) for x in zip(*trainimgs) ]
    valimgs  , validxs   = [ list(x) for x in zip(*  valimgs) ]
    
    return trainimgs, valimgs, trainidxs, validxs


class ImagePathsDataset(data.Dataset):
    '''Given paths, provide a dataset random access object.'''

    def __init__(self, root, imgs, classes=None, class_to_idx=None, transform=None, target_transform=None,
                 loader=default_loader):

        if len(imgs) == 0:
            raise(RuntimeError("Dataset size cannot be zero"))

        self.root = root
        self.imgs = imgs
        self.classes = classes
        self.class_to_idx = class_to_idx
        if classes is not None and class_to_idx is None:
            self.class_to_idx = {classes[i]: i for i in range(len(classes))}
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
        self.tags = ['mydatasets-api'] # duck-typing using self-proclamation

    def __getitem__(self, index):
        '''Note that pytorch's dataset loader supports returning torch.tensor, np.ndarray, number, string, a sequence of these, or a dict of these.'''

        path, target = self.imgs[index]
        img = self.loader(os.path.abspath(os.path.join(self.root,path)))
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.imgs)

    def extend_imgs( self, other ):

        # tags = ['mydatasets-api'] # duck-typing using self-proclamation
        assert isinstance( other, ImagePathsDataset ) or (hasattr(other, 'root') and hasattr(other, 'imgs'))
        relpath = os.path.relpath( other.root, start=self.root )
        relpath = '' if relpath == '.' else relpath
        oimgs = [ (os.path.join(relpath, img), l) for (img, l) in other.imgs ]
        if 'extended' not in self.tags:
            self.tags.append('extended')
        self.imgs.extend(oimgs)

def _h5d2iter(dset):
    return np.array(dset).flat
def _h5d2list(dset):
    return list(_h5d2iter(dset))
def _h5d2str(dset):
    return u''.join(chr(x) for x in np.array(dset).flat)

class LFWp(ImagePathsDataset):
    
    @staticmethod
    def _load_meta( root ):
        # get all images
        f = h5py.File( os.path.join(root, 'lfw+_labels.mat') )
        meta = dict()

        # parsing attribute names
        meta['attrs'] = [ _h5d2str(f[x]) for x in _h5d2iter(f['AttrName']) ]

        # parsing the k-fold image file names
        images = _h5d2iter(f['image_list_5fold'])
        meta['images'] = [ [ _h5d2str(f[xx]) for xx in _h5d2iter(f[x])] for x in images ]

        # parsing the labels
        labels = _h5d2iter(f['label'])
        meta['labels'] = [ np.array(f[x], dtype=int).T for x in labels ]

        return meta

    def __init__(self, paths=None, train=True, transform=None, target_transform=None,
                 split='val', loader=default_loader):

        paths = paths if paths is not None else default_paths
        root = paths.LFWproot
        root = os.path.join(root, '') # add trailing slash
        imgroot = os.path.join(root, 'lfw+_jpg24')

        meta = MU.cached( os.path.join(root, '__meta__.pkl'), type(self)._load_meta, root )
        meta = ezdict(meta)

        assert split in {'val', 'test'}
        if split == 'val':
            trainf = [0,1]
            valf   = [2]
        else:
            trainf = [0,1,2]
            valf   = [3,4]

        folds = trainf if train else valf
        names = [ xx for x in folds for xx in meta.images[x] ]
        labels = np.concatenate([ meta.labels[x] for x in folds ], axis=0)

        # initialize self
        imgs = [ nl for nl in zip(names, labels) ]
        classes = meta.attrs
        super().__init__(imgroot, imgs, classes, None, transform, target_transform, loader)

imnetSynset = collections.namedtuple('imnetSynset', ['ID', 'WNID', 'words', 'gloss', 'num_children', 'children', 'wordnet_height', 'num_train_images', ])

class ImageNetNavi(object):
    def __init__( self, paths=None ):
        paths = paths if paths is not None else default_paths
        root = paths if isinstance(paths, str) else paths.ImageNetroot 
        root = os.path.join(root, '') # add trailing slash
        meta_file = os.path.join(root, 'ILSVRC2012_devkit_t12/data/meta.mat')

        meta = sio.loadmat(meta_file)
        types = (int, str, str, str, int, list, int, int)
        synsets = [ tuple( typ(x[0]) for x, typ in zip( meta['synsets'][i][0], types ) ) for i in range(len(meta['synsets'])) ]
        synsets = [ imnetSynset(*x) for x in synsets ]
        self.synsets = synsets
        self.id2synset = { x.ID: x for x in synsets }
        self.wnid2synset = { x.WNID: x for x in synsets }
        self.id2parents = {}
        for x in synsets:
            if x.ID not in self.id2parents:
                self.id2parents[ x.ID ] = []
            for ch in x.children:
                if ch in self.id2parents:
                    self.id2parents[ ch ].append( x.ID )
                else:
                    self.id2parents[ ch ] = [x.ID]

    def __iter__(self):
        return iter(self.synsets)
        
    def __getitem__( self, ID ):
        if isinstance( ID, str ):
            synset = self.wnid2synset[ID]
        else:
            synset = self.id2synset[ID]
        return synset

    def traverse_DFS( self, ID, direction='children' ):
        assert direction in ['parents', 'children']
        direction = dict(
            children=lambda key: self[key].children,
            parents =lambda key: self.id2parents[key],
        )[direction]

        if isinstance(ID, str):
            ID = self[ID].ID
        root = ID
        stack = [root]
        traverse = []
        traverse_set = set()
        while len(stack) != 0:
            key = stack.pop()
            if key in traverse_set:
                continue
            traverse.append(key)
            traverse_set.update([key])
            new_children = [ x for x in direction(key) ]
            stack.extend(new_children)
        return traverse

    def get_leaf( self, ID ):
        trav = self.traverse_DFS(ID)
        return sorted(set([ x for x in trav if self[x].num_children == 0 ]))

class ImageNet(ImagePathsDataset):
    
    def _getMeta(self):
        if hasattr(self, 'classes'):
            return
        self._cls_navi = ImageNetNavi( self.root )
        synset_gloss = [ (x[0],x[1],x[2],x[3]) for x in self._cls_navi ]
        self.synset_gloss = { WNID: (v1, v2, ID) for ID, WNID, v1, v2 in synset_gloss }
        self.classes = sorted([ os.path.basename(x) for x in glob.glob(
                                    os.path.join(self.root, 'ILSVRC2012_img_train/n*')) ])
        
    def _getTrain(self):
        cache_file = self._cache % 'train'

        def __getTrain(this):
            # ls for each subfolder
            set_dir = 'ILSVRC2012_img_train'
            imgs = []
            for i, cls in enumerate(this.classes):
                cls_files = [ os.path.basename(x) for x in glob.glob(
                                    os.path.join(this.root, set_dir, cls, '*.*')) ]
                # add the class id
                cls_files = [ (os.path.join(set_dir, cls, x), i) for x in sorted(cls_files)
                                    if is_image_file(x) ]
                imgs.extend(cls_files)
            return imgs

        return MU.cached( cache_file, __getTrain, self )
        
        
    def _getVal(self):
        cache_file = self._cache % 'val'

        def __getVal(this):
            set_dir = 'ILSVRC2012_img_val'
            img_files = [ os.path.basename(x) for x in glob.glob(
                                os.path.join(this.root, set_dir, '*.*')) ]
            img_files = [ os.path.join(set_dir, x) for x in sorted(img_files)
                                if is_image_file(x) ]

            # get groundtruth and translate to new order
            with open(os.path.join(this.root, 'ILSVRC2012_devkit_t12/data/ILSVRC2012_validation_ground_truth.txt'), 'rt') as f:
                img_classes = [ int(x) for x in f.readlines() ]

            synset_id_old2str = { vid: k for k, (v1, v2, vid) in this.synset_gloss.items() }
            synset_id_str2new = { x: i for i, x in enumerate(this.classes) }
            img_classes = [ synset_id_str2new[ synset_id_old2str[x] ] for x in img_classes ]

            imgs = list(zip(img_files, img_classes))

            return imgs

        return MU.cached( cache_file, __getVal, self )


    def _getTest(self):
        cache_file = self._cache % 'test'
        
        def __getTest(this):
            # get file names
            set_dir = 'ILSVRC2012_img_test'
            img_files = [ os.path.basename(x) for x in glob.glob(
                                os.path.join(this.root, set_dir, '*.*')) ]
            imgs = [ (os.path.join(set_dir, x), 0) for x in sorted(img_files)
                                if is_image_file(x) ]

            return imgs

        return MU.cached( cache_file, __getTest, self )
        
        
    def __init__(self, paths=None, train=True, split='val',
                 transform=None, target_transform=None, loader=default_loader):
        
        paths = paths if paths is not None else default_paths
        root = paths.ImageNetroot
        root = os.path.join(root, '') # add trailing slash

        self.root = root
        self._cache = os.path.join(root, '__cache_%s__.pkl')
        self._getMeta()

        if split == 'val':
            if train:
                imgs = self._getTrain()
            else:
                imgs = self._getVal()
        elif split == 'test':
            if train:
                imgs = self._getTrain() + self._getVal()
            else:
                imgs = self._getTest()
        else:
            raise Exception('Split not recoginzed: "val" or "test"')

        classes = [ x + ': ' + self.synset_gloss[x][0] for x in self.classes ]
        super(ImageNet, self).__init__(self.root, imgs, classes, None, transform, target_transform, loader)

class VOC(ImagePathsDataset):

    def __init__(self, paths=None, train=True, split='val', transform=None, target_transform=None,
                 loader=default_loader):
        
        paths = paths if paths is not None else default_paths
        root = paths.VOCroot
        root = os.path.join(root, '') # add trailing slash

        assert split in {'val', 'test'}
        if split == 'val':
            phase = 'train' if train else 'val'
        else:
            phase = 'trainval' if train else 'test'

        meta = os.path.join(root, 'ImageSets/Main/') 
        # read in the image file names
        with open(os.path.join(meta, phase + '.txt'), 'rt') as f:
            lines = f.readlines()
            names = [ line.rstrip() for line in lines ]

        txtfiles = os.path.join(meta, '*_%s.txt' % phase)
        txtfiles = sorted( glob.glob(txtfiles) )
        classes = [ os.path.basename(x).split('_')[0] for x in txtfiles ]

        exist = [None] * len(classes)
        lbldict = {1:1, -1:0, 0:-100} # translate the notation of pos/neg/ignore
        for i_cls, cls in enumerate(classes):
            with open(os.path.join(meta, '%s_%s.txt' % (cls, phase)), 'rt') as f:
                lines = f.readlines()
                exist[i_cls] = [ lbldict[int(line.split()[1])] for line in lines ] # no need to .rstrip()

        labels = np.array(exist, dtype=np.float32).T
            

        imgs = [ ( name+'.jpg', label ) for name, label in zip(names, labels) ]
        imgroot = os.path.join(root, 'JPEGImages', '')
        super(VOC, self).__init__(imgroot, imgs, classes, None, transform, target_transform, loader)

class coco_anno2existence( object ):
    '''Transform class for parsing anno into "c class exists in anno" form'''
    def __init__(self, coco, clsCoco):
        # self.coco = coco
        self.n_cls = len(coco.cats)
        assert type(clsCoco) == list
        self.cls2clsCoco = clsCoco
        self.clsCoco2cls = { v:k for k,v in enumerate(clsCoco) }

    def __call__(self, annos):
        label = np.zeros(self.n_cls, dtype=np.float32)
        # with test data or img w/o label, just return all zeros
        if len(annos) == 0:
            return label

        # parse annos
        anno_classes = [ self.clsCoco2cls[ x['category_id'] ] for x in annos ]
        difficults = [ x['area'] < 50 for x in annos ] # I have no idea what to do

        # if there are only difficult objects, then difficult existence
        # otherwise normal existence
        anno_classes = list(zip(anno_classes, difficults))
        label[[ c for c,d in anno_classes if d ]] = -100
        label[[ c for c,d in anno_classes if not d ]] = 1
        return label


def CocoExistence( paths=None, train=True, split='val',
                   transform=None, target_transform=None, loader=default_loader ):
    '''MSCOCO Dataset, but treat as multi-label problem to classify existence of classes'''

    paths = paths if paths is not None else default_paths

    # deal with splits
    assert split in {'val', 'test'}
    if split == 'val':
        phase = 'train2014' if train else 'val2014'
        annFile = 'instances_%s.json' % phase
    else:
        phase = 'trainval' if train else 'test2015'
        annFile = 'image_info_%s.json' % phase
    if phase == 'trainval':
        ret_ = [
            CocoExistence( paths, True,  'val', transform, target_transform ),
            CocoExistence( paths, False, 'val', transform, target_transform ),
        ]
        ret = ret_[0]
        # Do not use data.ConcatDataset(ret_)
        ret.extend_imgs(ret_[1])
        return ret
    
    root = os.path.join(paths.Cocoroot, phase)

    def __cached_fn(annFile, root):
        # get dataset object
        annFile = os.path.join(paths.Cocoroot, 'annotations', annFile)

        with MU.nostdout(on=True):
            ret = CocoDetection(root, annFile, transform, target_transform)
        coco = ret.coco
        
        # add label parsing
        clsCoco = sorted(coco.cats.keys())
        ttf = coco_anno2existence( coco, clsCoco )
        ret.classes = ([ '{:s}:{:2d}'.format(coco.cats[x]['name'], x) for x in clsCoco ])
        # also provide labels for convenience
        # This is redundant code. No idea how to use CocoDetection without loading image...
        ret.imgs = [ 
            ( coco.loadImgs(x)[0]['file_name'], 
              ttf( coco.loadAnns( coco.getAnnIds(imgIds=x) ) )
            ) for x in sorted(ret.ids) ]
        return dict(imgs=ret.imgs, classes=ret.classes)

    cache_file = os.path.join(paths.Cocoroot, '__cache_{split}split_{train}__.pkl'.format(
        split=split, train='train' if train else 'eval'
    ))
    ret = ezdict(MU.cached( cache_file, __cached_fn, annFile, root ))

    # cast as ImagePathsDataset for uniform interfacing
    ret = ImagePathsDataset(root, ret.imgs, classes=ret.classes, class_to_idx=None, transform=transform, target_transform=target_transform, loader=loader)

    return ret

def LFWpGenderFamUnfDataset(paths=None, train=True, split='val',
        fam_mode='familiar', transform=None, target_transform=None, loader=default_loader
        ):
    assert fam_mode in {'familiar', 'unfamiliar'}
    assert split in {'val', 'test'}

    # Obtain original ImageNet
    lfwp = LFWp(paths=paths, train=train, transform=transform, 
                target_transform=target_transform, split=split, loader=loader)
    
    keep_lbl = dict( 
        familiar=(lambda x: not x),
        unfamiliar =(lambda x: x),
    ) [fam_mode] # input: isunfamiliar

    # get original label, age group, and isunfamiliar
    labels_ = np.stack([l for x,l in lfwp.imgs])
    labels = labels_[:,1].tolist() # Gender
    isunfamiliar = (np.logical_or(labels_[:,0]<18, labels_[:,0]>=60)).tolist() # By age

    lfwp.imgs = [ (x,lbl) for (x,_),isf,lbl in zip(lfwp.imgs, isunfamiliar, labels) if keep_lbl(isf) ]
    
    lfwp.classes = ['female', 'male']
    lfwp.class_to_idx = {lfwp.classes[i]: i for i in range(len(lfwp.classes))}

    return lfwp
            
class PetsFamUnfDataset(ImagePathsDataset):
    
    def __init__(self, paths=None, train=True, transform=None, target_transform=None,
                 split='val', fam_mode='familiar',
                 loader=default_loader):

        assert fam_mode in {'familiar', 'unfamiliar'}
        assert split in {'val', 'test', 'noval'}
        paths = paths if paths is not None else default_paths
        root = paths.OIIITPetsroot
        root = os.path.join(root, '') # add trailing slash
        imgroot = os.path.join(root, 'images')

        # get all images
        assert split in {'val', 'test'}
        if split == 'val':
            metafile = 'train' if train else 'val'
        else:
            metafile = 'trainval' if train else 'test'

        meta = os.path.join(root, 'annotations') 
        # read in the image file names
        with open(os.path.join(meta, metafile + '.txt'), 'rt') as f:
            lines = f.readlines()
            lines = [ line.rstrip().split() for line in lines if not line.startswith('#') ]

        # get labels for images
        all_names, classids, species, breeds = \
            zip(*[ (x[0],) + tuple(int(xi) for xi in x[1:]) for x in lines])
        assert set(species) == set([1,2])

        # filter cat / dog's familiar / unfamliar
        # note: read data are all 1-based not 0-based
        isfam  = [ ( l<=_num_notallcats if s==1 else l<=_num_notalldogs) for l, s in zip(breeds  , species) ]
        keep   = isfam if fam_mode == 'familiar' else [ (not x) for x in isfam ]
        names  = [ n                  for n, k in zip(all_names, keep) if k ]
        labels = [ (1 if s==1 else 0) for s, k in zip(species  , keep) if k ]

        # initialize self
        imgs = [ (n + '.jpg', l) for n,l in zip(names, labels) ]
        imgs = [ (n + '.jpg', l) for n,l in zip(names, labels) if l==1 ] \
             + [ (n + '.jpg', l) for n,l in zip(names, labels) if l==0 ]
        classes = { l:n for n,l in zip(names, labels) }
        classes = [ '_'.join(classes[x].split('_')[:-1]) for x in range(len(classes)) ]

        super().__init__(imgroot, imgs, classes, None, transform, target_transform, loader)

        self.classes = ['dog', 'cat']
        self.class_to_idx = {self.classes[i]: i for i in range(len(self.classes))}

def ImageNetFamUnfDataset(paths=None, train=True, split='val',
        fam_mode='familiar', 
        transform=None, target_transform=None, loader=default_loader
        ):
    assert fam_mode in {'familiar', 'unfamiliar'}
    assert split in {'val', 'test'}

    # deal with weird train/val/test split (because we do not have labels for test)
    if split == 'val':
        # get the training set of train/val split, and split afterwards
        imnet_split = 'val'
        imnet_train = True
        post_split = True
    elif split == 'test':
        # get the corresponding set of train/val split. No neet to split further
        imnet_split = 'val'
        imnet_train = train
        post_split = False

    # Obtain original ImageNet
    seed = 1234
    getunfamiliar = dict(familiar=False, unfamiliar=True) [ fam_mode ]
    imnet = ImageNet(paths=paths, train=imnet_train, transform=transform, 
                     target_transform=target_transform, split=imnet_split, loader=loader)
    
    # Process ImageNet to get super-classes
    navi = imnet._cls_navi
    classes = ['mammal', 'bird', 'herptile', 'fish']
    classnodes = [ [1204], [1274], [1322, 1358], [1365] ] # root of the classes
    n_classes = len(classnodes)
    for i in range(n_classes):
        classnodes[i] = sorted([ y for x in classnodes[i] for y in navi.get_leaf(x) ])
    assert all([ len(cn) == len(set(cn)) for cn in classnodes ])

    # Split to class/ignore and familiar/unfamiliar
    ID2cls4 = MU.defaultdict(-100)
    ID2unfm = MU.defaultdict(True)
    for i_cls, classnode in enumerate(classnodes):
        cutoff = (1 + len(classnode))//2
        for ID in classnode[:cutoff]:
            ID2cls4[ID] = i_cls
            ID2unfm[ID] = False
        for ID in classnode[cutoff:]:
            ID2cls4[ID] = i_cls
            ID2unfm[ID] = True
    clsind2ID = [ navi[x.split(':')[0]].ID for x in imnet.classes ]

    # Apply the filter
    imgs = [ [] for x in range(n_classes) ]
    for img,cls in imnet.imgs:
        assert not hasattr(cls, '__len__')
        ID = clsind2ID[cls]
        isunfamiliar = ID2unfm[ID]
        cls4 = ID2cls4[ID]
        if cls4 != -100 and isunfamiliar == getunfamiliar:
            imgs[cls4].append( (img, cls4) )

    # Subsample to taste
    imnet.imgs = []
    max_cls_samples = min( min( len(x) for x in imgs ), 1000 )
    rng = random.Random()
    rng.seed(seed)
    for i in range(n_classes):
        clsimg = imgs[i]
        random.shuffle( clsimg, random=rng.random )
        imnet.imgs.extend( clsimg[:max_cls_samples] )
        
    # Subsample for validation split
    if post_split:
        ret = split_imgs( imnet.imgs, split=0.8, seed=seed, by_class=True )
        if train:
            imgs_, idxs_ = ret[0], ret[2]
        else:
            imgs_, idxs_ = ret[1], ret[3]
        imnet.imgs = imgs_

    # other metadata
    imnet.classes = classes
    imnet.class_to_idx = {classes[i]: i for i in range(len(classes))}

    return imnet

def VOCCocoFamUnfDataset(paths=None, train=True, split='val',
        fam_mode='familiar', 
        transform=None, target_transform=None, loader=default_loader
        ):
    assert fam_mode in {'familiar', 'unfamiliar'}
    assert split in {'val', 'test'}
    # Because we are trying to get NLL and remap COCO labels, we cannot ever test on 
    # the actual test set of either VOC or COCO. Therefore, we use the val set for test
    # and split the train set for validation.
    # VOC: get dataset, split.
    # COCO: get dataset, split, and change labels

    # deal with weird train/val/test split (because we do not have labels for test)
    if split == 'val':
        # get the training set of train/val split, and split afterwards
        vc_split = 'val'
        vc_train = True
        post_split = True
    elif split == 'test':
        # get the corresponding set of train/val split. No neet to split further
        vc_split = 'val'
        vc_train = train
        post_split = False
    else:
        raise Exception('Split not recognized')
    seed = 1234

    # Obtain original dataset
    getunfamiliar = dict(familiar=False, unfamiliar=True) [ fam_mode ]
    if getunfamiliar:
        dset = CocoExistence(
            paths=paths, train=vc_train, split=vc_split, 
            transform=transform, target_transform=target_transform, loader=loader
        )
    else:
        dset = VOC(
            paths=paths, train=vc_train, split=vc_split, 
            transform=transform, target_transform=target_transform, loader=loader
        )

    # Split for validation
    if post_split:
        # ML labels can't split by class
        ret = split_imgs( dset.imgs, split=0.8, seed=seed, by_class=False ) 
        if train:
            imgs_, idxs_ = ret[0], ret[2]
        else:
            imgs_, idxs_ = ret[1], ret[3]
        dset.imgs = imgs_
    
    # COCO change labels
    if getunfamiliar:
        _voc_ = VOC() # get the train set and throw away everything but the metadata...
        dset.classes = _voc_.classes
        # map COCO labels to VOC labels
        dset.imgs = [ (x, lbl[voc2coco_cls]) for x,lbl in dset.imgs ]
        # import pdb; pdb.set_trace()

    dset.class_to_idx = {dset.classes[i]: i for i in range(len(dset.classes))}

    return dset
