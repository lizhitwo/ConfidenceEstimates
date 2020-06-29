import pickle
import contextlib
import io
import os
import sys
import numpy as np
import scipy
import copy
import itertools
import collections
import warnings
import socket
from easydict import EasyDict as ezdict
from .VOC_metrics import VOC_mAP

def defaultdict(__default__, *args, **kwargs):
    '''Dictionary with default option'''
    ret = collections.defaultdict(lambda: __default__)
    if len(args):
        assert len(args) == 1 and isinstance(args[0], dict)
        ret.update(args[0])
    ret.update(kwargs)
    return ret

def cached(cache_file, Callable, *args, **kwargs):
    '''Wrapper function to load cache from cache_file if it exists, or if not,
    execute Callable with args and kwargs and store into cache.
    Supporting .pkl, .npy, and .npz'''

    # without cache file, act as a dummy wrapper
    if cache_file is None:
        return Callable(*args, **kwargs)

    # make sure parent directory exists
    os.makedirs(os.path.split(cache_file)[0], exist_ok=True)
    ext = os.path.splitext(cache_file)[1]

    if ext == '.pkl':
        # get cache
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as f:
                ret = pickle.load(f)
        else:
        # run and save cache
            ret = Callable(*args, **kwargs)

            with open(cache_file, 'wb') as f:
                pickle.dump(ret, f, protocol=pickle.HIGHEST_PROTOCOL)
    elif ext == '.npy':
        # get cache
        if os.path.exists(cache_file):
            ret = np.load( cache_file )
            if ret.dtype == np.object and ret.ndim == 0:
                ret = ret.reshape([1])[0]
        else:
        # run and save cache
            ret = Callable(*args, **kwargs)

            np.save( cache_file, ret )
    elif ext == '.npz':
        # get cache
        if os.path.exists(cache_file):
            ret = np.load( cache_file )['arr_0']
            if ret.dtype == np.object and ret.ndim == 0:
                ret = ret.reshape([1])[0]
        else:
        # run and save cache
            ret = Callable(*args, **kwargs)

            np.savez_compressed( cache_file, ret )
    else:
        raise Exception('Extension %s not supported.'%ext)

    return ret

def isnp( x ):
    '''Whether a variable is a numpy tensor'''
    return type(x).__module__ == 'numpy'

class nostdout(object):
    '''Context that suppresses stdout. Usage:
    with nostdout():
        function_call()
    '''
    def __init__(self, on=True):
        self.on = on
    def __enter__(self):
        if self.on:
            self.save_stdout = sys.stdout
            sys.stdout = open(os.devnull, "w")
    def __exit__(self, exc_type, exc_value, traceback):
        if self.on:
            sys.stdout = self.save_stdout

def to_onehot( label, n_classes, ignore_index=-100 ):
    '''Convert from multi-class labels to one-hot labels. Works on both one label or a list of labels'''
    if hasattr(label, '__len__'):
        assert label.ndim == 1
        n = len(label)
        # init with zeros
        ret = np.zeros([n, n_classes], dtype=np.float32)
        # a label with ignore_index removed
        label_valid = label.copy()
        label_valid[label_valid==ignore_index] = 0
        # each row, at the label column, set value to one
        ret[np.arange(n), label_valid] = 1
        # set entire rows with ignore_index to ignore
        ret[label==ignore_index, :] = ignore_index
    else:
        ret = np.zeros(n_classes, dtype=np.float32)
        if label == ignore_index:
            ret[:] = ignore_index
        else:
            ret[label] = 1
    return ret

def logsumexp( *args, **kwargs ):
    '''Log-sum-exp, \log \sum_c \exp(logit_c), from scipy.'''
    if hasattr(scipy.misc, 'logsumexp'):
        return scipy.misc.logsumexp( *args, **kwargs )
    else:
        return scipy.special.logsumexp( *args, **kwargs )

def logmeanexp( x, axis=None, **kwargs ):
    '''Log-mean-exp, \log ( 1/C * \sum_c \exp(logit_c) ), from scipy.'''
    if axis is None:
        logN = np.log( x.size )
        return logsumexp( x, **kwargs ) - logN

    axis = list(axis) if type(axis) is tuple else axis
    logN = np.log( np.array(x.shape)[axis] ).sum()
    return logsumexp( x, axis=axis, **kwargs ) - logN

def max_mask( x, axis=-1 ):
    '''Return a mask that masks out the maximum item along an axis.'''
    assert isinstance(axis, int)
    # argmax, keepdims
    amax = np.expand_dims(x.argmax(axis=axis), axis=axis)
    # get an array of shape 1x1x1x...x1xNx1x...x1x1, 
    # where the non-trivial dimension is at axis `axis`
    # and the values are the index, going from 0 to N-1.
    reshap = np.ones(x.ndim, dtype=int)
    reshap[axis] = x.shape[axis]
    rang = np.arange(x.shape[axis]).reshape(reshap)
    # then the max mask is just if the argmax equals the index, 
    # broadcasted to the original shape.
    return rang==amax
    
def logsumexp_nomax( x, axis=None, **kwargs ):
    '''Log-sum-exp, but with the maximum probability weight set to zero.'''
    maxmask = max_mask( x, axis=axis )
    # log(P(-)) = logsumexp_{j!=i}(xj) - logsumexp(xj)
    logits_nomax = x.copy()
    logits_nomax[maxmask] = -np.inf
    return logsumexp( logits_nomax, axis=axis, **kwargs )

def np_zipaxis( *args, axis=-1 ):
    '''Iterate over a zip of a number of numpy arrays, along an axis.
    Example: np_zipaxis( np.ones(3,5), np.zeros(3,5), axis=1 ) gives you a 
    generator of length 5, with items ( array([1,1,1]), array(0,0,0) ).'''
    inzip = [ [ xx.squeeze(axis=axis) for xx in np.split(x, x.shape[axis], axis=axis) ] for x in args ]
    return zip(*inzip)

def softmax(x, axis=-1):
    """Compute softmax values for each sets of scores (softmax logits) in numpy array x."""
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / e_x.sum(axis=axis, keepdims=True)

def sigmoid(x, axis=-1):
    '''Compute sigmoid for numpy array x as sigmoid logits.'''
    return 1/(1+np.exp(-x))

class _SafeProbs( object ):
    '''Probability and criteria computing base class, done with logits to be numerically safe in stability'''
    _check_consistent = True
    def __init__( self, logpp, logpn ):
        '''Safe probability from positive and negative log probabilities.
    logpp: log P(positive)
    logpn: log P(negative)'''
        self.logpp = logpp
        self.logpn = logpn
        if not self.isconsistent():
            import pdb; pdb.set_trace()
            assert False, 'log probabilities inconsistent: probs do not add up to 1'

    def isconsistent( self ):
        '''Consistency check that P(positive) + P(negative) == 1 +- epsilon'''
        same_size = self.logpp.shape == self.logpn.shape
        if type(self)._check_consistent:
            sums_to_one = np.allclose(np.exp(self.logpp) + np.exp(self.logpn), 1)
        else:
            sums_to_one = True
        return same_size and sums_to_one

    def tonumpy( self ):
        '''Return the probability (positive).'''
        assert self.isconsistent(), 'log probabilities inconsistent: probs do not add up to 1'
        return np.exp(self.logpp)

    np = tonumpy

    @property
    def ndim(self):
        '''API pass-through for array ndim'''
        assert self.logpp.ndim == self.logpn.ndim
        return self.logpp.ndim
    # without a @ndim.setter

    @property
    def shape(self):
        '''API pass-through for array shape'''
        assert self.logpp.shape == self.logpn.shape
        return self.logpp.shape
    # without a @shape.setter

    def _ECE( self, confidences, accuracies, n_bins=10 ):
        '''Adapted from https://github.com/gpleiss/temperature_scaling/blob/master/temperature_scaling.py'''
        assert confidences.shape == accuracies.shape
        assert confidences.ndim == 1
        sort_idx = np.argsort(confidences)
        confidences = confidences[sort_idx]
        accuracies = accuracies[sort_idx]

        n_samples = len(confidences)
        # Bins are quantiles (Naeini et al., Ovadia et al.), not equal-spaced (Guo et al.)
        # generate n_bins quantiles
        bin_boundaries = np.linspace(0, n_samples, n_bins + 1)
        bin_boundaries = np.ceil(bin_boundaries).astype(int).tolist()
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]

        ece = 0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            # Calculate |confidence - accuracy| in each bin
            # weighted using #samples in each bin
            prob_in_bin = (bin_upper - bin_lower) / n_samples
            if prob_in_bin > 0:
                accuracy_in_bin = accuracies[bin_lower:bin_upper].mean()
                avg_confidence_in_bin = confidences[bin_lower:bin_upper].mean()
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prob_in_bin

        return ece

    def _Brier( self, p, onehot, reduce=True ):
        '''Brier score, i.e. L2 distance with one-hot encoding of ground truth.'''
        assert p.shape == onehot.shape
        assert onehot.max() <= 1 and onehot.min() >= 0
        d = p - onehot
        d *= d
        if reduce:
            return d.mean()
        return d

    @classmethod
    def from_another( cls, probs ):
        '''Construct current subclass from another SafeProbs object'''
        return cls( probs.logpp, probs.logpn )

    @classmethod
    def from_probs( cls, probs, axis=None, logclamp=None ):
        '''Construct from P(positive)'''
        eps = np.finfo(probs.dtype).epsneg
        probs = probs.astype(np.float64)
        nprobs = 1-probs
        # either clamp extreme values, or throw an error if there is any extreme value.
        if logclamp is None:
            assert (np.logical_and(probs>eps, nprobs>eps).all()), 'Input probabilities out of range or very unstable'
        else:
            clamp = np.exp(logclamp)
            assert 0.1 > clamp > 0, 'logclamp too small or too large'
            probs  = np.maximum( probs, clamp)
            nprobs = np.maximum(nprobs, clamp)
        # check either P(+) or P(-) is too close to zero
        sqrteps = np.sqrt(eps)
        if not np.logical_and(probs>sqrteps, nprobs>sqrteps).all():
            warnings.warn('Input probabilities unstable')
        logpp = np.log(  probs )
        logpn = np.log( nprobs )
        return cls( logpp, logpn )

    @classmethod
    def stack( cls, problist, axis=0 ):
        '''Stack SafeProbs along an axis'''
        if len(problist) and isnp(problist[0]):
            return np.stack( problist, axis=axis )
        return cls(
            np.stack([ x.logpp for x in problist ], axis=axis),
            np.stack([ x.logpn for x in problist ], axis=axis),
        )

    def apply( self, lmbd, inplace=False ):
        '''Apply elemental function lmbd(), such as transpose and indexing, to both logpp and logpn.'''
        logpp = lmbd(self.logpp)
        logpn = lmbd(self.logpn)
        assert isnp(logpp) and isnp(logpn)
        if inplace:
            self.logpp = logpp
            self.logpn = logpn
            assert self.isconsistent()
            return
        else:
            return self.__class__( logpp, logpn )


    def mean( self, axis=None ):
        '''Take the mean of an axis'''
        return self.__class__( 
            logmeanexp(self.logpp, axis=axis), 
            logmeanexp(self.logpn, axis=axis)
        )

    def dict_performance( self, gt ):
        '''Evaluate against ground truth on a number of criteria.'''
        res = ezdict(
            # Accuracy
            # acc = self.Accuracy(gt),
            # Error
            Err = 1 - self.Accuracy(gt),
            # Negative log-likelihood
            NLL = self.Clip().NLL(gt),
            # Per-sample NLL.
            # sNLL= self.Clip().NLL(gt, reduce=False),
            # Mean Average Precision. 
            # Note that VOC's flawed version is used for better comparison.
            mAP = self.mAP(gt),
            # This is the vanilla Brier score.
            # Bri = self.Bri(gt), 
            # This is our modified Brier loss.
            Bri = np.sqrt(self.Bri(gt)), 
            # Expected Calibration Error
            ECE = self.ECE(gt),
            # Error rate among 99% confident
            E99 = self.E99(gt),
            # Entropy
            Ent = self.Ent(gt),
        )
        return res


class SafeProbsML( _SafeProbs ):
    '''Multi-label probability and criteria computing class.
    Done with logits to be numerically safe in stability'''
    @staticmethod
    def from_logits( logits, axis=None ):
        '''Constructor from sigmoid logits.'''
        logits = logits.astype(np.float64)
        # log P(+) =  log( 1 / (1 + exp(-logits)) )
        #          = -log( exp(0) + exp(-logits) )
        logpp = -np.logaddexp( -logits, 0 )
        # log P(-) =  log( exp(-logits) / (1 + exp(-logits)) )
        #          = -log( exp(0) + exp(+logits) )
        logpn = -np.logaddexp(  logits, 0 )
        assert np.allclose( np.exp(logpp), sigmoid(logits), rtol=5e-4 )
        return SafeProbsML( logpp, logpn )

    def mAP( self, y_true, axis=1 ):
        '''Mean Average Precision (mAP). Uses VOC's flawed implementation to better compare with others'''
        assert axis == 1
        assert len(self.shape) == 2
        return VOC_mAP( y_true, self.np() )

    def KLDiv( self, y_true ):
        '''KL-divergence KL(y_true||y_prob) calculation (aka relative entropy), using x-ent.
        KL = -\sum p log q + \sum p log p
           = xent(p,q) - xent(p,p)
        where p = y_true, q = y_prob.
        '''
        eps = np.finfo(y_true.dtype).epsneg
        # assert (np.logical_and(y_true>eps, y_true<1-eps).all()), 'Deterministic y_true leads to infinite KL-divergence'
        return self.Xent( y_true ) - SafeProbsML.from_probs( y_true, logclamp=-16 ).Xent( y_true )

    def E99( self, y_true, thresh=0.99 ):
        '''Error rate among confidence 99%+.'''
        # first, get elem-wise correct or not
        acc, mask = self.Accuracy( y_true, thresh=0.5, reduce=False )
        # second, get a mask for confidence>99%
        c99 = np.maximum(self.logpn, self.logpp) > np.log(thresh)
        # get error among confidence>99%
        return 1-acc[np.logical_and(mask,c99)].mean()

    def NLL( self, y_true, reduce=True ):
        '''Negative log likelihood. Same thing as cross-entropy with a one-hot vector ground truth.
        When reduce is False, result entries with y_true==-100 will be zero.'''
        return self.Xent( y_true, reduce=reduce )

    def Xent( self, p_true, reduce=True ):
        '''Cross-entropy for binary classification. xent(p,q) = -\sum p log q, where p is p_true, and q is self.
        When reduce is False, result entries with p_true==-100 will be zero.'''
        mask = p_true != -100
        p_t = p_true * mask
        p_f = (1-p_true) * mask
        assert (np.logical_and(p_t>=0, p_f>=0).all())
        ret = -(p_t * self.logpp + p_f * self.logpn)
        if reduce:
            return ret.mean()
        return ret
        
    def Ent( self, p_true, reduce=True ):
        '''Entropy. Which is just Xent with self.'''
        return self.Xent( self.tonumpy(), reduce=reduce )
        
    def Accuracy( self, y_true, thresh=0.5, reduce=True ):
        '''Accuracy for multi-label classification. Threshold probability and judge correctness.
        When reduce is False, a mask will also be returned signifying entries with p_true==-100.'''
        mask = y_true != -100
        y_true_ = y_true[mask]
        assert (np.logical_and(y_true_>=0, y_true_<=1).all())
        ret = ((y_true>thresh) == (np.exp(self.logpp)>thresh))
        if reduce:
            return ret[mask].mean()
        return ret, mask

    def ECE( self, y_true, axis=-1, n_bins=10 ):
        '''Expected Calibration Error. 
        Get n_bins equal quantiles, and calculate the difference between the
        accuracy and average confidence in each quantile.
        ECE computed among each label, then averaged (this matters on the
        don't-care ground truths).
        Confidence is the maximum probability of all classes, rather than P(+),
        even for binary classification.'''
        p = self.tonumpy()
        assert p.ndim <= 2
        assert y_true.shape == p.shape
        # confidence is max prob. of the two classes
        confidences = np.maximum(p, 1-p)
        predictions = p > 0.5
        # per-sample accuracy
        accuracies = predictions == y_true
        valids = y_true!=-100

        if p.ndim == 2:
            # calculate ECE for each class and take average
            n_cls = accuracies.shape[axis]
            eces = [ self._ECE( c[v], a[v], n_bins=n_bins ) for c, a, v in np_zipaxis( confidences, accuracies, valids, axis=axis ) ]
            return np.mean(eces)
        else:
            return self._ECE( confidences[valids], accuracies[valids], n_bins=n_bins )

    def Bri( self, y_true, axis=-1, reduce=True ):
        '''Brier score. L2 between probability and one-hot ground truth.'''
        p = self.tonumpy()
        valids = y_true!=-100
        assert reduce, 'Not implemented'
        if p.ndim == 2:
            bris = [ self._Brier( p_[v], y_[v], reduce=reduce ) for p_, y_, v in np_zipaxis( p, y_true, valids, axis=axis ) ]
            return np.mean(bris) if reduce else bris
        else:
            return self._Brier( p, y_true, reduce=reduce )

    def Clip( self, clip=0.001, inplace=False ):
        '''Clip all probs into [clip, 1-clip].'''
        logmin = np.log(clip)
        logmax = np.log(1-clip)
        assert logmin < logmax
        logpp = np.clip( self.logpp, logmin, logmax )
        logpn = np.clip( self.logpn, logmin, logmax )
        if inplace:
            self.logpp = logpp
            self.logpn = logpn
            assert self.isconsistent()
            return
        else:
            return SafeProbsML( logpp, logpn )

class SafeProbsMC( _SafeProbs ):
    '''Multi-class probability and criteria computing class.
    Done with logits to be numerically safe in stability'''
    @staticmethod
    def from_logits( logits, axis=-1 ):
        '''Constructor from softmax logits.'''
        logits = logits.astype(np.float64)
        lse = logsumexp( logits, axis=axis, keepdims=True )
        # P_i(+) = exp(x_i) / sum_j( exp(x_j) )
        # log(P_i(+)) = x_i - logsumexp_j(x_j)
        logpp = logits - lse
        # the unstable version of log(P_i(-))
        logpn = np.log( np.maximum(1 - np.exp(logpp), np.finfo(logpp.dtype).tiny) )

        # only the maximum of each probability along axis is unstable when they are ~1.
        maxmask = max_mask( logits, axis=axis )
        # P_i(-) = sum_{j!=i}( exp(x_j) ) / sum_j( exp(x_j) )
        # log(P(-)) = logsumexp_{j!=i}(x_j) - logsumexp_j(x_j)
        logpn_for_argmax = logsumexp_nomax( logits, axis=axis, keepdims=True ) - lse
        # replace the maximum probability's logP(-)
        logpn[maxmask] = 0
        logpn += logpn_for_argmax * maxmask

        assert np.allclose( np.exp(logpp), softmax(logits, axis=axis), rtol=5e-4 )
        return SafeProbsMC( logpp, logpn )

    def mAP( self, y_true, axis=1 ):
        '''Mean Average Precision (mAP). 
        Uses VOC's flawed implementation to better compare with others.'''
        assert axis == 1
        assert len(self.shape) == 2
        return VOC_mAP( to_onehot(y_true, self.shape[axis]), self.np() )

    def KLDiv( self, y_true, axis=-1 ):
        '''Multinomial (multi-class) KL-divergence KL(y_true||y_prob) calculation (aka relative entropy), using x-ent.
        KL = -\sum p log q + \sum p log p
           = xent(p,q) - xent(p,p)
        where p = y_true, q = y_prob.'''
        yt_safe = SafeProbsMC.from_probs( y_true, logclamp=-16 )
        return self.Xent( y_true, axis=axis ) - yt_safe.Xent( y_true, axis=axis )

    def E99( self, y_true, axis=-1, thresh=0.99 ):
        '''Error rate among confidence 99%+.'''
        # first, get elem-wise correct or not
        acc, mask = self.Accuracy( y_true, axis=axis, reduce=False )
        # second, get a mask for confidence>99%
        c99 = np.max(self.logpp, axis=axis) > np.log(thresh)
        assert acc.shape == c99.shape # WIP: when y_true has -100
        # get error among confidence>99%
        return 1-acc[np.logical_and(mask,c99)].mean()

    def NLL( self, y_true, weight=None, axis=-1, reduce=True ):
        '''Negative log likelihood. 
        When reduce is False, result entries with y_true==-100 will be zero.'''
        assert self.ndim == y_true.ndim + 1
        assert np.allclose(y_true, y_true.astype(int))
        axis_permute = list(range(self.ndim))
        del axis_permute[axis]
        axis_permute = [axis] + axis_permute
        valid = y_true!=-100
        logpp = self.logpp
        if not valid.all():
            assert y_true.ndim == 1
            y_true = y_true[valid]
            logpp = logpp[valid]
        logpgt = y_true.astype(int).choose(logpp.transpose(axis_permute))
        if weight is not None:
            weight = weight[y_true]
            assert np.allclose(weight.mean(), 1)
            assert logpgt.shape == weight.shape
            logpgt *= weight
        if reduce:
            return -logpgt.mean()
        return -logpgt

    def Xent( self, p_true, axis=-1, reduce=True ):
        '''Multinomial cross-entropy for multi-class classification.
        xent(p,q) = -\sum p log q, where p is p_true, and q is self.'''
        assert (p_true>=0).all()
        assert np.allclose(p_true.sum(axis=axis), 1)
        assert p_true.shape == self.shape
        ret = -(p_true * self.logpp).sum(axis=axis)
        if reduce:
            return ret.mean()
        return ret

    def Ent( self, p_true, reduce=True ):
        '''Entropy is just Xent with self.'''
        return self.Xent( self.tonumpy(), reduce=reduce )
        
    def Accuracy( self, y_true, axis=-1, thresh=0.5, reduce=True ):
        '''Accuracy for multi-label classification. Predict according to maximum of probability.
        When reduce is False, returned shape may be smaller with y_true==-100 entries deleted.'''
        assert self.logpp.ndim == 2
        valid = y_true!=-100
        logpp = self.logpp
        if not valid.all():
            assert y_true.ndim == 1
            y_true = y_true[valid]
            logpp = logpp[valid]
        assert (np.logical_and(y_true>=0, y_true<self.shape[1]).all())
        ret = (y_true == (np.argmax(logpp, axis=axis)))
        if reduce:
            return ret.mean()
        return ret, valid

    def ECE( self, y_true, axis=-1, n_bins=10, reduce=True ):
        '''Expected Calibration Error. 
        Get n_bins equal quantiles, and calculate the difference between the
        accuracy and average confidence in each quantile.
        Confidence is the maximum probability of all classes, rather than P(+).
        '''
        p = self.tonumpy()

        assert reduce
        assert y_true.ndim == 1
        confidences, predictions = p.max(axis=axis), p.argmax(axis=axis)
        assert confidences.ndim == 1
        accuracies = predictions == y_true
        valid = y_true!=-100

        return self._ECE( confidences[valid], accuracies[valid], n_bins=n_bins )

    def Bri( self, y_true, axis=-1, reduce=True ):
        '''Brier score. L2 between probability and one-hot ground truth.'''
        p = self.tonumpy()
        valid = y_true!=-100
        assert y_true.ndim == 1
        assert reduce, 'Not implemented'
        onehot = to_onehot( y_true[valid], p.shape[axis] )
        return self._Brier( p[valid], onehot, reduce=reduce )

    def Clip( self, clip=0.001, inplace=False ):
        '''Clip all probs into [clip, 1-clip] as done in the paper. 

        NOTE: This operation does not normalize the probability, and the 
        clipped probabilities do not sum up to 1 with num of classes > 2.
        We only use this for NLL, and this means we are effectively putting
        an upper bound on its unbounded value.
        '''
        logmin = np.log(clip)
        logmax = np.log(1-clip)
        assert logmin < logmax
        logpp = np.clip( self.logpp, logmin, logmax )
        logpn = np.clip( self.logpn, logmin, logmax )
        if inplace:
            self.logpp = logpp
            self.logpn = logpn
            assert self.isconsistent()
            return
        else:
            return SafeProbsMC( logpp, logpn )

