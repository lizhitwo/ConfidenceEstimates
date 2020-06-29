import warnings
import numpy as np
from sklearn.metrics import precision_recall_curve
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from sklearn.metrics.base import _average_binary_score



def VOC_prec_recall_curve(y_true, y_score, sample_weight=None):
    '''The unstable version by VOC people.
    This function heavily copies from scikit-learn's stable code.
    Licence:
    # Authors: Alexandre Gramfort <alexandre.gramfort@inria.fr>
    #          Mathieu Blondel <mathieu@mblondel.org>
    #          Olivier Grisel <olivier.grisel@ensta.org>
    #          Arnaud Joly <a.joly@ulg.ac.be>
    #          Jochen Wersdorfer <jochen@wersdoerfer.de>
    #          Lars Buitinck
    #          Joel Nothman <joel.nothman@gmail.com>
    #          Noel Dawe <noel@dawe.me>
    # License: BSD 3 clause
    '''

    from sklearn.utils import assert_all_finite, check_consistent_length, column_or_1d
    from sklearn.utils.extmath import stable_cumsum

    check_consistent_length(y_true, y_score)
    y_true = column_or_1d(y_true)
    y_score = column_or_1d(y_score)
    assert_all_finite(y_true)
    assert_all_finite(y_score)

    if sample_weight is not None:
        sample_weight = column_or_1d(sample_weight)

    # ensure binary classification if pos_label is not specified
    classes = np.unique(y_true)
    if not (np.array_equal(classes, [0, 1]) or
             np.array_equal(classes, [-1, 1]) or
             np.array_equal(classes, [0]) or
             np.array_equal(classes, [-1]) or
             np.array_equal(classes, [1])):
        raise ValueError("Data is not binary and pos_label is not specified")
    else:
        pos_label = 1.

    # make y_true a boolean vector
    y_true = (y_true == pos_label)

    # sort scores and corresponding truth values
    # first flip for consistency with buggy MATLAB version
    y_score = y_score[::-1]
    y_true = y_true[::-1]
    desc_score_indices = np.argsort(y_score, kind="mergesort")[::-1]
    y_score = y_score[desc_score_indices]
    y_true = y_true[desc_score_indices]
    if sample_weight is not None:
        weight = sample_weight[desc_score_indices]
    else:
        weight = 1.

    # y_score typically has many tied values. Here we extract
    # the indices associated with the distinct values. We also
    # concatenate a value for the end of the curve.
    # distinct_value_indices = np.where(np.diff(y_score))[0]
    # threshold_idxs = np.r_[distinct_value_indices, y_true.size - 1]

    # VOC ignores this.
    threshold_idxs = np.r_[range(y_true.size)]

    # accumulate the true positives with decreasing threshold
    tps = stable_cumsum(y_true * weight)[threshold_idxs]
    if sample_weight is not None:
        fps = stable_cumsum(weight)[threshold_idxs] - tps
    else:
        fps = 1 + threshold_idxs - tps
    fps, tps, thresholds = fps, tps, y_score[threshold_idxs]


    # now copying from the caller
    precision = tps / (tps + fps)
    recall = tps / tps[-1]

    # stop when full recall attained
    # and reverse the outputs so recall is decreasing
    last_ind = tps.searchsorted(tps[-1])
    sl = slice(last_ind, None, -1)
    return np.r_[precision[sl], 1], np.r_[recall[sl], 0], thresholds[sl]
    



def VOC_interp_prec(precision):
    '''Although people are trashing VOC's interpolation, it is
    used for result comparability.
    See https://github.com/scikit-learn/scikit-learn/issues/4577'''

    assert precision[-1] == 1
    plast = precision[0]
    # sweep from recall=1 to recall=0, use max of current and all prev.
    newprec = precision.tolist()
    for i, p in enumerate(newprec):
        plast = max(p,plast)
        newprec[i] = plast
    return np.array(newprec)



        
def VOC_AP(y_true, y_score, sample_weight=None, strict_VOC=True):
    '''Implements VOC average precision for one class.
    Note that VOC does not try to merge entries with exact same scores, but
    scikit does. VOC's version may return different (larger but not smaller)
    values for the same samples when ordered differently.'''

    
    assert y_true.ndim == y_score.ndim == 1
    # ignore labels
    # y_valid = y_true != -100
    # y_true = y_true[y_valid]
    # y_score = y_score[y_valid]

    if strict_VOC:
        precision, recall, thresholds = VOC_prec_recall_curve(
            y_true, y_score, sample_weight=sample_weight)
        # warnings.warn("Using VOC's unstable mAP version. Use strict_VOC=False "
        #     "for a mathematically reasonable version at a cost of result comparability.")
    else:
        precision, recall, thresholds = precision_recall_curve(
            y_true, y_score, sample_weight=sample_weight)
    # now recall is 1...0, precision is ?...1
    # fill irregularities in precision
    precision = VOC_interp_prec(precision)
    # print(recall[-2::-1])
    # print(precision[-2::-1])

    return -np.sum(np.diff(recall) * np.array(precision)[:-1])



def VOC_mAP(y_trues, y_scores, sample_weight=None, ignore_index=-100):
    '''Calculate mean Average Precision (mAP) for VOC outputs.
    Assumes [n_samples, n_classes] for y_trues, y_scores.'''

    if y_trues.ndim == y_scores.ndim == 1 and y_trues.shape==y_scores.shape:
        y_trues = y_trues[:,None]
        y_scores = y_scores[:,None]
    assert y_trues.ndim==2 and y_scores.ndim==2 and y_trues.shape==y_scores.shape
    rets = []
    for x in range(y_scores.shape[1]):
        y_true, y_score = y_trues[:,x], y_scores[:,x]
        mask = y_true!=ignore_index
        y_true, y_score = y_true[mask], y_score[mask]
        ret = _average_binary_score(VOC_AP,
            y_true, y_score, average='macro',
            sample_weight=sample_weight if sample_weight is None else sample_weight[mask]
        )
        rets.append(ret)
    ret = np.mean(rets)

    return ret
