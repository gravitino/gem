import numpy as np
import collections as col
from math import ceil
import numbers

# for reproducibility hardcode random seed
np.random.seed(0)

###############################################################################
# taken from sklearn (http://scikit-learn.org/)
###############################################################################

class StratifiedShuffleSplit(object):
    """Stratified ShuffleSplit cross validation iterator

    Provides train/test indices to split data in train test sets.

    This cross-validation object is a merge of StratifiedKFold and
    ShuffleSplit, which returns stratified randomized folds. The folds
    are made by preserving the percentage of samples for each class.

    Note: like the ShuffleSplit strategy, stratified random splits
    do not guarantee that all folds will be different, although this is
    still very likely for sizeable datasets.

    Parameters
    ----------
    y : array, [n_samples]
        Labels of samples.

    n_iter : int (default 10)
        Number of re-shuffling & splitting iterations.

    test_size : float (default 0.1), int, or None
        If float, should be between 0.0 and 1.0 and represent the
        proportion of the dataset to include in the test split. If
        int, represents the absolute number of test samples. If None,
        the value is automatically set to the complement of the train size.

    train_size : float, int, or None (default is None)
        If float, should be between 0.0 and 1.0 and represent the
        proportion of the dataset to include in the train split. If
        int, represents the absolute number of train samples. If None,
        the value is automatically set to the complement of the test size.

    indices : boolean, optional (default True)
        Return train/test split as arrays of indices, rather than a boolean
        mask array. Integer indices are required when dealing with sparse
        matrices, since those cannot be indexed by boolean masks.

    random_state : int or RandomState
        Pseudo-random number generator state used for random sampling.

    Examples
    --------
    >>> from sklearn.cross_validation import StratifiedShuffleSplit
    >>> X = np.array([[1, 2], [3, 4], [1, 2], [3, 4]])
    >>> y = np.array([0, 0, 1, 1])
    >>> sss = StratifiedShuffleSplit(y, 3, test_size=0.5, random_state=0)
    >>> len(sss)
    3
    >>> print(sss)       # doctest: +ELLIPSIS
    StratifiedShuffleSplit(labels=[0 0 1 1], n_iter=3, ...)
    >>> for train_index, test_index in sss:
    ...    print("TRAIN:", train_index, "TEST:", test_index)
    ...    X_train, X_test = X[train_index], X[test_index]
    ...    y_train, y_test = y[train_index], y[test_index]
    TRAIN: [1 2] TEST: [3 0]
    TRAIN: [0 2] TEST: [1 3]
    TRAIN: [0 2] TEST: [3 1]
    """

    def __init__(self, y, n_iter=10, test_size=0.1, train_size=None,
                 indices=True, random_state=None, n_iterations=None):

        self.y = np.array(y)
        self.n = len(self.y)
        self.n_iter = n_iter
        if n_iterations is not None:  # pragma: no cover
            warnings.warn("n_iterations was renamed to n_iter for consistency"
                          " and will be removed in 0.16.")
            self.n_iter = n_iterations
        self.test_size = test_size
        self.train_size = train_size
        self.random_state = random_state
        self.indices = indices
        self.n_train, self.n_test, self.classes, self.y_indices = \
            _validate_stratified_shuffle_split(y, test_size, train_size)

    def __iter__(self):
        rng = check_random_state(self.random_state)
        cls_count = np.bincount(self.y_indices)
        p_i = cls_count / float(self.n)
        n_i = np.round(self.n_train * p_i).astype(int)
        t_i = np.minimum(cls_count - n_i,
                         np.round(self.n_test * p_i).astype(int))

        for n in range(self.n_iter):
            train = []
            test = []

            for i, cls in enumerate(self.classes):
                permutation = rng.permutation(n_i[i] + t_i[i])
                cls_i = np.where((self.y == cls))[0][permutation]

                train.extend(cls_i[:n_i[i]])
                test.extend(cls_i[n_i[i]:n_i[i] + t_i[i]])

            train = rng.permutation(train)
            test = rng.permutation(test)

            if self.indices:
                yield train, test
            else:
                train_m = np.zeros(self.n, dtype=bool)
                test_m = np.zeros(self.n, dtype=bool)
                train_m[train] = True
                test_m[test] = True

                yield train_m, test_m

    def __repr__(self):
        return ('%s(labels=%s, n_iter=%d, test_size=%s, indices=%s, '
                'random_state=%s)' % (
                    self.__class__.__name__,
                    self.y,
                    self.n_iter,
                    str(self.test_size),
                    self.indices,
                    self.random_state,
                ))

    def __len__(self):
        return self.n_iter

def check_random_state(seed):
    """Turn seed into a np.random.RandomState instance

    If seed is None, return the RandomState singleton used by np.random.
    If seed is an int, return a new RandomState instance seeded with seed.
    If seed is already a RandomState instance, return it.
    Otherwise raise ValueError.
    """
    if seed is None or seed is np.random:
        return np.random.mtrand._rand
    if isinstance(seed, (numbers.Integral, np.integer)):
        return np.random.RandomState(seed)
    if isinstance(seed, np.random.RandomState):
        return seed
    raise ValueError('%r cannot be used to seed a numpy.random.RandomState'
                     ' instance' % seed)

def _validate_shuffle_split(n, test_size, train_size):
    if test_size is None and train_size is None:
        raise ValueError(
            'test_size and train_size can not both be None')

    if test_size is not None:
        if np.asarray(test_size).dtype.kind == 'f':
            if test_size >= 1.:
                raise ValueError(
                    'test_size=%f should be smaller '
                    'than 1.0 or be an integer' % test_size)
        elif np.asarray(test_size).dtype.kind == 'i':
            if test_size >= n:
                raise ValueError(
                    'test_size=%d should be smaller '
                    'than the number of samples %d' % (test_size, n))
        else:
            raise ValueError("Invalid value for test_size: %r" % test_size)

    if train_size is not None:
        if np.asarray(train_size).dtype.kind == 'f':
            if train_size >= 1.:
                raise ValueError("train_size=%f should be smaller "
                                 "than 1.0 or be an integer" % train_size)
            elif np.asarray(test_size).dtype.kind == 'f' and \
                    train_size + test_size > 1.:
                raise ValueError('The sum of test_size and train_size = %f, '
                                 'should be smaller than 1.0. Reduce '
                                 'test_size and/or train_size.' %
                                 (train_size + test_size))
        elif np.asarray(train_size).dtype.kind == 'i':
            if train_size >= n:
                raise ValueError("train_size=%d should be smaller "
                                 "than the number of samples %d" %
                                 (train_size, n))
        else:
            raise ValueError("Invalid value for train_size: %r" % train_size)

    if np.asarray(test_size).dtype.kind == 'f':
        n_test = ceil(test_size * n)
    elif np.asarray(test_size).dtype.kind == 'i':
        n_test = float(test_size)

    if train_size is None:
        n_train = n - n_test
    else:
        if np.asarray(train_size).dtype.kind == 'f':
            n_train = floor(train_size * n)
        else:
            n_train = float(train_size)

    if test_size is None:
        n_test = n - n_train

    if n_train + n_test > n:
        raise ValueError('The sum of train_size and test_size = %d, '
                         'should be smaller than the number of '
                         'samples %d. Reduce test_size and/or '
                         'train_size.' % (n_train + n_test, n))

    return int(n_train), int(n_test)

def _validate_stratified_shuffle_split(y, test_size, train_size):
    classes, y = unique(y, return_inverse=True)
    n_cls = classes.shape[0]

    if np.min(np.bincount(y)) < 2:
        raise ValueError("The least populated class in y has only 1"
                         " member, which is too few. The minimum"
                         " number of labels for any class cannot"
                         " be less than 2.")

    n_train, n_test = _validate_shuffle_split(len(y), test_size, train_size)

    if n_train < n_cls:
        raise ValueError('The train_size = %d should be greater or '
                         'equal to the number of classes = %d' %
                         (n_train, n_cls))
    if n_test < n_cls:
        raise ValueError('The test_size = %d should be greater or '
                         'equal to the number of classes = %d' %
                         (n_test, n_cls))

    return n_train, n_test, classes, y


class StratifiedKFold(object):
    """Stratified K-Folds cross validation iterator

    Provides train/test indices to split data in train test sets.

    This cross-validation object is a variation of KFold, which
    returns stratified folds. The folds are made by preserving
    the percentage of samples for each class.

    Parameters
    ----------
    y : array-like, [n_samples]
        Samples to split in K folds.

    n_folds : int, default=3
        Number of folds.

    indices : boolean, optional (default True)
        Return train/test split as arrays of indices, rather than a boolean
        mask array. Integer indices are required when dealing with sparse
        matrices, since those cannot be indexed by boolean masks.

    Examples
    --------
    >>> from sklearn import cross_validation
    >>> X = np.array([[1, 2], [3, 4], [1, 2], [3, 4]])
    >>> y = np.array([0, 0, 1, 1])
    >>> skf = cross_validation.StratifiedKFold(y, n_folds=2)
    >>> len(skf)
    2
    >>> print(skf)
    sklearn.cross_validation.StratifiedKFold(labels=[0 0 1 1], n_folds=2)
    >>> for train_index, test_index in skf:
    ...    print("TRAIN:", train_index, "TEST:", test_index)
    ...    X_train, X_test = X[train_index], X[test_index]
    ...    y_train, y_test = y[train_index], y[test_index]
    TRAIN: [1 3] TEST: [0 2]
    TRAIN: [0 2] TEST: [1 3]

    Notes
    -----
    All the folds have size trunc(n_samples / n_folds), the last one has the
    complementary.
    """

    def __init__(self, y, n_folds=3, indices=True, k=None):
        if k is not None:  # pragma: no cover
            warnings.warn("The parameter k was renamed to n_folds and will be"
                          " removed in 0.15.", DeprecationWarning)
            n_folds = k
        y = np.asarray(y)
        n = y.shape[0]
        _validate_kfold(n_folds, n)
        _, y_sorted = unique(y, return_inverse=True)
        min_labels = np.min(np.bincount(y_sorted))
        if n_folds > min_labels:
            warnings.warn(("The least populated class in y has only %d"
                          " members, which is too few. The minimum"
                          " number of labels for any class cannot"
                          " be less than n_folds=%d."
                          % (min_labels, n_folds)), Warning)
        self.y = y
        self.n_folds = n_folds
        self.indices = indices

    def __iter__(self):
        n_folds = self.n_folds
        n = len(self.y)
        idx = np.argsort(self.y)
        if self.indices:
            ind = np.arange(n)
        for i in range(n_folds):
            test_index = np.zeros(n, dtype=np.bool)
            test_index[idx[i::n_folds]] = True
            train_index = np.logical_not(test_index)
            if self.indices:
                train_index = ind[train_index]
                test_index = ind[test_index]
            yield train_index, test_index

    def __repr__(self):
        return '%s.%s(labels=%s, n_folds=%i)' % (
            self.__class__.__module__,
            self.__class__.__name__,
            self.y,
            self.n_folds,
        )

    def __len__(self):
        return self.n_folds

def _validate_kfold(k, n_samples):
    if k <= 0:
        raise ValueError("Cannot have number of folds k below 1.")
    if k > n_samples:
        raise ValueError("Cannot have number of folds k=%d greater than"
                         " the number of samples: %d." % (k, n_samples))

def unique(ar, return_index=False, return_inverse=False):
    """A replacement for the np.unique that appeared in numpy 1.4.

    While np.unique existed long before, keyword return_inverse was
    only added in 1.4.
    """
    try:
        ar = ar.flatten()
    except AttributeError:
        if not return_inverse and not return_index:
            items = sorted(set(ar))
            return np.asarray(items)
        else:
            ar = np.asarray(ar).flatten()

    if ar.size == 0:
        if return_inverse and return_index:
            return ar, np.empty(0, np.bool), np.empty(0, np.bool)
        elif return_inverse or return_index:
            return ar, np.empty(0, np.bool)
        else:
            return ar

    if return_inverse or return_index:
        perm = ar.argsort()
        aux = ar[perm]
        flag = np.concatenate(([True], aux[1:] != aux[:-1]))
        if return_inverse:
            iflag = np.cumsum(flag) - 1
            iperm = perm.argsort()
            if return_index:
                return aux[flag], perm[flag], iflag[iperm]
            else:
                return aux[flag], iflag[iperm]
        else:
            return aux[flag], perm[flag]

    else:
        ar.sort()
        flag = np.concatenate(([True], ar[1:] != ar[:-1]))
        return ar[flag]

###############################################################################
# end sklearn (http://scikit-learn.org/)
###############################################################################


if __name__ == '__main__': 

    import UCRDatabase as ucr
    import pylab as pl
    
    # read data set from UCR database
    (testLabels, testSet), (trainLabels, trainSet)  = ucr.read(3)
    rho = float(len(testSet))/(len(trainSet)+len(testSet))
    
    
    
    # union of labels and items
    labels, items = ucr.merge(testLabels, testSet, trainLabels, trainSet)
    
    # check histograms
    sss = StratifiedShuffleSplit(labels, 15, test_size=rho, random_state=0)
    for train_index, test_index in sss:
        
        print("TRAIN:", train_index, "TEST:", test_index)
        
        pl.hist(labels)
        pl.hist(labels[train_index])
        pl.show()
    
