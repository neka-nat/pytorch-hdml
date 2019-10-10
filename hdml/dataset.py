import collections
import numpy as np
from sklearn.preprocessing import LabelEncoder
from fuel.datasets import H5PYDataset
from fuel.utils import find_in_data_path
from fuel.streams import DataStream
from fuel.schemes import IterationScheme, BatchSizeScheme, SequentialScheme
from .random_fixed_size_crop_mod import RandomFixedSizeCrop
import random


def get_streams(path, batch_size=50, method='triplet',
                crop_size=224):
    '''
    args:
        path (str): data file path.
        batch_size (int):
            number of examples per batch
        method (str or fuel.schemes.IterationScheme):
            batch construction method. Specify 'triplet'.
        crop_size (int or tuple of ints):
            height and width of the cropped image.
    '''

    dataset_class = H5PYDataset
    dataset_train = dataset_class(path, ['train'], load_in_memory=True)
    dataset_test = dataset_class(path, ['test'], load_in_memory=True)

    if not isinstance(crop_size, tuple):
        crop_size = (crop_size, crop_size)

    if method == 'n_pairs':
        labels = dataset_class(path, ['train'], sources=['targets'], load_in_memory=True).data_sources
        scheme = NPairLossScheme(labels, batch_size)
    elif method == 'triplet':
        labels = dataset_class(path, ['train'], sources=['targets'], load_in_memory=True).data_sources
        scheme = TripletLossScheme(labels, batch_size)
    else:
        raise ValueError("`method` must be 'n_pairs_mc' or 'clustering' "
                         "or subclass of IterationScheme.")
    stream = DataStream(dataset_train, iteration_scheme=scheme)
    stream_train = RandomFixedSizeCrop(stream, which_sources=('images',),
                                       random_lr_flip=True,
                                       window_shape=crop_size)

    stream_train_eval = RandomFixedSizeCrop(DataStream(
        dataset_train, iteration_scheme=SequentialScheme(
            dataset_train.num_examples, batch_size)),
        which_sources=('images',), center_crop=True, window_shape=crop_size)
    stream_test = RandomFixedSizeCrop(DataStream(
        dataset_test, iteration_scheme=SequentialScheme(
            dataset_test.num_examples, batch_size)),
        which_sources=('images',), center_crop=True, window_shape=crop_size)

    return stream_train, stream_train_eval, stream_test


class NPairLossScheme(BatchSizeScheme):
    def __init__(self, labels, batch_size):
        self._labels = np.array(labels).flatten()
        self._label_encoder = LabelEncoder().fit(self._labels)
        self._classes = self._label_encoder.classes_
        self.num_classes = len(self._classes)
        assert batch_size % 2 == 0, ("batch_size must be even number.")
        assert batch_size <= self.num_classes * 2, (
               "batch_size must not exceed twice the number of classes"
               "(i.e. set batch_size <= {}).".format(self.num_classes * 2))
        self.batch_size = batch_size

        self._class_to_indexes = []
        for c in self._classes:
            self._class_to_indexes.append(
                np.argwhere(self._labels == c).ravel())

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def next(self):
        anchor_indexes, positive_indexes = self._generate_indexes()
        indexes = anchor_indexes + positive_indexes
        return indexes

    def _generate_indexes(self):
        random_classes = np.random.choice(
            self.num_classes, self.batch_size // 2, False)
        anchor_indexes = []
        positive_indexes = []
        for c in random_classes:
            a, p = np.random.choice(self._class_to_indexes[c], 2, False)
            anchor_indexes.append(a)
            positive_indexes.append(p)
        return anchor_indexes, positive_indexes

    def get_request_iterator(self):
        return self


class TripletLossScheme(BatchSizeScheme):
    def __init__(self, labels, batch_size):
        self._labels = np.array(labels).flatten()
        self._label_encoder = LabelEncoder().fit(self._labels)
        self._classes = self._label_encoder.classes_
        self.num_classes = len(self._classes)
        assert batch_size % 3 == 0, ("batch_size must be 3*n.")
        self.batch_size = batch_size

        self._class_to_indexes = []
        for c in self._classes:
            self._class_to_indexes.append(
                np.argwhere(self._labels == c).ravel())

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def next(self):
        anchor_indexes, positive_indexes, negative_indexes= self._generate_indexes()
        indexes = anchor_indexes + positive_indexes + negative_indexes
        return indexes

    def _generate_indexes(self):
        random_classes = [random.sample(list(range(self.num_classes)), 2) for _ in range(self.batch_size // 3)]
        anchor_indexes = []
        positive_indexes = []
        negative_indexes = []
        for i in range(self.batch_size // 3):
            a, p = random.sample(list(self._class_to_indexes[random_classes[i][0]]), 2)
            anchor_indexes.append(a)
            positive_indexes.append(p)
            n = random.sample(list(self._class_to_indexes[random_classes[i][1]]), 1)
            negative_indexes.append(n[0])
        return anchor_indexes, positive_indexes, negative_indexes

    def get_request_iterator(self):
        return self