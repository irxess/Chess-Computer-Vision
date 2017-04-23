import os, collections, random
from skimage import io
from skimage.transform import rotate
import numpy as np

DATA_SET_DIR = os.path.join(os.getcwd(), '50x50')
PIECES = ['e', 'k', 'q', 'r', 'n', 'b', 'p']
COLORS = ['w', 'b', 'n']
Datasets = collections.namedtuple('Datasets', ['train', 'validation', 'test'])


class DataSet(object):

    def __init__(self,
                 images,
                 labels):
        self._num_examples = images.shape[0]
        images = images.astype(np.float32)
        images = np.multiply(images, 1.0 / 255.0)
        self._images = images
        self._labels = labels
        self._epochs_completed = 0
        self._index_in_epoch = 0

    @property
    def images(self):
        return self._images

    @property
    def labels(self):
        return self._labels

    @property
    def num_examples(self):
        return self._num_examples

    def next_batch(self, batch_size):
        start = self._index_in_epoch
        # Go to the next epoch
        if start + batch_size > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            # Get the rest examples in this epoch
            rest_num_examples = self._num_examples - start
            images_rest_part = self._images[start:self._num_examples]
            labels_rest_part = self._labels[start:self._num_examples]
            # Shuffle the data
            perm = np.arange(self._num_examples)
            np.random.shuffle(perm)
            self._images = self.images[perm]
            self._labels = self._labels[perm]
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size - rest_num_examples
            end = self._index_in_epoch
            images_new_part = self._images[start:end]
            labels_new_part = self._labels[start:end]
            return np.concatenate((images_rest_part, images_new_part), axis=0), np.concatenate(
                (labels_rest_part, labels_new_part), axis=0)
        else:
            self._index_in_epoch += batch_size
            end = self._index_in_epoch
            return self._images[start:end], self._labels[start:end]


def one_hot(index, size):
    label = np.zeros(size)
    label[index] = 1
    return label


def one_hot_index(p, c):
    if c in [0, 2]:
        return p
    else:
        return 6 + p


def expand_instance(instance):
    imgs = []
    imgs.append(instance[0])
    imgs.append(np.flip(imgs[0], 0))
    rotations = []
    for img in imgs:
        rotations.append(rotate(img, 90))
        rotations.append(rotate(img, 180))
        rotations.append(rotate(img, 270))
    imgs.extend(rotations)
    return [[img, instance[1]] for img in imgs]


def extract_data(validation_fraction, test_fraction):
    # Read data set from data directory
    data = []
    for piece in PIECES:
        p_path = os.path.join(DATA_SET_DIR, piece)
        p_white = []
        p_black = []
        p_none = []
        for filename in os.listdir(p_path):
            # Make sure file is a valid data set file
            if filename[0] not in PIECES:
                continue
            img = io.imread(os.path.join(p_path, filename))
            if filename[1] == 'w':
                p_white.append(img)
            elif filename[1] == 'b':
                p_black.append(img)
            else:
                p_none.append(img)
        random.shuffle(p_white)
        random.shuffle(p_black)
        random.shuffle(p_none)
        data.append([p_white, p_black, p_none])

    # Split data set into training, validation, and test sets
    tr = []
    vl = []
    ts = []
    p_index = 0
    for piece_data in data:
        c_index = 0
        for color_data in piece_data:
            l = len(color_data)
            vl_index = int(l*validation_fraction)
            ts_index = int(l - l*test_fraction)
            for img in color_data[:vl_index]:
                vl.append([img,
                           one_hot(one_hot_index(p_index, c_index), 13)])
            for img in color_data[vl_index:ts_index]:
                tr.append([img,
                           one_hot(one_hot_index(p_index, c_index), 13)])
            for img in color_data[ts_index:]:
                ts.append([img,
                           one_hot(one_hot_index(p_index, c_index), 13)])
            c_index += 1
        p_index += 1

    # Expand training set
    new_tr =[]
    for instance in tr:
        for derived_instance in expand_instance(instance):
            new_tr.append(derived_instance)
    tr = new_tr

    random.shuffle(tr)
    random.shuffle(vl)
    random.shuffle(ts)

    tr_imgs = np.array([instance[0] for instance in tr])
    tr_lbs = np.array([instance[1] for instance in tr])
    vl_imgs = np.array([instance[0] for instance in vl])
    vl_lbs = np.array([instance[1] for instance in vl])
    ts_imgs = np.array([instance[0] for instance in ts])
    ts_lbs = np.array([instance[1] for instance in ts])

    return {'tr_imgs': tr_imgs, 'tr_lbs': tr_lbs,
            'vl_imgs': vl_imgs, 'vl_lbs': vl_lbs,
            'ts_imgs': ts_imgs, 'ts_lbs': ts_lbs}


def read_data_sets(validation_fraction=0.0, test_fraction=0.2):
    data = extract_data(validation_fraction, test_fraction)
    train = DataSet(data['tr_imgs'], data['tr_lbs'])
    validation = DataSet(data['vl_imgs'], data['vl_lbs'])
    test = DataSet(data['ts_imgs'], data['ts_lbs'])

    return Datasets(train=train, validation=validation, test=test)
