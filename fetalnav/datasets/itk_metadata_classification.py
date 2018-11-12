# emacs: -*- coding: utf-8; mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-

"""
Dataset loading for ITK images and their respective labels within their metadata
header

"""
# Author: Nicolas Toussaint <nicolas.toussaint@gmail.com>
# 	  King's College London, UK

import SimpleITK as sitk
import numpy as np
import os
from torch.utils.data import Dataset

IMG_EXTENSIONS = ['.nii.gz', '.nii', '.mha', '.mhd']


def _is_image_file(filename):
    """
    Is the given extension in the filename supported ?
    """
    IMG_EXTENSIONS = ['.nii.gz', '.nii', '.mha', '.mhd']
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def load_image(fname):
    """
    Load supported image and return the loaded image.
    """
    return sitk.ReadImage(fname)


def save_image(itk_img, fname):
    """
    Save ITK image with the given filename
    """
    sitk.WriteImage(itk_img, fname)


def load_metadata(itk_img, key):
    """
    Load the metadata of the input itk image associated with key.
    """
    return itk_img.GetMetaData(key) if itk_img.HasMetaDataKey(key) else None


def extractlabelfromfile(fname):
    with open(fname) as f:
        for line in f:
            if "Label =" in line:
                f.close()
                return line.split(' = ')[1].strip()
    f.close()
    return None


def extractlabelsfromfile(fname):
    with open(fname) as f:
        for line in f:
            if "Labels =" in line:
                f.close()
                return line.split(' = ')[1].strip().split(',')
    f.close()
    return ['']


def find_classes(filenames):
    classes = []
    for idx, f in enumerate(filenames):
        items = extractlabelsfromfile(f)
        for c in items:
            if len(c):
                classes.append(c)
            else:
                print('Warning: file {} does not have any label'.format(f))

    classes = np.sort(classes)
    classes, counts = np.unique(classes, return_counts=True)
    class_to_idx = {classes[i]: i for i in range(len(classes))}

    labelstable = np.zeros([len(filenames), len(classes)], dtype=np.float32)

    for idx, f in enumerate(filenames):
        items = extractlabelsfromfile(f)
        for c in items:
            if c in classes:
                labelstable[idx, np.where(classes == c)[0][0]] = 1

    return labelstable, classes, class_to_idx, counts


def calculate_sample_weights(cardinalities):
    reciprocal_weights = cardinalities / float(np.sum(cardinalities))
    weights = (1. / np.array(reciprocal_weights))
    weights = weights / np.sum(weights)
    weights = np.array(weights, dtype=np.float64)
    return weights


class ITKMetaDataClassification(Dataset):
    """
    Arguments
    ---------
    root : string
        Root directory of dataset. The folder should contain all images for each
        mode of the dataset ('train', 'validate', or 'infer'). Each mode-version
        of the dataset should be in a subfolder of the root directory

        The images can be in any ITK readable format (e.g. .mha/.mhd)
        For the 'train' and 'validate' modes, each image should contain a metadata
        key 'Label' in its dictionary/header

    mode : string, (Default: 'train')
        'train', 'validate', or 'infer'
        Loads data from these folders.
        train and validate folders both must contain subfolders images and labels while
        infer folder needs just images subfolder.
    transform : callable, optional
        A function/transform that takes in input itk image or Tensor and returns a
        transformed
        version. E.g, ``transforms.RandomCrop``

    """

    def __init__(self, root, mode='train', transform=None, target_transform=None):
        # training set or test set

        assert(mode in ['train', 'validate', 'infer'])

        self.mode = mode

        if mode == 'train':
            self.root = os.path.join(root, 'train')
        elif mode == 'validate':
            self.root = os.path.join(root, 'validate')
        else:
            self.root = os.path.join(root, 'infer') if os.path.exists(os.path.join(root, 'infer')) else root

        def gglob(path, regexp=None):
            """Recursive glob
            """
            import fnmatch
            import os
            matches = []
            if regexp is None:
                regexp = '*'
            for root, dirnames, filenames in os.walk(path, followlinks=True):
                for filename in fnmatch.filter(filenames, regexp):
                    matches.append(os.path.join(root, filename))
            return matches

        # Get filenames of all the available images
        self.filenames = [y for y in gglob(self.root, '*.*') if _is_image_file(y)]

        if len(self.filenames) == 0:
            raise(RuntimeError("Found 0 images in subfolders of: " + self.root + "\n"
                               "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))

        self.filenames.sort()

        self.transform = transform
        self.target_transform = target_transform

        labels, classes, class_to_idx, cardinality = find_classes(self.filenames)

        self._labels = labels
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.class_cardinality = cardinality
        self.sample_weights = calculate_sample_weights(self.class_cardinality)

    def __getitem__(self, index):
        """
        Arguments
        ---------
        index : int
            index position to return the data
        Returns
        -------
        tuple: (image, label) where label the organ apparent in the image
        """

        image = load_image(self.filenames[index])

        labels = self.get_labels_from_index(index)

        if self.transform is not None:
            image = self.transform(image).float()
        if self.target_transform is not None:
            labels = self.target_transform(labels)

        if (self.mode == 'infer') or not len(labels):
            return image
        else:
            return image, labels

    def __len__(self):
        return len(self.filenames)

    def get_filenames(self):
        return self.filenames

    def get_root(self):
        return self.root

    def get_classes(self):
        return self.classes

    def get_class_cardinality(self):
        return self.class_cardinality
    
    def get_sample_weights(self):
        return self.sample_weights

    def get_labels_from_index(self, index):
        return self._labels[index]
