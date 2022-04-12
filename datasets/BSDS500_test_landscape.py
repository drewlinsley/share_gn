import os
import numpy as np
import tensorflow as tf
from glob import glob
from config import Config
from ops import tf_fun
from utils import py_utils
from scipy import io, misc
from tqdm import tqdm
from utils import image_processing as im_proc


class data_processing(object):
    def __init__(self):
        self.output_name = 'BSDS500_100_test'
        self.im_extension = '.jpg'
        self.lab_extension = '.mat'
        self.images_dir = '/media/data_cifs/pytorch_projects/datasets/BSDS500_crops/data/images/train'
        self.val_images_dir = '/media/data_cifs/pytorch_projects/datasets/BSDS500_crops/data/images/val'
        self.processed_labels = 'processed_labels'
        self.processed_images = 'processed_images'
        self.config = Config()
        self.train_size = int(1000 * 1)
        self.im_size = [320, 480, 3]  # [321, 481, 3]
        self.model_input_image_size = [320, 480, 3]  # [224, 224, 3]
        self.test_model_input_image_size = [320, 480, 3]
        self.output_size = [320, 480, 1]  # [321, 481, 1]
        self.label_size = self.output_size
        self.default_loss_function = 'pearson'
        self.score_metric = 'sigmoid_accuracy'
        self.aux_scores = ['f1']
        self.store_z = True
        self.input_normalization = 'none'  # 'zscore'
        self.preprocess = [None]  # Preprocessing before tfrecords
        self.folds = {
            'train': 'train',
            'val': 'val',
            'test': 'test',
        }
        self.fold_options = {
            'train': 'mean',
            'val': 'mean'
        }
        self.targets = {
            'image': tf_fun.bytes_feature,
            'label': tf_fun.bytes_feature
        }
        self.tf_dict = {
            'image': tf_fun.fixed_len_feature(dtype='string'),
            'label': tf_fun.fixed_len_feature(dtype='string')
        }
        self.tf_reader = {
            'image': {
                'dtype': tf.float32,
                'reshape': self.im_size
            },
            'label': {
                'dtype': tf.float32,
                'reshape': self.output_size
            }
        }

    def get_data(self):
        files, labels = self.get_files()
        return files, labels

    def get_files(self):
        """Get the names of files."""
        train_images = glob(os.path.join(self.images_dir, '*%s' % self.im_extension)) 
        train_labels = [x.replace(self.im_extension, '.npy').replace('images', 'groundTruth') for x in train_images]
        val_images = glob(os.path.join(self.val_images_dir, '*%s' % self.im_extension))
        val_labels = [x.replace(self.im_extension, '.npy').replace('images', 'groundTruth') for x in val_images]
        train_images = np.array(train_images)
        train_labels = np.array(train_labels)
        val_images = np.array(val_images)
        val_labels = np.array(val_labels)
        test_images = np.array(glob('/media/data_cifs/pytorch_projects/datasets/BSDS500_crops/data/images/test_nocrop/*.jpg'))
        test_labels = np.array([x.replace('images', 'groundTruth').replace('.jpg', '.npy') for x in test_images])
        test_labels = np.array([np.load(x) for x in test_labels])
        keep_idx = np.array([True if x.shape[0] < x.shape[1] else False for x in test_labels])
        test_images = test_images[keep_idx]
        test_labels = test_labels[keep_idx]
        test_images = np.stack([misc.imread(x) for x in test_images], 0)
        test_labels = np.stack(test_labels, 0)
        
        test_images = test_images[:, :320, :480]
        test_labels = test_labels[:, :320, :480][..., None]

        # Select images for training
        sort_idx = np.argsort(train_images)
        train_images = train_images[sort_idx[:self.train_size]]
        train_labels = train_labels[sort_idx[:self.train_size]]

        # Build CV dict
        cv_files, cv_labels = {}, {}
        cv_files[self.folds['train']] = train_images
        cv_files[self.folds['val']] = val_images
        cv_files[self.folds['test']] = test_images
        cv_labels[self.folds['train']] = train_labels
        cv_labels[self.folds['val']] = val_labels
        cv_labels[self.folds['test']] = test_labels
        return cv_files, cv_labels

    def get_test_files(self):
        """Get the names of files."""
        test_images = np.array(
            glob('/media/data_cifs/pytorch_projects/datasets/BSDS500_crops/data/images/test_nocrop/*.jpg'))
        test_labels = np.array(
            [x.replace('images', 'groundTruth').replace('.jpg', '.npy') for x in test_images])
        test_labels = np.array(
            [np.load(x) for x in test_labels])
        keep_idx = np.array([True if x.shape[0] < x.shape[1] else False for x in test_labels])
        test_images = test_images[keep_idx]
        return test_images
