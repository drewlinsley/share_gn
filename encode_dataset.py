#!/usr/bin/env python
import os
import numpy as np
from config import Config
from utils import py_utils
from argparse import ArgumentParser
from ops.data_to_tfrecords import data_to_tfrecords
from tqdm import tqdm


def pad_zeros(x, total):
    """Pad x with zeros to total digits."""
    num_pad = total - len(x)
    for idx in range(num_pad):
        x = '0' + x
    return x


def create_shards(
        it_shards,
        shard_dir,
        key,
        files,
        labels,
        targets,
        im_size,
        label_size,
        preprocess,
        store_z,
        normalize_im):
    """Build shards in a loop."""
    all_files = files[key]
    all_labels = labels[key]
    total_data = len(all_files) / it_shards
    mask = np.arange(it_shards).reshape(1, -1).repeat(total_data).reshape(-1)
    all_files = all_files[:len(mask)]
    all_labels = all_labels[:len(mask)]
    total_shards = pad_zeros(str(it_shards), 5)
    for idx in tqdm(
            range(it_shards), total=it_shards, desc='Building %s' % key):
        it_mask = mask == idx
        shard_label = pad_zeros(str(idx), 5)
        shard_name = os.path.join(
            shard_dir,
            '%s-%s-of-%s.tfrecords' % (key, shard_label, total_shards))
        it_files = {key: all_files[it_mask]}
        it_labels = {key: all_labels[it_mask]}
        data_to_tfrecords(
            files=it_files,
            labels=it_labels,
            targets=targets,
            ds_name=shard_name,
            im_size=im_size,
            label_size=label_size,
            preprocess=preprocess,
            store_z=store_z,
            it_ds_name=shard_name,
            normalize_im=normalize_im)


def encode_dataset(dataset, train_shards=0, val_shards=0, force_val=False):
    config = Config()
    data_class = py_utils.import_module(
        module=dataset, pre_path=config.dataset_classes)
    data_proc = data_class.data_processing()
    data = data_proc.get_data()
    if len(data) == 2:
        files, labels = data
        nhot = None
    elif len(data) == 3:
        files, labels, nhot = data
    else:
        raise NotImplementedError
    targets = data_proc.targets
    im_size = data_proc.im_size
    if hasattr(data_proc, 'preprocess'):
        preproc_list = data_proc.preprocess
    else:
        preproc_list = []
    if hasattr(data_proc, 'label_size'):
        label_size = data_proc.label_size
    else:
        label_size = None
    if hasattr(data_proc, 'label_size'):
        store_z = data_proc.store_z
    else:
        store_z = False
    if hasattr(data_proc, 'normalize_im'):
        normalize_im = data_proc.normalize_im
    else:
        normalize_im = False
    if not train_shards:
        ds_name = os.path.join(config.tf_records, data_proc.output_name)
        data_to_tfrecords(
            files=files,
            labels=labels,
            targets=targets,
            nhot=nhot,
            ds_name=ds_name,
            im_size=im_size,
            label_size=label_size,
            preprocess=preproc_list,
            store_z=store_z,
            normalize_im=normalize_im)
    else:
        assert val_shards > 0, 'Choose the number of val shards.'
        raise NotImplementedError('Needs support for nhot.')
        shard_dir = os.path.join(config.tf_records, data_proc.output_name)
        py_utils.make_dir(shard_dir)
        if not force_val:
            create_shards(
                it_shards=train_shards,
                shard_dir=shard_dir,
                key='train',
                files=files,
                labels=labels,
                targets=targets,
                im_size=im_size,
                label_size=label_size,
                preprocess=preproc_list,
                store_z=store_z,
                normalize_im=normalize_im)
        create_shards(
            it_shards=val_shards,
            shard_dir=shard_dir,
            key='val',
            files=files,
            labels=labels,
            targets=targets,
            im_size=im_size,
            label_size=label_size,
            preprocess=preproc_list,
            store_z=store_z,
            normalize_im=normalize_im)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument(
        '--dataset',
        dest='dataset',
        help='Name of the dataset.')
    parser.add_argument(
        '--train_shards',
        type=int,
        default=0,
        dest='train_shards',
        help='Number of train shards for the dataset.')
    parser.add_argument(
        '--val_shards',
        type=int,
        default=128,
        dest='val_shards',
        help='Number of val shards for the dataset.')
    parser.add_argument(
        '--force_val',
        dest='force_val',
        action='store_true',
        help='Force creation of validation dataset.')
    args = parser.parse_args()
    encode_dataset(**vars(args))

