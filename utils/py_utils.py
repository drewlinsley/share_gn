import os
# import re
from datetime import datetime
import numpy as np


def merge(dict1, dict2):
    return(dict2.update(dict1))


def get_dt_stamp():
    """Get date-timestamp."""
    return str(datetime.now()).replace(
        ' ',
        '_').replace(
        ':',
        '_').replace(
        '-',
        '_').replace(
        '.',
        '_')


def flatten_list(l):
    """Flatten a list of lists."""
    warning_msg = 'Warning: returning None.'
    assert len(l), 'Encountered empty list.'
    if l is None or l[0] is None:
        print(warning_msg)
        return [None]
    else:
        return [val for sublist in l for val in sublist]


def import_module(module, pre_path='datasets'):
    """Dynamically import a module."""
    return getattr(
        __import__(pre_path, fromlist=[module]), module)


def make_dir(d):
    """Make directory d if it does not exist."""
    if not os.path.exists(d):
        os.makedirs(d)


def prepare_directories(config, exp_label):
    """Prepare some useful directories."""
    dir_list = {
        'checkpoints': os.path.join(
            config.checkpoints, exp_label),
        'summaries': os.path.join(
            config.summaries, exp_label),
        'condition_evaluations': os.path.join(
            config.condition_evaluations, exp_label),
        'weights': os.path.join(
            config.condition_evaluations, exp_label, 'weights')
    }
    [make_dir(v) for v in dir_list.values()]
    return dir_list


def save_npys(data, model_name, output_string):
    """Save key/values in data as numpys."""
    for k, v in data.iteritems():
        k = k.replace('/', '_')
        output = os.path.join(
            output_string,
            '%s_%s' % (model_name, k))
        try:
            np.save(output, v)
        except Exception:
            print('Failed to save %s' % k)


def check_path(data_pointer, msg):
    """Check that the path exists."""
    if not os.path.exists(data_pointer):
        print(msg)
        return False
    else:
        return data_pointer


def convert_to_tuple(v):
    """Convert v to a tuple."""
    if not isinstance(v, tuple):
        return tuple(v)
    else:
        return v


def add_to_config(d, config):
    """Add attributes to config class."""
    try:
        for k, v in d.iteritems():
            if isinstance(v, list) and len(v) == 1:
                v = v[0]
            setattr(config, k, v)
    except:
        for k, v in d.items():
            if isinstance(v, list) and len(v) == 1:
                v = v[0]
            setattr(config, k, v)
    return config


def get_data_pointers(dataset, base_dir, local_dir, cv):
    """Get data file pointers."""
    data_pointer = os.path.join(local_dir, '%s_%s.tfrecords' % (dataset, cv))
    data_means = os.path.join(local_dir, '%s_%s_means.npy' % (dataset, cv))
    local_check = check_path(
        data_pointer,
        'Local path for %s not found, trying CIFS.' % data_pointer)
    if not local_check:
        data_pointer = os.path.join(
            base_dir, '%s_%s.tfrecords' % (dataset, cv))
        data_means = os.path.join(
            base_dir, '%s_%s_means.npy' % (dataset, cv))
        check_path(
            data_pointer, '%s not found.' % data_pointer)
    else:
        print('*' * 30)
        print('Reading from a local file.')
        print('*' * 30)
    mean_loc = check_path(
        data_means,
        '%s not found for cv: %s. Trying .npz fallback.' % (data_means, cv))
    data_means_image, data_means_label = None, None
    if not mean_loc:
        alt_data_pointer = data_means.replace('.npy', '.npz')
        alt_data_pointer = check_path(
            alt_data_pointer,
            '%s not found. npz fallback failed.' % alt_data_pointer)
        # TODO: Fix this API and make it more flexible. Kill npzs in Allen?
        if not alt_data_pointer:
            # No mean for this dataset
            data_means = None
        else:
            print('Using mean file: %s' % alt_data_pointer)
            data_means = np.load(alt_data_pointer, allow_pickle=True, encoding="latin1")
            if 'image' in data_means.keys():
                data_means_image = data_means['image']
            if 'images' in data_means.keys():
                data_means_image = data_means['images']
            if 'label' in data_means.keys():
                data_means_label = data_means['label']
            if 'labels' in data_means.keys():
                data_means_label = data_means['labels']
            if data_means_image is not None and isinstance(
                    data_means_image, np.object):
                data_means_image = data_means_image.item()
            if data_means_label is not None and isinstance(
                    data_means_label, np.object):
                data_means_label = data_means_label.item()
    else:
        try:
            data_means_image = np.load(data_means)
        except Exception as e:
            print('Failed to load means: %s' % e)
    return data_pointer, data_means_image, data_means_label
