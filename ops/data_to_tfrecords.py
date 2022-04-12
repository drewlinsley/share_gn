"""Routines for encoding data into TFrecords."""
import os
import numpy as np
import tensorflow as tf
from scipy import misc
from tqdm import tqdm
from utils import image_processing
from skimage import color
from skimage.io import imread


def hed_pad(lim, r_stride=320, bsds_path='/media/data_cifs/image_datasets/hed_bsds/HED-BSDS', process_label=False):
    """Pad HED images for bsds."""
    if process_label:
        lim /= 255.
        if len(lim.shape) == 3:
            lim = lim[..., 0]
        lim = lim[..., None]
    lsh = lim.shape
    if lsh[0] > lsh[1]:
        # Flip all to landscape
        lim = lim.transpose((1, 0, 2))
        lsh = lim.shape
    if lsh[0] < r_stride:
        # Pad to 320
        up_offset = (r_stride - lsh[0]) // 2
        down_offset = up_offset
        if up_offset + down_offset + lsh[0] < r_stride:
            down_offset += 1
        elif up_offset + down_offset + lsh[0] > r_stride:
            down_offset -= 1
        pad_up_offset = np.zeros((up_offset, lsh[1], lsh[-1]))
        pad_down_offset = np.zeros((down_offset, lsh[1], lsh[-1]))
        lim = np.concatenate((pad_up_offset, lim, pad_down_offset), 0)
    if lsh[1] < r_stride:
        # Pad to 320
        up_offset = (r_stride - lsh[1]) // 2
        down_offset = up_offset
        if up_offset + down_offset + lsh[1] < r_stride:
            down_offset += 1
        elif up_offset + down_offset + lsh[1] > r_stride:
            down_offset -= 1
        pad_up_offset = np.zeros((lsh[0], up_offset, lsh[-1]))
        pad_down_offset = np.zeros((lsh[0], down_offset, lsh[-1]))
        lim = np.concatenate((pad_up_offset, lim, pad_down_offset), 1)
    return lim


def load_image(f, im_size=False, repeat_image=False):
    """Load image and convert it to a 4D tensor."""
    if '.npy' in f:
        image = np.load(f).astype(np.float32)
    else:
        image = misc.imread(f).astype(np.float32)
    if len(image.shape) < 3 and repeat_image and im_size:  # Force H/W/C
        image = np.repeat(image[:, :, None], im_size[-1], axis=-1)
    return image


def normalize(im):
    """Normalize to [0, 1]."""
    min_im = im.min()
    max_im = im.max()
    return (im - min_im) / (max_im - min_im)


def create_example(data_dict):
    """Create entry in tfrecords."""
    data_dict = {k: v for k, v in data_dict.iteritems() if v is not None}
    return tf.train.Example(
        # Example contains a Features proto object
        features=tf.train.Features(
            # Features has a map of string to Feature proto objects
            feature=data_dict
        )
    )


def preprocess_image(image, preprocess, im_size, process_label=False):
    """Preprocess image files before encoding in TFrecords."""
    for pre in preprocess:
        if pre == 'crop_center':
            image = image_processing.crop_center(image, im_size)
        elif pre == 'crop_center_resize':
            im_shape = image.shape
            min_shape = np.min(im_shape[:2])
            crop_im_size = [min_shape, min_shape, im_shape[-1]]
            image = image_processing.crop_center(image, crop_im_size)
            image = image_processing.resize(image, im_size)
        elif pre == 'resize':
            image = image_processing.resize(image, im_size)
        elif pre == 'pad_resize':
            image = image_processing.pad_square(image)
            image = image_processing.resize(image, im_size)
        if pre == 'hed_pad':
            image = hed_pad(image, process_label=process_label)
        if pre == 'trim_extra_dims':
            im_shape = image.shape
            if im_shape[-1] > im_size[-1]:
                image = image[..., :im_size[-1]]
            elif im_shape[-1] == im_size[-1]:
                pass
            else:
                raise RuntimeError('Failed preproc on trim_extra_dims.')
        if pre == 'rgba2rgb':
            image = image[:, :, :-1]
        if pre == 'to_float32':
            image = image.astype(np.float32)
        if pre == 'rgba2gray':
            image = color.rgb2gray(image[:, :, :-1])
        if pre == 'exclude_white':
            thresh = 0.25
            hw = np.prod(image.shape[:-1])
            white_check = np.sum(
                np.std(image, axis=-1) < 0.01) / hw
            if white_check > thresh:
                return False
    return image.astype(np.float32)


def encode_tf(encoder, x):
    """Process data for TFRecords."""
    encoder_name = encoder.func_name
    if 'bytes' in encoder_name:
        return encoder(x.tostring())
    else:
        return encoder(x)


def data_to_tfrecords(
        files,
        labels,
        targets,
        nhot,
        ds_name,
        im_size,
        label_size,
        preprocess,
        store_z=False,
        normalize_im=False,
        it_ds_name=None,
        repeat_image=False):
    """Convert dataset to tfrecords."""
    print('Building dataset: %s' % ds_name)
    no_means = False
    for idx, ((fk, fv), (lk, lv)) in enumerate(
        zip(
            files.iteritems(),
            labels.iteritems())):
        if it_ds_name is None:
            it_ds_name = '%s_%s.tfrecords' % (ds_name, fk)
        if store_z:
            means = []
        else:
            means = np.zeros((im_size))
        if nhot is not None:
            use_nhot = True
            f_nhot = nhot[fk]
        else:
            use_nhot = False
        with tf.python_io.TFRecordWriter(it_ds_name) as tfrecord_writer:
            image_count = 0
            for f_idx, (it_f, it_l) in tqdm(
                    enumerate(zip(fv, lv)),
                    total=len(fv),
                    desc='Building %s' % fk):
                example = None
                try:
                    if isinstance(it_f, basestring):
                        if '.npy' in it_f:
                            image = np.load(it_f)
                        else:
                            image = load_image(
                                it_f,
                                im_size,
                                repeat_image=repeat_image).astype(np.float32)
                        if len(image.shape) > 1:
                            image = preprocess_image(
                                image, preprocess,
                                im_size)
                        if image is False:
                            print('Skipping image')
                            continue
                            print('Check')
                    else:
                        image = preprocess_image(it_f, preprocess, im_size)
                    if normalize_im:
                        image = normalize(image)
                    if store_z:
                        means += [image]
                    else:
                        if np.all(np.array(image.shape) == np.array(means.shape)):
                            means += image
                        else:
                            no_means = True
                    if isinstance(it_l, basestring):
                        if '.npy' in it_l:
                            label = np.load(it_l)
                        else:
                            label = load_image(
                                it_l,
                                label_size,
                                repeat_image=False).astype(np.float32)
                        if len(label.shape) > 1:
                            label = preprocess_image(
                                label, preprocess, label_size, process_label=True)
                    else:
                        label = it_l
                        if isinstance(
                                label, np.ndarray) and len(label.shape) > 1:
                            label = preprocess_image(
                                label, preprocess, label_size, process_label=True)
                    data_dict = {
                        'image': encode_tf(targets['image'], image),
                        'label': encode_tf(targets['label'], label)
                    }
                    if use_nhot:
                        data_dict['nhot'] = encode_tf(
                            targets['nhot'],
                            f_nhot[f_idx])
                    if targets.get('height', False):
                        data_dict['height'] = encode_tf(targets['height'], image.shape[0])
                    if targets.get('width', False):
                        data_dict['width'] = encode_tf(targets['width'], image.shape[1])
                    example = create_example(data_dict)
                except Exception:
                    pass
                if example is not None:
                    # Keep track of how many images we use
                    image_count += 1
                    # use the proto object to serialize the example to a string
                    serialized = example.SerializeToString()
                    # write the serialized object to disk
                    tfrecord_writer.write(serialized)
                    example = None
            if store_z:
                means = np.asarray(means).reshape(len(means), -1)
                np.savez(
                    '%s_%s_means' % (ds_name, fk),
                    image={
                        'mean': means.mean(),
                        'std': means.std()
                    })
            elif not no_means:
                np.save(
                    '%s_%s_means' % (ds_name, fk), means / float(image_count))
            else:
                print('Failed to save means.')
            print('Finished %s with %s images (dropped %s)' % (
                it_ds_name, image_count, len(fv) - image_count))
            it_ds_name = None

