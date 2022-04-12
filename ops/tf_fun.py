"""Misc. functions for tensorflow model construction and training."""
import os
import numpy as np
import tensorflow as tf
import json
from utils import py_utils
from tqdm import tqdm
from scipy import sparse
from scipy import ndimage as ndi
from sklearn.metrics import average_precision_score
from skimage.morphology import watershed
from skimage.feature import peak_local_max
from collections import OrderedDict
from matplotlib import pyplot as plt


def visualize_recurrence(
        idx, image, label, logits, ff, h2s, config, debug=False):
    """Visualize norm of diffs of timesteps of activity."""
    f, axarr = plt.subplots(
        2,
        len(h2s), figsize=(30, 15))
    image = image.squeeze()
    post_decoded_final = logits.squeeze()
    if image.shape[-1] == 3:
        axarr[0, 2].imshow(
            (image).astype(np.uint8))
    else:
        axarr[0, 2].imshow(
            (image).astype(np.uint8), cmap='Greys_r')
    axarr[0, 3].imshow(label.squeeze(), cmap='Greys', vmin=0, vmax=0.5)
    axarr[0, 4].imshow(
        sigmoid_fun(post_decoded_final),
        vmin=0.0,
        vmax=1.0,
        cmap='Greys_r')
    for its, post in enumerate(h2s):
        axarr[1, its].imshow(
            sigmoid_fun(post.squeeze()), vmin=0, vmax=1, cmap='Greys_r')
    [axi.set_xticks([]) for axi in axarr.ravel()]
    [axi.set_yticks([]) for axi in axarr.ravel()]
    py_utils.make_dir(config.model)
    plt.savefig(
        os.path.join(
            config.model,
            '%d.pdf' % idx))
    if debug:
        plt.show()
    plt.close(f)
    norms = ((h2s[:-1] - h2s[1:]) ** 2).mean(-1).mean(-1).mean(-1)
    return norms


def get_batch_queues(images, labels, gpu_device, capacity=100):
    """Return batch queue objects for multi gpu training."""
    return tf.contrib.slim.prefetch_queue.prefetch_queue(
        [images, labels], capacity=capacity * len(gpu_device))


def interpret_nl(self, nl_type):
    """Return activation function."""
    if nl_type == 'tanh':
        return tf.nn.tanh
    elif nl_type == 'relu':
        return tf.nn.relu
    elif nl_type == 'elu':
        return tf.nn.elu
    elif nl_type == 'selu':
        return tf.nn.selu
    elif nl_type == 'leaky_relu':
        return tf.nn.leaky_relu
    elif nl_type == 'hard_tanh':
        return lambda z: tf.maximum(tf.minimum(z, 1), 0)
    else:
        raise NotImplementedError(nl_type)


def get_gammanet_constructor(
        compression,
        ff_kernels,
        ff_repeats,
        features,
        fgru_kernels):
    """Convert model options into an ordered dict."""
    model_struct = OrderedDict()
    print('compression %s' % len(compression))
    print('ff_kernels %s' % len(ff_kernels))
    print('ff_repeats %s' % len(ff_repeats))
    print('features %s' % len(features))
    print('fgru_kernels %s' % len(fgru_kernels))
    assert (
        len(compression) ==
        len(ff_kernels) ==
        len(ff_repeats) ==
        len(features) ==
        len(fgru_kernels)), 'Gammanet constructor args are bad.'
    for idx, (comp, ff_k, ff_r, feats, fgru_k) in enumerate(zip(
            compression,
            ff_kernels,
            ff_repeats,
            features,
            fgru_kernels)):
        layer_dict = {
            'compression': comp,
            'features': feats,
            'ff_kernels': ff_k,
            'ff_repeats': ff_r,
            'fgru_kernels': fgru_k,
        }
        model_struct[idx] = layer_dict
    return model_struct


def interpret_data_format(data_tensor, data_format):
    """Interepret short data format, fix input, and return a long string."""
    if data_format is 'NCHW':
        data_tensor = tf.transpose(data_tensor, (0, 3, 1, 2))
        long_data_format = 'channels_first'
    else:
        long_data_format = 'channels_last'
    return data_tensor, long_data_format


def check_early_stop(
        perf_history,
        minimum_length=20,
        short_history=3,
        long_history=5,
        fail_function=np.less_equal):
    """
    Determine whether to stop early. Using deepgaze criteria:

    We determine this point by comparing the performance from
    the last three epochs to the performance five epochs before those.
    Training runs for at least 20 epochs, and is terminated if all three
    of the last epochs show decreased performance or if
    800 epochs are reached.

    """
    if len(perf_history) < minimum_length:
        early_stop = False
    else:
        short_perf = perf_history[-short_history:]
        long_perf = perf_history[-long_history + short_history:short_history]
        short_check = fail_function(np.mean(long_perf), short_perf)
        if all(short_check):  # If we should stop
            early_stop = True
        else:
            early_stop = False

    return early_stop


def count_parameters(var_list, keyword='symm', print_count=False):
    """Count the parameters in a tf model."""
    params = []
    symm_weights = []
    for v in var_list:
        if keyword in v.name:  # TODO: ID vars w/ grad overrides
            count = np.maximum(np.prod(
                [x for x in v.get_shape().as_list()
                    if x > 1]), 1)
            count = (count / 2) + v.get_shape().as_list()[-1]
            params += [count]
            symm_weights += [v.name]
        else:
            params += [
                np.maximum(
                    np.prod(
                        [x for x in v.get_shape().as_list()
                            if x > 1]), 1)]
    if print_count:
        param_list = [
            {v.name: [str(v.get_shape().as_list()), k]} for k, v in zip(
                params, var_list)]
        print(json.dumps(param_list, indent=4))
        print('Found %s variables' % len(param_list))
    if len(symm_weights):
        print('Found the following symmetric weights: %s' % symm_weights)
    return np.sum(params).astype(int)


def check_shapes(scores, labels):
    """Check and fix the shapes of scores and labels."""
    if not isinstance(scores, list):
        if len(
                scores.get_shape()) != len(
                    labels.get_shape()):
            score_shape = scores.get_shape().as_list()
            label_shape = labels.get_shape().as_list()
            if len(
                score_shape) == 2 and len(
                    label_shape) == 1 and score_shape[-1] == 1:
                labels = tf.expand_dims(labels, axis=-1)
            elif len(
                score_shape) == 2 and len(
                    label_shape) == 1 and score_shape[-1] == 1:
                scores = tf.expand_dims(scores, axis=-1)
    return scores, labels


def bytes_feature(values):
    """Bytes features for writing TFRecords."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[values]))


def int64_feature(values):
    """Int64 features for writing TFRecords."""
    if not isinstance(values, (tuple, list)):
        values = [values]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=values))


def float_feature(values):
    """Float features for writing TFRecords."""
    if isinstance(values, np.ndarray):
        values = [v for v in values]
    if not isinstance(values, (tuple, list)):
        values = [values]
    return tf.train.Feature(float_list=tf.train.FloatList(value=values))


def fixed_len_feature(length=[], dtype='int64'):
    """Features for reading TFRecords."""
    if dtype == 'int64':
        return tf.FixedLenFeature(length, tf.int64)
    elif dtype == 'int32':
        return tf.FixedLenFeature(length, tf.int32)
    elif dtype == 'string':
        return tf.FixedLenFeature(length, tf.string)
    elif dtype == 'float' or dtype == 'float32':
        return tf.FixedLenFeature(length, tf.float32)
    else:
        raise RuntimeError('Cannot understand the fixed_len_feature dtype.')


def image_summaries(
        images,
        tag):
    """Wrapper for creating tensorboard image summaries.

    Parameters
    ----------
    images : tensor
    tag : str
    """
    im_shape = [int(x) for x in images.get_shape()]
    tag = '%s images' % tag
    if im_shape[-1] <= 3 and (
            len(im_shape) == 3 or len(im_shape) == 4):
        tf.summary.image(tag, images)
    elif im_shape[-1] <= 3 and len(im_shape) == 5:
        # Spatiotemporal image set
        res_ims = tf.reshape(
            images,
            [im_shape[0] * im_shape[1]] + im_shape[2:])
        tf.summary.image(tag, res_ims)


def sigmoid_fun(x):
    """Apply sigmoid to maps before mAP."""
    return 1 / (1 + np.exp(x))


def calculate_map(
        it_val_dict,
        exp_label,
        config,
        map_dir='maps',
        auto_adjust=False):
    """Calculate map and ARAND for segmentation performance."""
    py_utils.make_dir(map_dir)

    def get_segments(x, threshold=0.5, comp=np.greater):
        """Watershed boundary map."""
        distance = ndi.distance_transform_edt(comp(x, threshold).astype(float))
        local_maxi = peak_local_max(
            distance,
            indices=False,
            footprint=np.ones((10, 10)))
        markers = ndi.label(local_maxi)[0]
        return watershed(-distance, markers).astype(np.int32)

    maps, arands = [], []
    if it_val_dict is not None:
        for ol in tqdm(range(len(it_val_dict)), desc='Evaluating mAPs'):
            eval_dicts = it_val_dict[ol]
            eval_logits = eval_dicts['logits']
            eval_labels = eval_dicts['labels']
            for log, lab in zip(eval_logits, eval_labels):
                lab = lab.squeeze()
                log_shape = log.shape
                if len(log.shape) > 2:
                    if log_shape[-1] > 1:
                        log = -1 * log[..., [0, 1]].mean(-1)
                        lab = (lab[..., [0, 1]].mean(-1) > 0.5).astype(
                            np.int32)
                if auto_adjust and lab.mean() < 0.5:
                    lab = 1. - lab
                sig_pred = 1. - sigmoid_fun(log.squeeze())

                # First calculate map
                maps += [average_precision_score(
                    y_score=sig_pred.ravel(),
                    y_true=lab.ravel())]

                # Then get ARAND on segments
                pred_segs = get_segments(sig_pred)
                lab_segs = get_segments(lab)
                lab_mask = (lab == 0).astype(np.int32)
                pred_segs *= lab_mask
                lab_segs *= lab_mask
                arands += [adapted_rand(pred_segs, lab_segs)]

        out_path = os.path.join(map_dir, exp_label)
        print('mAP is: %s' % np.mean(maps))
        print('ARAND is: %s' % np.mean(arands))
        print('Saved to: %s.npz' % out_path)
        np.savez(
            out_path,
            maps=maps,
            arands=arands,
            config=config,
            val_dict=it_val_dict)
    else:
        print('Received an empty validation dict.')
    return maps, arands


def adapted_rand(seg, gt, all_stats=False):
    """Compute Adapted Rand error as defined by the SNEMI3D contest [1]
    Formula is given as 1 - the maximal F-score of the Rand index
    (excluding the zero component of the original labels). Adapted
    from the SNEMI3D MATLAB script, hence the strange style.
    Parameters
    ----------
    seg : np.ndarray
        the segmentation to score, where each value is the label at that point
    gt : np.ndarray, same shape as seg
        the groundtruth to score against, where each value is a label
    all_stats : boolean, optional
        return precision and recall as a 3-tuple with rand_error
    Returns
    -------
    are : float
        The adapted Rand error; equal to $1 - \frac{2pr}{p + r}$,
        where $p$ and $r$ are the precision and recall described below.
    prec : float, optional
        The adapted Rand precision. (Returned when `all_stats` is ``True``.)
    rec : float, optional
        The adapted Rand recall.  (Only returned when `all_stats` is ``True``.)
    References
    ----------
    [1]: http://brainiac2.mit.edu/SNEMI3D/evaluation
    """
    # segA is truth, segB is query
    segA = np.ravel(gt).astype(int)
    segB = np.ravel(seg).astype(int)
    n = segA.size

    n_labels_A = np.amax(segA) + 1
    n_labels_B = np.amax(segB) + 1

    ones_data = np.ones(n)

    p_ij = sparse.csr_matrix((
        ones_data, (segA[:], segB[:])), shape=(n_labels_A, n_labels_B))

    a = p_ij[1:n_labels_A, :]
    b = p_ij[1:n_labels_A, 1:n_labels_B]
    c = p_ij[1:n_labels_A, 0].todense()
    d = b.multiply(b)

    a_i = np.array(a.sum(1))
    b_i = np.array(b.sum(0))

    sumA = np.sum(a_i * a_i)
    sumB = np.sum(b_i * b_i) + (np.sum(c) / n)
    sumAB = np.sum(d) + (np.sum(c) / n)

    precision = sumAB / sumB
    recall = sumAB / sumA

    fScore = 2.0 * precision * recall / (precision + recall)
    are = 1.0 - fScore

    if all_stats:
        return (are, precision, recall)
    else:
        return are


def laplace(x):
    """Regularize the laplace of the kernel."""
    kernel = np.asarray([
        [0.5, 1, 0.5],
        [1, -6, 1],
        [0.5, 1, 0.5]
    ])[:, :, None, None]
    kernel = np.repeat(
        kernel,
        int(x.get_shape()[-1]), axis=-2).astype(np.float32)
    tf_kernel = tf.get_variable(
        name='laplace_%s' % x.name.split('/')[-1].split(':')[0],
        initializer=kernel)
    reg_activity = tf.nn.conv2d(
        x,
        filter=tf_kernel,
        strides=[1, 1, 1, 1],
        padding='SAME')
    return tf.reduce_mean(tf.pow(reg_activity, 2))


def orthogonal(x, eps=1e-12):
    """Regularization for orthogonal components."""
    x_shape = [int(d) for d in x.get_shape()]
    out_rav_x = tf.reshape(tf.transpose(x, [3, 0, 1, 2]), [x_shape[3], -1])
    z = tf.matmul(out_rav_x, out_rav_x, transpose_b=True)  # Dot products
    z -= tf.eye(x_shape[3])
    return tf.reduce_sum(tf.abs(z))  # Minimize off-diagonals


def affinitize(img, dst, dtype='float32'):
    """
    Transform segmentation to 3D affinity graph.
    Args:
        img: 3D indexed image, with each index corresponding to each segment.
    Returns:
        ret: affinity graph
    """
    ret = np.zeros((1,) + img.shape, dtype=dtype)

    (dz, dy, dx) = dst
    if dz != 0:
        # z-affinity.
        assert dz and abs(dz) < img.shape[0]
        if dz > 0:
            ret[0, dz:, :, :] = (
                img[dz:, :, :] == img[:-dz, :, :]) & (img[dz:, :, :] > 0)
        else:
            dz = abs(dz)
            ret[0, :-dz, :, :] = (
                img[dz:, :, :] == img[:-dz, :, :]) & (img[dz:, :, :] > 0)

    if dy != 0:
        # y-affinity.
        assert dy and abs(dy) < img.shape[2]
        if dy > 0:
            ret[0, :, dy:, :] = (
                img[:, dy:, :] == img[:, :-dy, :]) & (img[:, dy:, :] > 0)
        else:
            dy = abs(dy)
            ret[0, :, :-dy, :] = (
                img[:, dy:, :] == img[:, :-dy, :]) & (img[:, dy:, :] > 0)

    if dx != 0:
        # x-affinity.
        assert dx and abs(dx) < img.shape[1]
        if dx > 0:
            ret[0, :, :, dx:] = (
                img[:, :, dx:] == img[:, :, :-dx]) & (img[:, :, dx:] > 0)
        else:
            dx = abs(dx)
            ret[0, :, :, :-dx] = (
                img[:, :, dx:] == img[:, :, :-dx]) & (img[:, :, dx:] > 0)
    return np.squeeze(ret)


def derive_affinities(label_volume, long_range=True, use_3d=True):
    """Derive affinities from label_volume."""
    # distances = np.rot90(np.eye(affinity)).astype(int)
    # Hard code this to be long-range
    if long_range:
        if use_3d:
            distances = [
                [0, 0, 1],
                [0, 1, 0],
                [1, 0, 0],
                [0, 0, 3],
                [0, 3, 0],
                [2, 0, 0],
                [0, 0, 9],
                [0, 9, 0],
                [3, 0, 0],
                [0, 0, 27],
                [0, 27, 0],
                [4, 0, 0],
            ]
        else:
            distances = [
                [0, 0, 1],
                [0, 1, 0],
                [0, 0, 3],
                [0, 3, 0],
                [0, 0, 9],
                [0, 9, 0],
                [0, 0, 27],
                [0, 27, 0],
            ]
    ground_truth_affinities = []
    for i in range(len(distances)):
        aff = affinitize(label_volume, dst=distances[i])
        ground_truth_affinities += [aff.astype(int)]
    label_volume = np.array(
        ground_truth_affinities).transpose(1, 2, 3, 0)
    return label_volume


class Initializer(object):
    """Custom TF initializations."""
    def __init__(self, dtype=tf.float32):
        self.dtype = dtype


class Identity(Initializer):
    """Do the mely-identity init."""
    def __init__(self, dtype=tf.float32, randomize=True):
        self.dtype = tf.as_dtype(dtype)
        self.randomize = randomize

    def __call__(self, shape, dtype=None, partition_info=None):
        """Return the initializer tensor."""
        assert shape[-2] == shape[-1], 'Designed for In=Out channel tensors.'
        if dtype is not None:
            self.dtype = dtype
        tensor = np.ones(shape)
        for k in range(shape[-1]):
            tensor[:, :, k, k] = 1.
        tensor /= shape[0] ** 2
        if self.randomize:
            tensor = tf.initializers.variance_scaling(scale=1)(shape) + tensor
            # tensor = tf.initializers.orthogonal(gain=0.5)(shape) + tensor
        return tf.cast(tensor, self.dtype)

