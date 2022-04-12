#!/usr/bin/env python
import tensorflow as tf
from layers.feedforward import normalization
from layers.feedforward import conv
from layers.recurrent import recurrent_vgg16_cheap_deepest_simple_sym_minimal as vgg16
from ops import tf_fun


def weight_decay():
    return 0.0002


def get_aux():
    """Auxilary options for GN."""
    return {
        'attention': False,  # 'gala',  # 'gala',  # 'gala', 'se', False
        'attention_layers': 1,
        'norm_attention': False,
        'saliency_filter': 3,
        # 'gate_nl': tf.keras.activations.hard_sigmoid,
        'use_homunculus': False,
        'gate_homunculus': False,
        'single_homunculus': False,
        'combine_fgru_output': False,
        'upsample_nl': False,
        'upsample_convs': False,
        'separable_upsample': False,
        'separable_convs': False,  # Multiplier
        # 'fgru_output_normalization': True,
        'fgru_output_normalization': False,
        'fgru_batchnorm': True,
        'c1_c2_norm': True,
        'skip_connections': False,
        'residual': True,  # intermediate resid connections
        'while_loop': False,
        'image_resize': tf.image.resize_bilinear,  # tf.image.resize_nearest_neighbor
        'bilinear_init': False,
        'nonnegative': True,
        'adaptation': False,
        'symmetric_weights': False,  # 'channel',  # 'spatial_channel', 'channel', False
        'force_alpha_divisive': False,
        'force_omega_nonnegative': False,
        'td_cell_state': False,
        'td_gate': False,  # Add top-down activity to the in-gate
        'dilations': [1, 1, 1, 1],
        'weight_norm': False,
        'partial_padding': False
    }
# Turned off fgru batchnorm and turned on weight_norm

def v2_small():
    compression = ['pool', 'pool', 'upsample']
    ff_kernels = [[False]] * len(compression)
    ff_repeats = [[False]] * len(compression)
    features = [128, 256, 128]  # Default
    fgru_kernels = [[1, 1], [1, 1], [1, 1]]
    ar = ['']  # , 'fgru_3', 'fgru_4']  # Output layer ids
    return compression, ff_kernels, ff_repeats, features, fgru_kernels, ar


def v2_big_working():
    compression = ['pool', 'pool', 'pool', 'pool', 'upsample', 'upsample', 'upsample']
    ff_kernels = [[False]] * len(compression)
    ff_repeats = [[False]] * len(compression)
    # features = [24, 18, 18, 48, 64, 48, 18, 18, 24]  # Bottleneck
    features = [128, 256, 512, 512, 512, 256, 128]  # Default
    # features = [16, 18, 20, 48, 20, 18, 16]  # DEFAULT
    # fgru_kernels = [[11, 11], [7, 7], [5, 5], [3, 3], [1, 1], [1, 1], [1, 1]]
    fgru_kernels = [[9, 9], [5, 5], [3, 3], [1, 1], [1, 1], [1, 1], [1, 1]]
    fgru_kernels = [[9, 9], [3, 3], [3, 3], [1, 1], [1, 1], [1, 1], [1, 1]]
    fgru_kernels = [[3, 3], [3, 3], [3, 3], [3, 3], [1, 1], [1, 1], [1, 1]]
    # fgru_kernels = [[3, 3], [3, 3], [3, 3], [3, 3], [1, 1], [1, 1], [1, 1]]
    ar = ['']  # , 'fgru_3', 'fgru_4']  # Output layer ids
    return compression, ff_kernels, ff_repeats, features, fgru_kernels, ar


def build_model(
        data_tensor,
        reuse,
        training,
        output_shape,
        data_format='NHWC'):
    """Create the hgru from Learning long-range..."""
    if isinstance(output_shape, list):
        output_shape = output_shape[-1]
    elif isinstance(output_shape, dict):
        output_shape = output_shape['output']
    # norm_moments_training = training  # Force instance norm
    # normalization_type = 'no_param_batch_norm_original'

    normalization_type = 'column_zscore'  # 'no_param_instance_norm'
    # normalization_type = "none"  # "no_param_layer_norm"  # 'no_param_instance_norm'

    output_normalization_type = 'layer_norm'  # 'instance_norm'
    # output_normalization_type = "layer_norm"  # 'instance_norm'

    data_tensor, long_data_format = tf_fun.interpret_data_format(
        data_tensor=data_tensor,
        data_format=data_format)

    # Prepare gammanet structure
    (
        compression,
        ff_kernels,
        ff_repeats,
        features,
        fgru_kernels,
        additional_readouts) = v2_big_working()
    gammanet_constructor = tf_fun.get_gammanet_constructor(
        compression=compression,
        ff_kernels=ff_kernels,
        ff_repeats=ff_repeats,
        features=features,
        fgru_kernels=fgru_kernels)
    aux = get_aux()

    # Build model
    with tf.variable_scope('vgg', reuse=reuse):
        aux = get_aux()
        vgg = vgg16.Vgg16(
            vgg16_npy_path='/media/data_cifs/clicktionary/pretrained_weights/vgg16.npy',
            # vgg16_npy_path='/cifs/data/tserre/CLPS_Serre_Lab/clicktionary/pretrained_weights/vgg16.npy',
            reuse=reuse,
            aux=aux,
            train=training,
            timesteps=8,
            fgru_normalization_type=normalization_type,
            ff_normalization_type=normalization_type)
        vgg(rgb=data_tensor, constructor=gammanet_constructor)
        # activity = vgg.fgru_0

    with tf.variable_scope('fgru', reuse=reuse):
        # Get side weights
        h2_rem = [
            vgg.fgru_0_e]  # vgg.fgru_0_e
        for idx, res in enumerate(h2_rem):
            res = normalization.apply_normalization(
                activity=res,
                name='output_norm1_%s' % idx,
                normalization_type=output_normalization_type,
                data_format=data_format,
                training=training,
                trainable=training,
                reuse=reuse)
            res = aux['image_resize'](
                res,
                data_tensor.get_shape().as_list()[1:3],
                align_corners=True)

        activity = tf.layers.conv2d(
            res,
            filters=output_shape,
            kernel_size=(1, 1),
            padding='same',
            data_format=long_data_format,
            name='out',
            # activation=tf.nn.relu,
            activation=None,
            trainable=training,
            use_bias=True,
            # use_bias=False,
            reuse=reuse)

    if long_data_format is 'channels_first':
        activity = tf.transpose(activity, (0, 2, 3, 1))
    extra_activities = {}  # idx: v for idx, v in enumerate(hs_0)}
    if activity.dtype != tf.float32:
        activity = tf.cast(activity, tf.float32)
    # return [activity, h_deep], extra_activities
    return activity, extra_activities
