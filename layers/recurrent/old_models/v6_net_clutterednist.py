#!/usr/bin/env python
import tensorflow as tf
from layers.feedforward import conv
from layers.feedforward import normalization
from layers.feedforward import pooling

def build_model(data_tensor, reuse, training, output_shape):
    """Create the hgru from Learning long-range..."""
    if isinstance(output_shape, list):
        output_shape = output_shape[0]
    with tf.variable_scope('cnn', reuse=reuse):
        with tf.variable_scope('input', reuse=reuse):
            in_emb = tf.layers.conv2d(
                inputs=data_tensor,
                filters=4,
                kernel_size=7,
                name='l0',
                strides=(1, 1),
                padding='same',
                activation=tf.nn.elu,
                trainable=training,
                use_bias=True)
            in_emb = normalization.batch(
                bottom=in_emb,
                name='l0_bn',
                training=training)
            in_emb = tf.nn.relu(in_emb)
            in_emb = pooling.max_pool(
                bottom=in_emb,
                name='p1',
                k=[1, 2, 2, 1],
                s=[1, 2, 2, 1])
            in_emb = tf.layers.conv2d(
                inputs=in_emb,
                filters=8,
                kernel_size=7,
                name='l1',
                strides=(1, 1),
                padding='same',
                activation=tf.nn.elu,
                trainable=training,
                use_bias=True)
            in_emb = normalization.batch(
                bottom=in_emb,
                name='l1_bn',
                training=training)
            in_emb = tf.nn.relu(in_emb)
            in_emb = pooling.max_pool(
                bottom=in_emb,
                name='p2',
                k=[1, 2, 2, 1],
                s=[1, 2, 2, 1])

        with tf.variable_scope('v6', reuse=reuse):
            from layers.recurrent import v6_net as fgru_net
            from layers.feedforward import v6_ln_mk1 as fgru_layer
            in_shape = in_emb.get_shape().as_list()
            fgru_layer_optional_args = {
                               'swap_mix_sources': False,
                               'swap_gate_sources': False,
                               'turn_off_gates': False,
                               'featurewise_control': False,
                               'no_relu_h1': False}
            layer_hgru = fgru_net.hGRU(var_scope='fgru_net',
                                 timesteps = 6,
                                 in_k = in_shape[-1],
                                 use_global_pool=True,
                                 share_ff_td_kernels=True,
                                 fgru_module_class = fgru_layer,
                                 hgru_fsiz = [5, 5],
                                 hgru_fanout_factor = 3,
                                 hgru_h2_k = 12,
                                 ff_conv_fsiz = [[5, 5], [3, 3], [3, 3]], #from low to high
                                 ff_conv_k = [16, 20, 28],
                                 ff_conv_strides = [[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]],
                                 ff_pool_fsiz = [[2, 2], [2, 2], [2, 2]],
                                 ff_pool_strides = [[2, 2], [2, 2], [2, 2]],
                                 fb_conv_fsiz = [[6, 6], [6, 6], [4, 4], [4, 4]], #from low to high (which means higher layers will be called first during TD phase)
                                 fb_conv_k = [12, 16, 20, 28],
                                 train = True,
                                 dtype = tf.float32,
                                 **fgru_layer_optional_args)
            """
            ####### NOTE ABOUT SWAPPING AN fGRU LAYER: #######
            You can define an fgru module class and feed it to the fgru_net constructor.
            The constructor takes the class and internally constructs two fGRU layers.
            An fgru module class should take the following arguments:
                (<Str: layer name>,
                 <Int: input # channels>,
                 <Int: h1 channel fan-out factor>,
                 <Int: h2 # channels>,
                 <List: filter size>,
                 <Bool: use 3d data>,
                 <Bool: use symmetric kernel>,
                 <Bool: reuse BN params over timesteps>,
                 <Bool: train mode>,
                 <tf.dtype: data type>,
                 **fgru_layer_optional_args)
            As you can see, all the args marked by <...> are automatically defined based on fgru_net arguments.
            All you need to define at this level are the class-specific optional args as a dict.
            """
            bottom, top = layer_hgru.build(in_emb)
            top = normalization.batch(
                bottom=top,
                name='hgru_bn',
                fused=True,
                training=training)

        with tf.variable_scope('readout', reuse=reuse):
            pre_activity = tf.layers.dense(
                inputs=top,
                units=28)
            activity = tf.layers.dense(
                inputs=pre_activity,
                units=output_shape)
    extra_activities = {
        'activity': activity,
    }
    return activity, extra_activities
