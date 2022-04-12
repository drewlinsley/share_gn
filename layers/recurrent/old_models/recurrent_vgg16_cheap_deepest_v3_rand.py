import inspect
import os
import numpy as np
import tensorflow as tf
from collections import OrderedDict
from ops import tf_fun
from layers.recurrent.gn_params import CreateGNParams
from layers.recurrent.gn_params import defaults
from layers.recurrent.gammanet_refactored import GN
from layers.recurrent.gn_recurrent_ops import GNRnOps
# from layers.recurrent.gammanet_refactored_alt import GN
# from layers.recurrent.gn_recurrent_ops_alt_bn import GNRnOps
from layers.recurrent.gn_feedforward_ops import GNFFOps
from layers.feedforward import normalization


class Vgg16(GN, CreateGNParams, GNRnOps, GNFFOps):
    def __init__(
            self,
            vgg16_npy_path,
            train,
            timesteps,
            reuse,
            fgru_normalization_type,
            ff_normalization_type,
            layer_name='recurrent_vgg16',
            ff_nl=tf.nn.relu,
            horizontal_kernel_initializer=tf.initializers.orthogonal(),
            # horizontal_kernel_initializer=tf_fun.Identity(),
            kernel_initializer=tf.initializers.orthogonal(),
            gate_initializer=tf.initializers.orthogonal(),
            train_ff_gate=None,
            train_fgru_gate=None,
            train_norm_moments=None,
            train_norm_params=None,
            train_fgru_kernels=None,
            train_fgru_params=None,
            up_kernel=None,
            stop_loop=False,
            recurrent_ff=False,
            strides=[1, 1, 1, 1],
            pool_strides=[2, 2],  # Because fgrus are every other down-layer
            pool_kernel=[2, 2],
            data_format='NHWC',
            horizontal_padding='SAME',
            ff_padding='SAME',
            vgg_dtype=tf.bfloat16,
            aux=None):
        if vgg16_npy_path is None:
            path = inspect.getfile(Vgg16)
            path = os.path.abspath(os.path.join(path, os.pardir))
            path = os.path.join(path, "vgg16.npy")
            vgg16_npy_path = path
            print path
        self.data_format = data_format
        self.pool_strides = pool_strides
        self.strides = strides
        self.pool_kernel = pool_kernel
        self.fgru_normalization_type = fgru_normalization_type
        self.ff_normalization_type = ff_normalization_type
        self.horizontal_padding = horizontal_padding
        self.ff_padding = ff_padding
        self.train = train
        self.layer_name = layer_name
        self.data_format = data_format
        self.horizontal_kernel_initializer = horizontal_kernel_initializer
        self.kernel_initializer = kernel_initializer
        self.gate_initializer = gate_initializer
        self.fgru_normalization_type = fgru_normalization_type
        self.ff_normalization_type = ff_normalization_type
        self.recurrent_ff = recurrent_ff
        self.stop_loop = stop_loop
        self.ff_nl = ff_nl
        self.fgru_connectivity = ''
        self.reuse = reuse
        self.timesteps = timesteps
        if train_ff_gate is None:
            self.train_ff_gate = self.train
        else:
            self.train_ff_gate = train_ff_gate
        if train_fgru_gate is None:
            self.train_fgru_gate = self.train
        else:
            self.train_fgru_gate = train_fgru_gate
        if train_norm_moments is None:
            self.train_norm_moments = self.train
        else:
            self.train_norm_moments = train_norm_moments
        if train_norm_moments is None:
            self.train_norm_params = self.train
        else:
            self.train_norm_params = train_norm_params
        if train_fgru_kernels is None:
            self.train_fgru_kernels = self.train
        else:
            self.train_fgru_kernels = train_fgru_kernels
        if train_fgru_kernels is None:
            self.train_fgru_params = self.train
        else:
            self.train_fgru_params = train_fgru_params

        default_vars = defaults()
        if aux is not None and isinstance(aux, dict):
            for k, v in aux.iteritems():
                default_vars[k] = v
        self.update_params(default_vars)
        # Store variables in the order they were created. Hack for python 2.x.
        self.variable_list = OrderedDict()
        self.hidden_dict = OrderedDict()

        # Kernel info
        if data_format is 'NHWC':
            self.prepared_pool_kernel = [1] + self.pool_kernel + [1]
            self.prepared_pool_stride = [1] + self.pool_strides + [1]
            self.up_strides = [1] + self.pool_strides + [1]
        else:
            raise NotImplementedError
            self.prepared_pool_kernel = [1, 1] + self.pool_kernel
            self.prepared_pool_stride = [1, 1] + self.pool_strides
            self.up_strides = [1, 1] + self.pool_strides
        self.sanity_check()
        if self.symmetric_weights:
            self.symmetric_weights = self.symmetric_weights.split('_')

        # Nonlinearities and initializations
        if isinstance(self.recurrent_nl, basestring):
            self.recurrent_nl = tf_fun.interpret_nl(self.recurrent_nl)

        # Set initializers for greek letters
        if self.force_alpha_divisive:
            self.alpha_initializer = tf.initializers.variance_scaling
        else:
            self.alpha_initializer = tf.constant_initializer(1.)  # tf.zeros_initializer
        self.mu_initializer = tf.zeros_initializer
        self.omega_initializer = tf.constant_initializer(0.5)  # tf.zeros_initializer
        self.kappa_initializer = tf.constant_initializer(0.5)  # tf.zeros_initializer

        # Handle BN scope reuse
        self.scope_reuse = reuse

        # Load weights
        self.data_dict = np.load(vgg16_npy_path, encoding='latin1').item()
        print("npy file loaded")

    def create_bn_params(self, constructor):
        """Create shared bn params."""
        idx = [True if v['compression'] == 'upsample' else False for k, v in constructor.iteritems()].index(True) - 1
        idx = np.arange(idx, len(constructor) - 1)
        cons = {k: v for k, v in constructor.iteritems() if k in idx}
        max_id = np.max(cons.keys())
        for k, v in cons.iteritems():
            var_id = max_id - k
            features = v['features']
            gamma = tf.get_variable(
                name='vgg_gamma_1x1_%s' % var_id,
                shape=[1, 1, 1, features],
                dtype=self.dtype,
                trainable=self.train,
                initializer=tf.constant_initializer(0.1))
            beta = tf.get_variable(
                name='vgg_beta_1x1_%s' % var_id,
                shape=[1, 1, 1, features],
                dtype=self.dtype,
                trainable=self.train,
                initializer=tf.constant_initializer(0.0))
            setattr(self, 'vgg_fgru_%s_td_gamma' % var_id, gamma) 
            setattr(self, 'vgg_fgru_%s_td_beta' % var_id, beta)

    def __call__(self, rgb, constructor=None):
        """
        load variable from npy to build the VGG

        :param rgb: rgb image [batch, height, width, 3] values scaled [0, 1]
        """
        self.gammanet_constructor = constructor
        X_shape = rgb.get_shape().as_list()
        self.N = X_shape[0]
        self.dtype = rgb.dtype
        self.input = rgb
        self.ff_reuse = self.scope_reuse
        self.conv1_1 = self.conv_layer(self.input, "conv1_1")
        self.conv1_2 = self.conv_layer(self.conv1_1, "conv1_2")
        self.pool1 = self.max_pool(self.conv1_2, 'pool1')
        X_shape = self.pool1.get_shape().as_list()
        self.prepare_tensors(X_shape, allow_resize=False)
        self.create_hidden_states(
            constructor=self.gammanet_constructor,
            shapes=self.layer_shapes,
            recurrent_ff=self.recurrent_ff,
            init=self.hidden_init,
            dtype=self.dtype)
        # self.create_bn_params(constructor)
        self.fgru_0 = self.pool1
        for idx in range(self.timesteps):
            self.build(i0=idx)
            self.ff_reuse = tf.AUTO_REUSE

    def build(self, i0, extra_convs=True, conv_nl=tf.nn.relu):
        # Convert RGB to BGR
        with tf.variable_scope('fgru'):
            error, fgru_activity = self.fgru_ops(  # h^(1), h^(2)
                ff_drive=self.pool1,
                h2=self.fgru_0,
                layer_id=0,
                i0=i0)
        self.fgru_0 = fgru_activity  # + self.conv2_2
        self.conv2_1 = self.conv_layer(self.fgru_0, "conv2_1")
        self.conv2_2 = self.conv_layer(self.conv2_1, "conv2_2")
        self.pool2 = self.max_pool(self.conv2_2, 'pool2')
        if i0 == 0:
            self.fgru_1 = tf.zeros_like(self.pool2)
        with tf.variable_scope('fgru'):
            error, fgru_activity = self.fgru_ops(  # h^(1), h^(2)
                ff_drive=self.pool2,
                h2=self.fgru_1,
                layer_id=1,
                i0=i0)
        self.fgru_1 = fgru_activity  # + self.conv2_2

        self.conv3_1 = self.conv_layer(self.fgru_1, "conv3_1")
        self.conv3_2 = self.conv_layer(self.conv3_1, "conv3_2")
        self.conv3_3 = self.conv_layer(self.conv3_2, "conv3_3")
        self.pool3 = self.max_pool(self.conv3_3, 'pool3')
        if i0 == 0:
            self.fgru_2 = tf.zeros_like(self.pool3)
        with tf.variable_scope('fgru'):
            error, fgru_activity = self.fgru_ops(  # h^(1), h^(2)
                ff_drive=self.pool3,
                h2=self.fgru_2,
                layer_id=2,
                i0=i0)
        self.fgru_2 = fgru_activity  # + self.conv3_3

        self.conv4_1 = self.conv_layer(self.fgru_2, "conv4_1")
        self.conv4_2 = self.conv_layer(self.conv4_1, "conv4_2")
        self.conv4_3 = self.conv_layer(self.conv4_2, "conv4_3")
        self.pool4 = self.max_pool(self.conv4_3, 'pool4')
        self.conv5_1 = self.conv_layer(self.pool4, "conv5_1")
        self.conv5_2 = self.conv_layer(self.conv5_1, "conv5_2")
        self.conv5_3 = self.conv_layer(self.conv5_2, "conv5_3")
        if i0 == 0:
            self.fgru_3 = tf.zeros_like(self.conv5_3)
        with tf.variable_scope('fgru'):
            error, fgru_activity = self.fgru_ops(  # h^(1), h^(2)
                ff_drive=self.conv5_3,
                h2=self.fgru_3,
                layer_id=3,
                i0=i0)
        self.fgru_3 = fgru_activity  # + self.conv5_3

        # Resize and conv
        with tf.variable_scope('fgru'):
            fgru_2_td = fgru_activity
            # fgru_2_td = normalization.apply_normalization(
            #     activity=fgru_2_td,
            #     name='td_norm2_%s' % i0,
            #     normalization_type=self.fgru_normalization_type,
            #     data_format=self.data_format,
            #     training=self.train,
            #     trainable=self.train,
            #     reuse=self.reuse)
            # fgru_2_td = self.vgg_fgru_2_td_gamma * fgru_2_td + self.vgg_fgru_2_td_beta
            fgru_2_td = self.image_resize(
                fgru_2_td,
                self.fgru_2.get_shape().as_list()[1:3],
                align_corners=True)
            fgru_2_td = self.conv_layer(
                fgru_2_td,
                '4_to_3',
                learned=True,
                apply_relu=False,
                shape=[
                    1,
                    1,
                    fgru_2_td.get_shape().as_list()[-1],
                    self.fgru_2.get_shape().as_list()[-1] // 64])
            fgru_2_td = conv_nl(fgru_2_td)
            if extra_convs:
                fgru_2_td = self.conv_layer(
                    fgru_2_td,
                    '4_to_3_2',
                    learned=True,
                    apply_relu=False,
                    shape=[
                        1,
                        1,
                        self.fgru_2.get_shape().as_list()[-1] // 64,
                        self.fgru_2.get_shape().as_list()[-1]])
            error, fgru_activity = self.fgru_ops(  # h^(1), h^(2)
                ff_drive=self.fgru_2,
                h2=fgru_2_td,
                layer_id=4,
                i0=i0)
        self.fgru_2 = fgru_activity
        # self.fgru_2 += fgru_activity

        # Resize and conv
        with tf.variable_scope('fgru'):
            fgru_1_td = fgru_activity
            # fgru_1_td = normalization.apply_normalization(
            #     activity=fgru_1_td,
            #     name='td_norm1_%s' % i0,
            #     normalization_type=self.fgru_normalization_type,
            #     data_format=self.data_format,
            #     training=self.train,
            #     trainable=self.train,
            #     reuse=self.reuse)
            # fgru_1_td = self.vgg_fgru_1_td_gamma * fgru_1_td + self.vgg_fgru_1_td_beta
            fgru_1_td = self.image_resize(
                fgru_1_td,
                self.fgru_1.get_shape().as_list()[1:3],
                align_corners=True)
            fgru_1_td = self.conv_layer(
                fgru_1_td,
                '3_to_2',
                learned=True,
                apply_relu=False,
                shape=[
                    1,
                    1,
                    fgru_1_td.get_shape().as_list()[-1],
                    self.fgru_1.get_shape().as_list()[-1] // 32])
            fgru_1_td = conv_nl(fgru_1_td)
            # fgru_1_td = tf.nn.elu(fgru_1_td)
            if extra_convs:
                fgru_1_td = self.conv_layer(
                    fgru_1_td,
                    '3_to_2_2',
                    learned=True,
                    apply_relu=False,
                    shape=[
                        1,
                        1,
                        self.fgru_1.get_shape().as_list()[-1] // 32,
                        self.fgru_1.get_shape().as_list()[-1]])
            error, fgru_activity = self.fgru_ops(  # h^(1), h^(2)
                ff_drive=self.fgru_1,
                h2=fgru_1_td,
                layer_id=5,
                i0=i0)
        self.fgru_1 = fgru_activity
        # self.fgru_1 += fgru_activity

        # Resize and conv
        with tf.variable_scope('fgru'):
            fgru_0_td = fgru_activity
            # fgru_0_td = normalization.apply_normalization(
            #     activity=fgru_0_td,
            #     name='td_norm0_%s' % i0,
            #     normalization_type=self.fgru_normalization_type,
            #     data_format=self.data_format,
            #     training=self.train,
            #     trainable=self.train,
            #     reuse=self.reuse)
            # fgru_0_td = self.vgg_fgru_0_td_gamma * fgru_0_td + self.vgg_fgru_0_td_beta
            fgru_0_td = self.image_resize(
                fgru_0_td,
                self.fgru_0.get_shape().as_list()[1:3],
                align_corners=True)
            fgru_0_td = self.conv_layer(
                fgru_0_td,
                '2_to_1',
                learned=True,
                apply_relu=False,
                shape=[
                    1,
                    1,
                    fgru_0_td.get_shape().as_list()[-1],
                    self.fgru_0.get_shape().as_list()[-1] // 16])
            fgru_0_td = conv_nl(fgru_0_td)
            if extra_convs:
                fgru_0_td = self.conv_layer(
                    fgru_0_td,
                    '2_to_1_2',
                    learned=True,
                    apply_relu=False,
                    shape=[
                        1,
                        1,
                        self.fgru_0.get_shape().as_list()[-1] // 16,
                        self.fgru_0.get_shape().as_list()[-1]])
            error, fgru_activity = self.fgru_ops(  # h^(1), h^(2)
                ff_drive=self.fgru_0,
                h2=fgru_0_td,
                layer_id=6,
                i0=i0)
        self.fgru_0 = fgru_activity
        # self.fgru_0 += fgru_activity

    def max_pool(self, bottom, name):
        return tf.nn.max_pool(
            bottom,
            ksize=[1, 2, 2, 1],
            strides=[1, 2, 2, 1],
            padding='SAME',
            name=name)

    def conv_layer(
            self,
            bottom,
            name,
            learned=False,
            shape=False,
            apply_relu=True):
        with tf.variable_scope(name):
            filt = self.get_conv_filter(name, learned=learned, shape=shape)
            conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='SAME')

            conv_biases = self.get_bias(name, learned=learned, shape=shape)
            bias = tf.nn.bias_add(conv, conv_biases)

            if apply_relu:
                relu = tf.nn.relu(bias)
            else:
                relu = bias
            return relu

    def get_conv_filter(self, name, learned=False, shape=None):
        with tf.variable_scope('ff_vars', reuse=self.ff_reuse):
            if learned:
                return tf.get_variable(
                    name='%s_kernel' % name,
                    shape=shape,
                    dtype=self.dtype,
                    trainable=self.train,
                    initializer=self.kernel_initializer)
            else:
                return tf.get_variable(
                    name='%s_kernel' % name,
                    initializer=self.data_dict[name][0],
                    trainable=self.train)

    def get_bias(self, name, learned=False, shape=None):
        with tf.variable_scope('ff_vars', reuse=self.ff_reuse):
            if learned:
                return tf.get_variable(
                    name='%s_bias' % name,
                    shape=[shape[-1]],
                    dtype=self.dtype,
                    trainable=self.train,
                    initializer=tf.initializers.zeros)
            else:
                return tf.get_variable(
                    name='%s_bias' % name,
                    initializer=self.data_dict[name][1],
                    trainable=self.train)

