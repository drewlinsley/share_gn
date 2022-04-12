import inspect
import os
import numpy as np
import tensorflow as tf
from collections import OrderedDict
from ops import tf_fun
from layers.recurrent.gn_params_minimal import CreateGNParams
from layers.recurrent.gn_params_minimal import defaults
# from layers.recurrent.gammanet_refactored import GN
# from layers.recurrent.gn_recurrent_ops import GNRnOps
from layers.recurrent.gammanet_refactored_alt_minimal import GN
from layers.recurrent.gn_recurrent_ops_alt_bn_sym_minimal import GNRnOps
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
            from_scratch=False,
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
            skip=True,
            aux=None):
        # if vgg16_npy_path is None:
        #     path = inspect.getfile(Vgg16)
        #     path = os.path.abspath(os.path.join(path, os.pardir))
        #     path = os.path.join(path, "vgg16.npy")
        #     vgg16_npy_path = path
        #     print(path)
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
        self.skip = skip
        self.from_scratch = from_scratch
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
            try:
                for k, v in aux.iteritems():
                    default_vars[k] = v
            except:
                for k, v in aux.items():
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
        if isinstance(self.recurrent_nl, str):
            self.recurrent_nl = tf_fun.interpret_nl(self.recurrent_nl)

        # Set initializers for greek letters
        if self.force_alpha_divisive:
            raise NotImplementedError
            self.alpha_initializer = tf.initializers.variance_scaling
        else:
            self.alpha_initializer = tf.random_uniform_initializer(minval=1e-4, maxval=0.1)  # tf.constant_initializer(0.1) 
        self.mu_initializer = tf.random_uniform_initializer(minval=1e-4, maxval=0.1)  #tf.constant_initializer(0.1)
        self.omega_initializer = tf.random_uniform_initializer(minval=1e-4, maxval=0.1)  # tf.constant_initializer(0.1)
        self.kappa_initializer = tf.random_uniform_initializer(minval=1e-4, maxval=0.1)  # tf.constant_initializer(0.1)

        #     self.alpha_initializer = tf.initializers.orthogonal  # tf.constant_initializer(1.)
        # self.mu_initializer = tf.initializers.orthogonal  # tf.constant_initializer(0.1)
        # self.omega_initializer = tf.initializers.orthogonal  # tf.constant_initializer(0.1)
        # self.kappa_initializer = tf.initializers.orthogonal  # tf.constant_initializer(1.)

        # Handle BN scope reuse
        self.scope_reuse = reuse

        # Load weights
        self.data_dict = np.load(vgg16_npy_path, encoding='latin1', allow_pickle=True).item()
        print("npy file loaded")

    def __call__(self, rgb, constructor=None, store_timesteps=False):
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
        self.conv2_1 = self.conv_layer(self.pool1, "conv2_1")
        self.conv2_2 = self.conv_layer(self.conv2_1, "conv2_2")
        X_shape = self.conv2_2.get_shape().as_list()
        self.prepare_tensors(X_shape, allow_resize=False)
        self.create_hidden_states(
            constructor=self.gammanet_constructor,
            shapes=self.layer_shapes,
            recurrent_ff=self.recurrent_ff,
            init=self.hidden_init,
            dtype=self.dtype)
        ta = []
        for idx in range(self.timesteps):
            self.build(i0=idx)
            self.ff_reuse = tf.AUTO_REUSE
            if store_timesteps:
                ta += [self.fgru_0_e]
        if store_timesteps:
            return ta

    def build(self, i0, extra_convs=True, td_reduction={"4_3": 8, "3_2": 4, "2_1": 2}):
        if i0 == 0:
            self.fgru_0_i = self.conv_layer(
                self.conv2_2,
                'i_init_0',
                learned=True,
                apply_relu=True,
                shape=[
                    1,
                    1,
                    self.conv2_2.get_shape().as_list()[-1],
                    self.conv2_2.get_shape().as_list()[-1]])
            self.fgru_0_e = self.conv_layer(
                self.conv2_2,
                'e_init_0',
                learned=True,
                apply_relu=True,
                shape=[
                    1,
                    1,
                    self.conv2_2.get_shape().as_list()[-1],
                    self.conv2_2.get_shape().as_list()[-1]])

        with tf.variable_scope('fgru'):
            self.fgru_0_i, self.fgru_0_e = self.fgru_ops(  # h^(1), h^(2)
                ff_drive=self.conv2_2,
                h1=self.fgru_0_i,
                h2=self.fgru_0_e,
                layer_id=0,
                i0=i0)
        self.fgru_0_e_23 = self.fgru_0_e
        self.pool2 = self.max_pool(self.fgru_0_e, 'pool2')
        self.conv3_1 = self.conv_layer(self.pool2, "conv3_1")
        self.conv3_2 = self.conv_layer(self.conv3_1, "conv3_2")
        self.conv3_3 = self.conv_layer(self.conv3_2, "conv3_3")

        if i0 == 0:
            # self.fgru_1 = self.conv3_3
            self.fgru_1_i = self.conv_layer(
                self.conv3_3,
                'i_init_1',
                learned=True,
                apply_relu=True,
                shape=[
                    1,
                    1,
                    self.conv3_3.get_shape().as_list()[-1],
                    self.conv3_3.get_shape().as_list()[-1]])
            self.fgru_1_e = self.conv_layer(
                self.conv3_3,
                'e_init_1',
                learned=True,
                apply_relu=True,
                shape=[
                    1,
                    1,
                    self.conv3_3.get_shape().as_list()[-1],
                    self.conv3_3.get_shape().as_list()[-1]])
        with tf.variable_scope('fgru'):
            self.fgru_1_i, self.fgru_1_e = self.fgru_ops(  # h^(1), h^(2)
                ff_drive=self.conv3_3,
                h1=self.fgru_1_i,
                h2=self.fgru_1_e,
                layer_id=1,
                i0=i0)
        self.pool3 = self.max_pool(self.fgru_1_e, 'pool3')
        self.conv4_1 = self.conv_layer(self.pool3, "conv4_1")
        self.conv4_2 = self.conv_layer(self.conv4_1, "conv4_2")
        self.conv4_3 = self.conv_layer(self.conv4_2, "conv4_3")

        if i0 == 0:
            self.fgru_2_i = self.conv_layer(
                self.conv4_3,
                'i_init_2',
                learned=True,
                apply_relu=True,
                shape=[
                    1,
                    1,
                    self.conv4_3.get_shape().as_list()[-1],
                    self.conv4_3.get_shape().as_list()[-1]])
            self.fgru_2_e = self.conv_layer(
                self.conv4_3,
                'e_init_2',
                learned=True,
                apply_relu=True,
                shape=[
                    1,
                    1,
                    self.conv4_3.get_shape().as_list()[-1],
                    self.conv4_3.get_shape().as_list()[-1]])
        with tf.variable_scope('fgru'):
            self.fgru_2_i, self.fgru_2_e = self.fgru_ops(  # h^(1), h^(2)
                ff_drive=self.conv4_3,
                h1=self.fgru_2_i,
                h2=self.fgru_2_e,
                layer_id=2,
                i0=i0)
        self.pool4 = self.max_pool(self.fgru_2_e, 'pool4')
        self.conv5_1 = self.conv_layer(self.pool4, "conv5_1")
        self.conv5_2 = self.conv_layer(self.conv5_1, "conv5_2")
        self.conv5_3 = self.conv_layer(self.conv5_2, "conv5_3")
        if i0 == 0:
            # self.fgru_3 = self.conv5_3
            self.fgru_3_i = self.conv_layer(
                self.conv5_3,
                'i_init_3',
                learned=True,
                apply_relu=True,
                shape=[
                    1,
                    1,
                    self.conv5_3.get_shape().as_list()[-1],
                    self.conv5_3.get_shape().as_list()[-1]])
            self.fgru_3_e = self.conv_layer(
                self.conv5_3,
                'e_init_3',
                learned=True,
                apply_relu=True,
                shape=[
                    1,
                    1,
                    self.conv5_3.get_shape().as_list()[-1],
                    self.conv5_3.get_shape().as_list()[-1]])
        with tf.variable_scope('fgru'):
            self.fgru_3_i, self.fgru_3_e = self.fgru_ops(  # h^(1), h^(2)
                ff_drive=self.conv5_3,
                h1=self.fgru_3_i,
                h2=self.fgru_3_e,
                layer_id=3,
                i0=i0)

        # Resize and conv
        with tf.variable_scope('fgru'):
            fgru_2_td = normalization.apply_normalization(
                activity=self.fgru_3_e,
                name='td_norm2_%s' % i0,
                # normalization_type='instance_norm',
                normalization_type=self.fgru_normalization_type,
                data_format=self.data_format,
                training=self.train,
                trainable=self.train,
                reuse=self.reuse)
            fgru_2_td = self.conv_layer(
                fgru_2_td,
                '4_to_3',
                learned=True,
                apply_relu=True,
                shape=[
                    1,
                    1,
                    fgru_2_td.get_shape().as_list()[-1],
                    self.fgru_2.get_shape().as_list()[-1] // td_reduction["4_3"]])
            fgru_2_td = normalization.apply_normalization(
                activity=fgru_2_td,
                name='td_norm2_1%s' % i0,
                # normalization_type='instance_norm',
                normalization_type=self.fgru_normalization_type,
                data_format=self.data_format,
                training=self.train,
                trainable=self.train,
                reuse=self.reuse)
            if extra_convs:
                fgru_2_td = self.conv_layer(
                    fgru_2_td,
                    '4_to_3_2',
                    learned=True,
                    shape=[
                        1,
                        1,
                        self.fgru_2.get_shape().as_list()[-1] // td_reduction["4_3"],
                        self.fgru_2.get_shape().as_list()[-1]])
            fgru_2_td = self.image_resize(
                fgru_2_td,
                self.fgru_2.get_shape().as_list()[1:3],
                align_corners=True)
            if i0 == 0:
                self.fgru_2_td_i = self.conv_layer(
                    fgru_2_td,
                    'i_init_td_2',
                    learned=True,
                    apply_relu=True,
                    shape=[
                        1,
                        1,
                        self.fgru_2.get_shape().as_list()[-1],
                        self.fgru_2.get_shape().as_list()[-1]])
            fgru_2_td = tf.nn.relu(fgru_2_td)
            self.fgru_2_td_i, fgru_2_td = self.fgru_ops(  # h^(1), h^(2)
                ff_drive=self.fgru_2_e,
                h1=self.fgru_2_td_i,
                h2=fgru_2_td,
                layer_id=4,
                i0=i0)
        if self.skip:
            self.fgru_2_e += fgru_2_td
        else:
            self.fgru_2_e = fgru_2_td

        # Resize and conv
        with tf.variable_scope('fgru'):
            fgru_1_td = normalization.apply_normalization(
                activity=self.fgru_2_e,
                name='td_norm1_%s' % i0,
                # normalization_type='instance_norm',
                normalization_type=self.fgru_normalization_type,
                data_format=self.data_format,
                training=self.train,
                trainable=self.train,
                reuse=self.reuse)
            fgru_1_td = self.conv_layer(
                fgru_1_td,
                '3_to_2',
                learned=True,
                apply_relu=True,
                shape=[
                    1,
                    1,
                    fgru_1_td.get_shape().as_list()[-1],
                    self.fgru_1.get_shape().as_list()[-1] // td_reduction["3_2"]])
            fgru_1_td = normalization.apply_normalization(
                activity=fgru_1_td,
                name='td_norm1_1%s' % i0,
                # normalization_type='instance_norm',
                normalization_type=self.fgru_normalization_type,
                data_format=self.data_format,
                training=self.train,
                trainable=self.train,
                reuse=self.reuse)
            if extra_convs:
                fgru_1_td = self.conv_layer(
                    fgru_1_td,
                    '3_to_2_2',
                    learned=True,
                    shape=[
                        1,
                        1,
                        self.fgru_1.get_shape().as_list()[-1] // td_reduction["3_2"],
                        self.fgru_1.get_shape().as_list()[-1]])
            fgru_1_td = self.image_resize(
                fgru_1_td,
                self.fgru_1.get_shape().as_list()[1:3],
                align_corners=True)
            if i0 == 0:
                self.fgru_1_td_i = self.conv_layer(
                    fgru_1_td,
                    'i_init_td_1',
                    learned=True,
                    apply_relu=True,
                    shape=[
                        1,
                        1,
                        self.fgru_1.get_shape().as_list()[-1],
                        self.fgru_1.get_shape().as_list()[-1]])
            fgru_1_td = tf.nn.relu(fgru_1_td)
            self.fgru_1_td_i, fgru_1_td = self.fgru_ops(  # h^(1), h^(2)
                ff_drive=self.fgru_1_e,
                h1=self.fgru_1_td_i,
                h2=fgru_1_td,
                layer_id=5,
                i0=i0)
        if self.skip:
            self.fgru_1_e += fgru_1_td
        else:
            self.fgru_1_e = fgru_1_td

        # Resize and conv
        with tf.variable_scope('fgru'):
            fgru_0_td = normalization.apply_normalization(
                activity=self.fgru_1_e,
                name='td_norm0_%s' % i0,
                # normalization_type='instance_norm',
                normalization_type=self.fgru_normalization_type,
                data_format=self.data_format,
                training=self.train,
                trainable=self.train,
                reuse=self.reuse)
            fgru_0_td = self.conv_layer(
                fgru_0_td,
                '2_to_1',
                learned=True,
                apply_relu=True,
                shape=[
                    1,
                    1,
                    fgru_0_td.get_shape().as_list()[-1],
                    self.fgru_0.get_shape().as_list()[-1] // td_reduction["2_1"]])
            fgru_0_td = normalization.apply_normalization(
                activity=fgru_0_td,
                name='td_norm0_1%s' % i0,
                # normalization_type='instance_norm',
                normalization_type=self.fgru_normalization_type,
                data_format=self.data_format,
                training=self.train,
                trainable=self.train,
                reuse=self.reuse)
            if extra_convs:
                fgru_0_td = self.conv_layer(
                    fgru_0_td,
                    '2_to_1_2',
                    learned=True,
                    shape=[
                        1,
                        1,
                        self.fgru_0.get_shape().as_list()[-1] // td_reduction["2_1"],
                        self.fgru_0.get_shape().as_list()[-1]])
            fgru_0_td = self.image_resize(
                fgru_0_td,
                self.fgru_0.get_shape().as_list()[1:3],
                align_corners=True)
            if i0 == 0:
                self.fgru_0_td_i = self.conv_layer(
                    fgru_0_td,
                    'i_init_td_0',
                    learned=True,
                    apply_relu=True,
                    shape=[
                        1,
                        1,
                        self.fgru_0.get_shape().as_list()[-1],
                        self.fgru_0.get_shape().as_list()[-1]])
            fgru_0_td = tf.nn.relu(fgru_0_td)
            self.fgru_0_td_i, fgru_0_td = self.fgru_ops(  # h^(1), h^(2)
                ff_drive=self.fgru_0_e,
                h1=self.fgru_0_td_i,
                h2=fgru_0_td,
                layer_id=6,
                i0=i0)
        if self.skip:
            self.fgru_0_e += fgru_0_td
        else:
            self.fgru_0_e = fgru_0_td

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
                    initializer=tf.initializers.variance_scaling)
            elif self.from_scratch:
                return tf.get_variable(
                    name='%s_kernel' % name,
                    shape=self.data_dict[name][0].shape,
                    dtype=self.dtype,
                    trainable=self.train,
                    initializer=tf.initializers.variance_scaling)
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
            elif self.from_scratch:
                return tf.get_variable(
                    name='%s_bias' % name,
                    shape=self.data_dict[name][1].shape,
                    dtype=self.dtype,
                    trainable=self.train,
                    initializer=tf.initializers.zeros)
            else:
                return tf.get_variable(
                    name='%s_bias' % name,
                    initializer=self.data_dict[name][1],
                    trainable=self.train)
