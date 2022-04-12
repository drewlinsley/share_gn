"""Contextual model with partial filters."""
import numpy as np
import scipy as sp
import tensorflow as tf
from scipy import stats
from ops import initialization


def _sgw(k, s):
    """ Shifted histogram of Gaussian weights, centered appropriately """
    x = sp.linspace(0.0, 1.0, k)
    if s == sp.inf:
        w = sp.ones((k,)) / float(k)
    else:
        w = stats.norm.pdf(x, loc=x[k // 2], scale=s)
    return sp.roll(w / w.sum(), shift=int(sp.ceil(k / 2.0)))


def _sdw(k, s):
    """ Shifted histogram of discontinuous weights, centered appropriately """
    g1 = _sgw(k=k, s=s).max()
    g2 = (1.0 - g1) / (k - 1)
    return sp.array([g1] + [g2] * (k - 1))


class hGRU(object):
    def __getitem__(self, name):
        return getattr(self, name)

    def __contains__(self, name):
        return hasattr(self, name)

    def __init__(
            self,
            layer_name,
            x_shape,
            timesteps=8,
            is_continuous=False,
            train_channels=True,
            train_space=False,
            crf=5,
            omega=0.15,
            near_surround='auto',
            far_surround='auto',
            h_ext=1,
            strides=[1, 1, 1, 1],
            padding='SAME',
            aux=None,
            separate_convs=True,
            hidden_states='gru',  # gru or mely
            data_format='NHWC',
            train=True):
        """Global initializations and settings."""
        if data_format == 'NHWC':
            self.n, self.h, self.w, self.k = x_shape
            self.bias_shape = [1, 1, 1, self.k]
        elif data_format == 'NCHW':
            self.n, self.k, self.h, self.w = x_shape
            self.bias_shape = [1, self.k, 1, 1]
        else:
            raise NotImplementedError(data_format)
        self.timesteps = timesteps
        self.strides = strides
        self.padding = padding
        self.train = train
        self.layer_name = layer_name
        self.data_format = data_format
        self.is_continuous = is_continuous
        self.omega = omega
        self.train_channels = train_channels
        self.train_space = train_space
        self.separate_convs = separate_convs
        self.hidden_states = hidden_states

        # Sort through and assign the auxilliary variables
        default_vars = self.defaults()
        if aux is not None and isinstance(aux, dict):
            for k, v in aux.iteritems():
                default_vars[k] = v
        self.update_params(default_vars)

        # Kernel shapes
        assert h_ext == 1, 'Model requires 1x1 learnable h_ext.'
        self.h_ext = h_ext
        self.h_shape = [self.h_ext, self.h_ext, self.k, self.k]
        self.g_shape = [self.gate_filter, self.gate_filter, self.k, self.k]
        self.m_shape = [self.gate_filter, self.gate_filter, self.k, self.k]
        self.crf = crf
        assert type(crf) is int, 'crf must be an integer'
        if near_surround == 'auto':
            self.near_surround = crf * 2
        if far_surround == 'auto':
            self.far_surround = int(round(crf * 5.5))

        # Nonlinearities and initializations
        if isinstance(self.recurrent_nl, basestring):
            self.recurrent_nl = self.interpret_nl(self.recurrent_nl)

        # Set integration operations
        self.ii, self.oi = self.input_integration, self.output_integration

        # Handle BN scope reuse
        if self.reuse:
            self.scope_reuse = tf.AUTO_REUSE
        else:
            self.scope_reuse = None
        self.param_initializer = {
            'moving_mean': tf.constant_initializer(0.),
            'moving_variance': tf.constant_initializer(1.),
            'gamma': tf.constant_initializer(0.1)
        }
        self.param_trainable = {
            'moving_mean': False,
            'moving_variance': False,
            'gamma': True
        }
        self.param_collections = {
            'moving_mean': None,  # [tf.GraphKeys.UPDATE_OPS],
            'moving_variance': None,  # [tf.GraphKeys.UPDATE_OPS],
            'gamma': None
        }
        self.kernel_initializer = tf.initializers.variance_scaling

    def defaults(self):
        """A dictionary containing defaults for auxilliary variables.

        These are adjusted by a passed aux dict variable."""
        return {
            'lesion_alpha': False,
            'lesion_mu': False,
            'lesion_omega': False,
            'lesion_kappa': False,
            'lesion_beta': False,
            'lesion_nu': False,
            'lesion_xi': False,
            'dtype': tf.float32,
            'np_dtype': np.float32,
            'hidden_init': 'zeros',
            'gate_bias_init': 'chronos',
            # 'train': True,
            'while_loop': False,
            'recurrent_nl': tf.nn.relu,
            'gate_nl': tf.nn.sigmoid,
            'normal_initializer': False,
            'symmetric_weights': True,
            'symmetric_gate_weights': False,
            'symmetric_init': True,
            'nonnegative': True,
            'gate_filter': 1,  # Gate kernel size
            'gamma': True,  # Scale P
            'alpha': True,  # divisive eCRF
            'mu': True,  # subtractive eCRF
            'beta': True,
            'nu': True,
            'xi': True,
            'adaptation': False,
            'multiplicative_excitation': True,
            'horizontal_dilations': [1, 1, 1, 1],
            'reuse': False,
            'constrain': False  # Constrain greek letters to be +
        }

    def interpret_nl(self, nl_type):
        """Return activation function."""
        if nl_type == 'tanh':
            return tf.nn.tanh
        elif nl_type == 'relu':
            return tf.nn.relu
        elif nl_type == 'selu':
            return tf.nn.selu
        elif nl_type == 'leaky_relu':
            return tf.nn.leaky_relu
        elif nl_type == 'sigmoid':
            return tf.sigmoid
        elif nl_type == 'hard_tanh':
            return lambda z: tf.maximum(tf.minimum(z, 1), 0)
        elif nl_type == 'relu6':
            return tf.nn.relu6
        else:
            raise NotImplementedError(nl_type)

    def update_params(self, kwargs):
        """Update the class attributes with kwargs."""
        if kwargs is not None:
            for k, v in kwargs.iteritems():
                setattr(self, k, v)

    def symmetric_initializer(self, w):
        """Initialize symmetric weight sharing."""
        return 0.5 * (w + tf.transpose(w, (0, 1, 3, 2)))

    def prepare_tensors(self):
        """ Prepare recurrent/forward weight matrices.
        (np.prod([h, w, k]) / 2) - k params in the surround filter
        """
        if self.constrain:
            constraint = lambda x: tf.clip_by_value(x, 0, np.infty)
        else:
            constraint = None
        self.var_scope = '%s_hgru_weights' % self.layer_name
        with tf.variable_scope(self.var_scope):
            # Create separate channel kernels for U/T/Q/P
            q_array = _sgw(k=self.k, s=self.omega) \
                if self.is_continuous else _sdw(k=self.k, s=self.omega)
            q_array = sp.array(
                [sp.roll(q_array, shift=shift) for shift in range(self.k)])
            if self.separate_convs:
                self.Q_sp = tf.get_variable(
                    name='Q_sp',
                    dtype=self.dtype,
                    initializer=(
                        q_array * np.ones((
                            self.crf, self.crf, 1, 1))
                    ).astype(self.np_dtype)[..., [0]],
                    trainable=self.train_space)
                self.Q_ch = tf.get_variable(
                    name='Q_ch',
                    dtype=self.dtype,
                    initializer=q_array.astype(
                        self.np_dtype)[None, None, :, :],
                    trainable=self.train_channels)
            else:
                self.Q_ch = tf.get_variable(
                    name='Q_ch',
                    dtype=self.dtype,
                    initializer=(q_array * np.ones((
                        self.crf, self.crf, 1, 1))).astype(self.np_dtype),
                    trainable=self.train_channels)
            u_array = (
                1.0 / self.k * np.ones((1, 1, self.k, self.k))).astype(
                    self.np_dtype)
            if self.separate_convs:
                self.U_sp = tf.get_variable(
                    name='U_sp',
                    dtype=self.dtype,
                    initializer=(
                        u_array * np.ones((
                            self.crf, self.crf, 1, 1))
                    ).astype(self.np_dtype)[..., [0]],
                    trainable=self.train_space)
                self.U_ch = tf.get_variable(
                    name='U_ch',
                    dtype=self.dtype,
                    initializer=u_array.astype(self.np_dtype),
                    trainable=self.train_channels)
            else:
                self.U_ch = tf.get_variable(
                    name='U_ch',
                    dtype=self.dtype,
                    initializer=(u_array * np.ones((
                        self.crf, self.crf, 1, 1))).astype(self.np_dtype),
                    trainable=self.train_channels)

            # eCRFs
            SSN_ = 2 * np.floor(self.near_surround / 2.0).astype(int) + 1
            p_array = np.zeros((self.k, self.k, SSN_, SSN_))

            # Uniform weights
            for pdx in range(self.k):
                p_array[
                    pdx,
                    pdx,
                    :self.near_surround + 1,
                    :self.near_surround + 1] = 1.
            half_near = self.near_surround // 2
            floor_classic = np.floor(self.crf / 2.0).astype(int)
            ceil_classic = np.ceil(self.crf / 2.0).astype(int)
            p_array[
                :, :,  # exclude classical receptive field!
                half_near - floor_classic:half_near + ceil_classic,
                half_near - floor_classic:half_near + ceil_classic] = 0.

            # normalize to get true average
            p_array /= self.near_surround**2 - self.crf**2

            # Tf dimension reordering
            p_array = p_array.transpose(2, 3, 0, 1)
            if self.separate_convs:
                self.P_sp = tf.get_variable(
                    name='P_sp',
                    dtype=self.dtype,
                    initializer=(
                        p_array[:, :, 0, 0][:, :, None, None] * np.ones((
                            SSN_, SSN_, self.k, 1))).astype(
                                self.np_dtype),
                    trainable=self.train_space)
                self.P_ch = tf.get_variable(
                    name='P_ch',
                    dtype=self.dtype,
                    initializer=(p_array[0, 0, :, :][None, None]).astype(
                        self.np_dtype),
                    trainable=self.train_channels)
            else:
                self.P_ch = tf.get_variable(
                    name='P_ch',
                    dtype=self.dtype,
                    initializer=p_array.astype(self.np_dtype),
                    trainable=self.train_channels)

            # T
            SSF_ = 2 * np.floor(self.far_surround / 2.0).astype(int) + 1
            t_array = sp.zeros((self.k, self.k, SSF_, SSF_))

            # Uniform weights
            for tdx in range(self.k):
                t_array[
                    tdx,
                    tdx,
                    :self.far_surround + 1,
                    :self.far_surround + 1] = 1.0

            half_far = self.far_surround // 2
            floor_near = np.floor(self.near_surround / 2.0).astype(int)
            ceil_near = np.ceil(self.near_surround / 2.0).astype(int)
            t_array[
                :,
                :,  # exclude near surround!
                half_far - floor_near:half_far + ceil_near + 1,
                half_far - floor_near:half_far + ceil_near + 1] = 0.0

            # normalize to get true average
            t_array /= self.far_surround ** 2 - self.near_surround ** 2

            # Tf dimension reordering
            t_array = t_array.transpose(2, 3, 0, 1)
            if self.separate_convs:
                self.T_sp = tf.get_variable(
                    name='T_sp',
                    dtype=self.dtype,
                    initializer=(
                        t_array[:, :, 0, 0][:, :, None, None] * np.ones((
                            SSF_, SSF_, self.k, 1))).astype(
                                self.np_dtype),
                    trainable=self.train_space)
                self.T_ch = tf.get_variable(
                    name='T_ch',
                    dtype=self.dtype,
                    initializer=(t_array[0, 0, :, :][None, None]).astype(
                        self.np_dtype),
                    trainable=self.train_channels)
            else:
                self.T_ch = tf.get_variable(
                    name='T_ch',
                    dtype=self.dtype,
                    initializer=t_array.astype(self.np_dtype),
                    trainable=self.train_channels)

            # Create gate kernels
            self.gain_kernels = tf.get_variable(
                name='%s_gain' % self.layer_name,
                dtype=self.dtype,
                shape=self.g_shape,
                initializer=tf.initializers.variance_scaling(),
                trainable=self.train)
            self.mix_kernels = tf.get_variable(
                name='%s_mix' % self.layer_name,
                dtype=self.dtype,
                shape=self.m_shape,
                initializer=tf.initializers.variance_scaling(),
                trainable=self.train)
            if self.symmetric_gate_weights and self.symmetric_inits:
                self.gain_kernels = self.symmetric_init(self.gain_kernels)
                self.mix_kernels = self.symmetric_init(self.mix_kernels)

            # Gain bias
            if self.gate_bias_init == 'chronos':
                bias_init = -tf.log(
                    tf.random_uniform(
                        self.bias_shape, minval=1, maxval=self.timesteps - 1))
            else:
                bias_init = -tf.ones(self.bias_shape)
            self.gain_bias = tf.get_variable(
                name='%s_gain_bias' % self.layer_name,
                dtype=self.dtype,
                trainable=self.train,
                initializer=bias_init)
            if self.gate_bias_init == 'chronos':
                bias_init = -bias_init
            else:
                bias_init = tf.ones(self.bias_shape)
            self.mix_bias = tf.get_variable(
                name='%s_mix_bias' % self.layer_name,
                dtype=self.dtype,
                trainable=self.train,
                initializer=bias_init)

            # Divisive params
            if self.alpha and not self.lesion_alpha:
                self.alpha = tf.get_variable(
                    name='%s_alpha' % self.layer_name,
                    constraint=constraint,
                    shape=self.bias_shape,
                    initializer=tf.initializers.variance_scaling(),
                    trainable=self.train)
            elif self.lesion_alpha:
                self.alpha = tf.constant(0.)
            else:
                self.alpha = tf.constant(1.)
            if self.beta and not self.lesion_beta:
                self.beta = tf.get_variable(
                    name='%s_beta' % self.layer_name,
                    constraint=constraint,
                    shape=self.bias_shape,
                    initializer=tf.initializers.variance_scaling(),
                    trainable=self.train)
            elif self.lesion_alpha:
                self.beta = tf.constant(0.)
            else:
                self.beta = tf.constant(1.)

            if self.mu and not self.lesion_mu:
                self.mu = tf.get_variable(
                    name='%s_mu' % self.layer_name,
                    constraint=constraint,
                    shape=self.bias_shape,
                    initializer=tf.initializers.variance_scaling(),
                    trainable=self.train)
            if self.nu and not self.lesion_nu:
                self.nu = tf.get_variable(
                    name='%s_nu' % self.layer_name,
                    constraint=constraint,
                    shape=self.bias_shape,
                    initializer=tf.initializers.variance_scaling(),
                    trainable=self.train)

            elif self.lesion_mu:
                self.mu = tf.constant(0.)
            else:
                self.mu = tf.constant(1.)

            if self.gamma:
                self.gamma = tf.get_variable(
                    name='%s_gamma' % self.layer_name,
                    constraint=constraint,
                    shape=self.bias_shape,
                    initializer=tf.initializers.variance_scaling(),
                    trainable=self.train)
            else:
                self.gamma = tf.constant(1.)
            if self.xi:
                self.xi = tf.get_variable(
                    name='%s_xi' % self.layer_name,
                    constraint=constraint,
                    shape=self.bias_shape,
                    initializer=tf.initializers.variance_scaling(),
                    trainable=self.train)
            else:
                self.xi = tf.constant(1.)

            if self.multiplicative_excitation:
                if self.lesion_kappa:
                    self.kappa = tf.constant(0.)
                else:
                    self.kappa = tf.get_variable(
                        name='%s_kappa' % self.layer_name,
                        constraint=constraint,
                        shape=self.bias_shape,
                        initializer=tf.initializers.variance_scaling(),
                        trainable=self.train)

                if self.lesion_omega:
                    self.omega = tf.constant(0.)
                else:
                    self.omega = tf.get_variable(
                        name='%s_omega' % self.layer_name,
                        constraint=constraint,
                        shape=self.bias_shape,
                        initializer=tf.initializers.variance_scaling(),
                        trainable=self.train)

            else:
                self.kappa = tf.constant(1.)
                self.omega = tf.constant(1.)

            if self.adaptation:
                self.eta = tf.get_variable(
                    trainable=self.train,
                    name='%s_eta' % self.layer_name,
                    shape=[self.timesteps],
                    initializer=tf.random_uniform_initializer)
            if self.lesion_omega:
                self.omega = tf.constant(0.)
            if self.lesion_kappa:
                self.kappa = tf.constant(0.)
            if self.reuse:
                # Make the batchnorm variables
                scopes = ['g1_bn', 'g2_bn', 'c1_bn', 'c2_bn']
                bn_vars = ['moving_mean', 'moving_variance', 'gamma']
                for s in scopes:
                    with tf.variable_scope(s):
                        for v in bn_vars:
                            tf.get_variable(
                                trainable=self.param_trainable[v],
                                name=v,
                                shape=[self.k],
                                collections=self.param_collections[v],
                                initializer=self.param_initializer[v])
                self.param_initializer = None

    def conv_2d_op(
            self,
            data,
            weights,
            dilations=[1, 1, 1, 1],
            symmetric_weights=False):
        """2D convolutions for hgru."""
        w_shape = [int(w) for w in weights.get_shape()]
        if len(w_shape) > 1 and int(w_shape[-2]) > 1:
            # Full convolutions
            if symmetric_weights:
                g = tf.get_default_graph()
                with g.gradient_override_map({'Conv2D': 'SymmetricConv'}):
                    activities = tf.nn.conv2d(
                        data,
                        weights,
                        self.strides,
                        dilations=dilations,
                        data_format=self.data_format,
                        padding=self.padding)
            else:
                activities = tf.nn.conv2d(
                    data,
                    weights,
                    self.strides,
                    dilations=dilations,
                    data_format=self.data_format,
                    padding=self.padding)
        else:
            raise RuntimeError
        return activities

    def circuit_input(self, var_scope, h2):
        """Calculate gain and inh horizontal activities."""
        g1_intermediate = self.conv_2d_op(
            data=h2,
            weights=self.gain_kernels,
            symmetric_weights=self.symmetric_gate_weights)
        with tf.variable_scope(
                '%s/g1_bn' % var_scope,
                reuse=self.scope_reuse) as scope:
            g1_intermediate = tf.contrib.layers.batch_norm(
                inputs=g1_intermediate + self.gain_bias,
                scale=True,
                center=False,
                fused=True,
                renorm=False,
                param_initializers=self.param_initializer,
                updates_collections=None,
                scope=scope,
                reuse=self.reuse,
                is_training=self.train)
        g1 = self.gate_nl(g1_intermediate)

        # Gate the hidden state
        gated_h2 = h2 * g1

        # U and T convolutions
        if self.separate_convs:
            U_sp = tf.nn.depthwise_conv2d(
                input=gated_h2,
                filter=self.U_sp,
                strides=self.strides,
                padding=self.padding)
            U_ch = self.conv_2d_op(
                data=U_sp,
                weights=self.U_ch,
                symmetric_weights=self.symmetric_weights,
                dilations=self.horizontal_dilations)
            T_sp = tf.nn.depthwise_conv2d(
                input=gated_h2,
                filter=self.T_sp,
                strides=self.strides,
                padding=self.padding)
            T_ch = self.conv_2d_op(
                data=T_sp,
                weights=self.T_ch,
                symmetric_weights=self.symmetric_weights,
                dilations=self.horizontal_dilations)
        else:
            U_ch = self.conv_2d_op(
                data=gated_h2,
                weights=self.U_ch,
                symmetric_weights=self.symmetric_weights,
                dilations=self.horizontal_dilations)
            T_ch = self.conv_2d_op(
                data=gated_h2,
                weights=self.T_ch,
                symmetric_weights=self.symmetric_weights,
                dilations=self.horizontal_dilations)
        with tf.variable_scope(
                '%s/U_ch_bn' % var_scope,
                reuse=self.scope_reuse) as scope:
            U_ch = tf.contrib.layers.batch_norm(
                inputs=U_ch,
                scale=True,
                center=False,
                fused=True,
                renorm=False,
                param_initializers=self.param_initializer,
                updates_collections=None,
                scope=scope,
                reuse=self.reuse,
                is_training=self.train)
        with tf.variable_scope(
                '%s/T_ch_bn' % var_scope,
                reuse=self.scope_reuse) as scope:
            T_ch = tf.contrib.layers.batch_norm(
                inputs=T_ch,
                scale=True,
                center=False,
                fused=True,
                renorm=False,
                param_initializers=self.param_initializer,
                updates_collections=None,
                scope=scope,
                reuse=self.reuse,
                is_training=self.train)
        return U_ch, T_ch

    def circuit_output(self, var_scope, h1):
        """Calculate mix and exc horizontal activities."""
        g2_intermediate = self.conv_2d_op(
            data=h1,
            weights=self.mix_kernels,
            symmetric_weights=self.symmetric_gate_weights)
        with tf.variable_scope(
                '%s/g2_bn' % var_scope,
                reuse=self.scope_reuse) as scope:
            g2_intermediate = tf.contrib.layers.batch_norm(
                inputs=g2_intermediate + self.mix_bias,
                scale=True,
                center=False,
                fused=True,
                renorm=False,
                param_initializers=self.param_initializer,
                updates_collections=None,
                scope=scope,
                reuse=self.reuse,
                is_training=self.train)
        # Calculate and apply dropout if requested
        g2 = self.gate_nl(g2_intermediate)

        # U and T convolutions
        if self.separate_convs:
            Q_sp = tf.nn.depthwise_conv2d(
                input=h1,
                filter=self.Q_sp,
                strides=self.strides,
                padding=self.padding)
            Q_ch = self.conv_2d_op(
                data=Q_sp,
                weights=self.Q_ch,
                symmetric_weights=self.symmetric_weights,
                dilations=self.horizontal_dilations)
            P_sp = tf.nn.depthwise_conv2d(
                input=h1,
                filter=self.P_sp,
                strides=self.strides,
                padding=self.padding)
            P_ch = self.conv_2d_op(
                data=P_sp,
                weights=self.P_ch,
                symmetric_weights=self.symmetric_weights,
                dilations=self.horizontal_dilations)
        else:
            Q_ch = self.conv_2d_op(
                data=h1,
                weights=self.Q_ch,
                symmetric_weights=self.symmetric_weights,
                dilations=self.horizontal_dilations)
            P_ch = self.conv_2d_op(
                data=h1,
                weights=self.P_ch,
                symmetric_weights=self.symmetric_weights,
                dilations=self.horizontal_dilations)
        with tf.variable_scope(
                '%s/Q_ch_bn' % var_scope,
                reuse=self.scope_reuse) as scope:
            Q_ch = tf.contrib.layers.batch_norm(
                inputs=Q_ch,
                scale=True,
                center=False,
                fused=True,
                renorm=False,
                param_initializers=self.param_initializer,
                updates_collections=None,
                scope=scope,
                reuse=self.reuse,
                is_training=self.train)
        with tf.variable_scope(
                '%s/P_ch_bn' % var_scope,
                reuse=self.scope_reuse) as scope:
            P_ch = tf.contrib.layers.batch_norm(
                inputs=P_ch,
                scale=True,
                center=False,
                fused=True,
                renorm=False,
                param_initializers=self.param_initializer,
                updates_collections=None,
                scope=scope,
                reuse=self.reuse,
                is_training=self.train)
        return Q_ch, P_ch, g2

    def input_integration(self, x, U_ch, T_ch, h2, h1_i):
        """Integration on the input."""
        if h1_i is None:
            intermediate_inh = (
                self.alpha * h2 + self.mu) * U_ch + (
                self.beta * h2 + self.nu) * T_ch
        else:
            intermediate_inh = (
                self.alpha * h1_i + self.mu) * U_ch + (
                self.beta * h1_i + self.nu) * T_ch
        if self.nonnegative:
            return self.recurrent_nl(x - self.recurrent_nl(intermediate_inh))
        else:
            return self.recurrent_nl(x - intermediate_inh)

    def output_integration(self, h1, Q_ch, P_ch, g2, h2):
        """Integration on the output."""
        if self.multiplicative_excitation:
            # Multiplicative gating I * (P + Q)
            Q_e = self.gamma * Q_ch
            P_e = self.xi * P_ch
            a = self.kappa * (h1 + Q_e + P_e)
            m = self.omega * (h1 * Q_e * P_e)
            h2_hat = self.recurrent_nl(a + m)
        else:
            # Additive gating I + P + Q
            Q_e = self.gamma * Q_ch
            P_e = self.xi * P_ch
            h2_hat = self.recurrent_nl(
                h1 + Q_e + P_e)
        return (g2 * h2) + ((1 - g2) * h2_hat)

    def full(self, i0, x, h1, h1_i, h2, var_scope='hgru_weights'):
        """hGRU body."""
        var_scope = 'hgru_weights'
        if not self.while_loop:
            var_scope = '%s_t%s' % (var_scope, i0)

        U_ch, T_ch = self.circuit_input(
            var_scope=var_scope,
            h2=h2)

        # Calculate input (-) integration: h1 (4)
        h1 = self.input_integration(
            x=x,
            U_ch=U_ch,
            T_ch=T_ch,
            h1_i=h1_i,
            h2=h2)

        # Circuit output receives recurrent input h1
        Q_ch, P_ch, g2 = self.circuit_output(var_scope=var_scope, h1=h1)

        # Calculate output (+) integration: h2 (8, 9)
        h2 = self.output_integration(
            h1=h1,
            Q_ch=Q_ch,
            P_ch=P_ch,
            g2=g2,
            h2=h2)
        if self.adaptation:
            e = tf.gather(self.eta, i0, axis=-1)
            h2 *= e

        # Iterate loop
        i0 += 1
        return i0, x, h1, h1_i, h2

    def condition(self, i0, x, h1, h1_i, h2):
        """While loop halting condition."""
        return i0 < self.timesteps

    def build(self, x):
        """Run the backprop version of the CCircuit."""
        self.prepare_tensors()
        x_shape = x.get_shape().as_list()
        h1_i = None
        if self.hidden_init == 'identity':
            h1 = tf.identity(x)
            h2 = tf.identity(x)
            if self.hidden_states == 'mely':
                h1_i = tf.identity(x)
        elif self.hidden_init == 'random':
            h1 = initialization.xavier_initializer(
                shape=x_shape,
                uniform=self.normal_initializer,
                mask=None)
            h2 = initialization.xavier_initializer(
                shape=x_shape,
                uniform=self.normal_initializer,
                mask=None)
            if self.hidden_states == 'mely':
                h1_i = initialization.xavier_initializer(
                    shape=x_shape,
                    uniform=self.normal_initializer,
                    mask=None)
        elif self.hidden_init == 'zeros':
            h1 = tf.zeros_like(x)
            h2 = tf.zeros_like(x)
            if self.hidden_states == 'mely':
                h1_i = tf.zeros_like(x)
        else:
            raise RuntimeError

        if not self.while_loop:
            for idx in range(self.timesteps):
                _, x, h1, h1_i, h2 = self.full(
                    i0=idx,
                    x=x,
                    h1=h1,
                    h1_i=h1_i,
                    h2=h2)
        else:
            # While loop
            i0 = tf.constant(0)
            elems = [
                i0,
                x,
                h1,
                h1_i,
                h2
            ]
            returned = tf.while_loop(
                self.condition,
                self.full,
                loop_vars=elems,
                back_prop=True,
                swap_memory=False)

            # Prepare output
            i0, x, h1, h1_i, h2 = returned
        return h2
