"""Contextual model with partial filters."""
import numpy as np
import tensorflow as tf
from ops import initialization


# Dependency for symmetric weight ops is in models/layers/ff.py
class hGRU(object):
    def __getitem__(self, name):
        return getattr(self, name)

    def __contains__(self, name):
        return hasattr(self, name)

    def __init__(
            self,
            layer_name,
            x_shape,
            h_ext=15,
            strides=[1, 1, 1, 1],
            padding='SAME',
            aux=None,
            data_format='NHWC',
            train=True):
        """Global initializations and settings."""
        if data_format == 'NHWC':
            self.n, self.timesteps, self.h, self.w, self.k = x_shape
            self.bias_shape = [1, 1, 1, self.k]
        elif data_format == 'NCHW':
            self.n, self.timesteps, self.k, self.h, self.w = x_shape
            self.bias_shape = [1, self.k, 1, 1]
        else:
            raise NotImplementedError(data_format)
        self.strides = strides
        self.padding = padding
        self.train = train
        self.layer_name = layer_name
        self.data_format = data_format
        # Sort through and assign the auxilliary variables
        default_vars = self.defaults()
        if aux is not None and isinstance(aux, dict):
            for k, v in aux.iteritems():
                default_vars[k] = v
        self.update_params(default_vars)

        # Kernel shapes
        self.h_ext = h_ext
        self.h_shape = [self.h_ext, self.h_ext, self.k, self.k]
        self.g_shape = [self.gate_filter, self.gate_filter, self.k, self.k]
        self.m_shape = [self.gate_filter, self.gate_filter, self.k, self.k]

        # Nonlinearities and initializations
        if isinstance(self.recurrent_nl, basestring):
            self.recurrent_nl = self.interpret_nl(self.recurrent_nl)

        # Set integration operations
        self.ii, self.oi = self.input_integration, self.output_integration

    def defaults(self):
        """A dictionary containing defaults for auxilliary variables.

        These are adjusted by a passed aux dict variable."""
        return {
            'lesion_alpha': False,
            'lesion_mu': False,
            'lesion_omega': False,
            'lesion_kappa': False,
            'dtype': tf.float32,
            'hidden_init': 'random',
            'gate_bias_init': 'chronos',
            'train': True,
            'recurrent_nl': tf.nn.tanh,
            'gate_nl': tf.nn.sigmoid,
            'normal_initializer': False,
            'symmetric_weights': False,
            'symmetric_gate_weights': False,
            'gate_filter': 1,  # Gate kernel size
            'gamma': True,  # Scale P
            'alpha': True,  # divisive eCRF
            'mu': True,  # subtractive eCRF
            'adaptation': False,
            'multiplicative_excitation': True,
            'reuse': False,
            'while_loop': False,
            'warmup': 0,
            'additional_recurrence': 0,
            'constrain': True  # Constrain greek letters to be +
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

    def prepare_tensors(self):
        """ Prepare recurrent/forward weight matrices.
        (np.prod([h, w, k]) / 2) - k params in the surround filter
        """
        if self.constrain:
            constraint = lambda x: tf.clip_by_value(x, 0, np.infty)
        else:
            constraint = None
        with tf.variable_scope('%s_hgru_weights' % self.layer_name):
            self.horizontal_kernels_1 = tf.get_variable(
                name='%s_horizontal_1' % self.layer_name,
                dtype=self.dtype,
                initializer=initialization.xavier_initializer(
                    shape=self.h_shape,
                    uniform=self.normal_initializer),
                trainable=self.train)
            self.horizontal_kernels_2 = tf.get_variable(
                name='%s_horizontal_2' % self.layer_name,
                dtype=self.dtype,
                initializer=initialization.xavier_initializer(
                    shape=self.h_shape,
                    uniform=self.normal_initializer),
                trainable=self.train)
            self.gain_kernels = tf.get_variable(
                name='%s_gain' % self.layer_name,
                dtype=self.dtype,
                trainable=self.train,
                initializer=initialization.xavier_initializer(
                    shape=self.g_shape,
                    uniform=self.normal_initializer,
                    mask=None))
            self.mix_kernels = tf.get_variable(
                name='%s_mix' % self.layer_name,
                dtype=self.dtype,
                trainable=self.train,
                initializer=initialization.xavier_initializer(
                    shape=self.m_shape,
                    uniform=self.normal_initializer,
                    mask=None))

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
                    trainable=self.train,
                    constraint=constraint,
                    initializer=initialization.xavier_initializer(
                        shape=self.bias_shape,
                        uniform=self.normal_initializer,
                        mask=None))
            elif self.lesion_alpha:
                self.alpha = tf.constant(0.)
            else:
                self.alpha = tf.constant(1.)

            if self.mu and not self.lesion_mu:
                self.mu = tf.get_variable(
                    name='%s_mu' % self.layer_name,
                    trainable=self.train,
                    constraint=constraint,
                    initializer=initialization.xavier_initializer(
                        shape=self.bias_shape,
                        uniform=self.normal_initializer,
                        mask=None))

            elif self.lesion_mu:
                self.mu = tf.constant(0.)
            else:
                self.mu = tf.constant(1.)

            if self.gamma:
                self.gamma = tf.get_variable(
                    name='%s_gamma' % self.layer_name,
                    trainable=self.train,
                    constraint=constraint,
                    initializer=initialization.xavier_initializer(
                        shape=self.bias_shape,
                        uniform=self.normal_initializer,
                        mask=None))
            else:
                self.gamma = tf.constant(1.)

            if self.multiplicative_excitation:
                if self.lesion_kappa:
                    self.kappa = tf.constant(0.)
                else:
                    self.kappa = tf.get_variable(
                        name='%s_kappa' % self.layer_name,
                        trainable=self.train,
                        constraint=constraint,
                        initializer=initialization.xavier_initializer(
                            shape=self.bias_shape,
                            uniform=self.normal_initializer,
                            mask=None))
                if self.lesion_omega:
                    self.omega = tf.constant(0.)
                else:
                    self.omega = tf.get_variable(
                        name='%s_omega' % self.layer_name,
                        trainable=self.train,
                        constraint=constraint,
                        initializer=initialization.xavier_initializer(
                            shape=self.bias_shape,
                            uniform=self.normal_initializer,
                            mask=None))

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

    def conv_2d_op(
            self,
            data,
            weights,
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
                        data_format=self.data_format,
                        padding=self.padding)
            else:
                activities = tf.nn.conv2d(
                    data,
                    weights,
                    self.strides,
                    data_format=self.data_format,
                    padding=self.padding)
        else:
            raise RuntimeError
        return activities

    def circuit_input(self, h2, idx):
        """Calculate gain and inh horizontal activities."""
        g1_intermediate = self.conv_2d_op(
            data=h2,
            weights=self.gain_kernels,
            symmetric_weights=self.symmetric_gate_weights)
        # with tf.variable_scope('g1_batch_normalization_%s' % idx) as scope:
        g1_intermediate = tf.contrib.layers.batch_norm(
            inputs=g1_intermediate + self.gain_bias,
            scale=True,
            center=False,
            fused=True,
            renorm=False,
            param_initializers={'gamma': tf.constant_initializer(0.1)},
            updates_collections=None,
            # scope=scope,
            reuse=self.reuse,
            is_training=self.train)
        g1 = self.gate_nl(g1_intermediate)

        # Horizontal activities
        c1 = self.conv_2d_op(
            data=h2 * g1,
            weights=self.horizontal_kernels_1,
            symmetric_weights=self.symmetric_weights)
        return c1

    def circuit_output(self, h1, idx):
        """Calculate mix and exc horizontal activities."""
        g2_intermediate = self.conv_2d_op(
            data=h1,
            weights=self.mix_kernels,
            symmetric_weights=self.symmetric_gate_weights)

        # with tf.variable_scope('g2_batch_normalization_%s' % idx) as scope:
        g2_intermediate = tf.contrib.layers.batch_norm(
            inputs=g2_intermediate + self.mix_bias,
            scale=True,
            center=False,
            fused=True,
            renorm=False,
            param_initializers={'gamma': tf.constant_initializer(0.1)},
            updates_collections=None,
            # scope=scope,
            reuse=self.reuse,
            is_training=self.train)
        # Calculate and apply dropout if requested
        g2 = self.gate_nl(g2_intermediate)

        # Horizontal activities
        c2 = self.conv_2d_op(
            data=h1,
            weights=self.horizontal_kernels_2,
            symmetric_weights=self.symmetric_weights)
        return c2, g2

    def input_integration(self, x, c1, h2):
        """Integration on the input."""
        return self.recurrent_nl(
            x - ((self.alpha * h2 + self.mu) * c1))

    def output_integration(self, h1, c2, g2, h2):
        """Integration on the output."""
        if self.multiplicative_excitation:
            # Multiplicative gating I * (P + Q)
            e = self.gamma * c2
            a = self.kappa * (h1 + e)
            m = self.omega * (h1 * e)
            h2_hat = self.recurrent_nl(a + m)
        else:
            # Additive gating I + P + Q
            h2_hat = self.recurrent_nl(
                h1 + self.gamma * c2)
        return (g2 * h2) + ((1 - g2) * h2_hat)

    def full(self, i0, x, h1, h2, dh2, idx=None):
        """hGRU body."""
        # Get x at this timestep
        if self.while_loop:
            it_x = tf.gather(x, i0, axis=1)
            idx = i0
        else:
            it_x = x

        # Circuit input receives recurrent output h2
        c1 = self.circuit_input(h2, idx)

        # with tf.variable_scope('c1_batch_normalization_%s' % idx) as scope:
        c1 = tf.contrib.layers.batch_norm(
            inputs=c1,
            scale=True,
            center=False,
            fused=True,
            renorm=False,
            param_initializers={'gamma': tf.constant_initializer(0.1)},
            updates_collections=None,
            # scope=scope,
            reuse=self.reuse,
            is_training=self.train)

        # Calculate input (-) integration: h1 (4)
        h1 = self.input_integration(
            x=it_x,
            c1=c1,
            h2=h2)

        # Circuit output receives recurrent input h1
        c2, g2 = self.circuit_output(h1, idx)
        # with tf.variable_scope('c2_batch_normalization_%s' % idx) as scope:
        c2 = tf.contrib.layers.batch_norm(
            inputs=c2,
            scale=True,
            center=False,
            fused=True,
            renorm=False,
            param_initializers={'gamma': tf.constant_initializer(0.1)},
            updates_collections=None,
            # scope=scope,
            reuse=self.reuse,
            is_training=self.train)

        # Calculate output (+) integration: h2 (8, 9)
        h2 = self.output_integration(
            h1=h1,
            c2=c2,
            g2=g2,
            h2=h2)
        if self.adaptation:
            e = tf.gather(self.eta, i0, axis=-1)
            h2 *= e

        # Write to dynamic array
        if dh2 is not None:
            dh2 = dh2.write(i0, h2)

        # Iterate loop
        i0 += 1
        return i0, x, h1, h2, dh2

    def condition(self, i0, x, h1, h2, dh2):
        """While loop halting condition."""
        return i0 < self.timesteps

    def build(self, x):
        """Run the backprop version of the CCircuit."""
        self.prepare_tensors()
        full_x_shape = x.get_shape().as_list()
        i0 = tf.constant(0)

        # Remove time dimension
        x_shape = [full_x_shape[0]] + full_x_shape[2:]

        # Set up hiddens
        if self.hidden_init == 'identity':
            h1 = tf.identity(x[:, 0])
            h2 = tf.identity(x[:, 0])
        elif self.hidden_init == 'random':
            h1 = initialization.xavier_initializer(
                shape=x_shape,
                uniform=self.normal_initializer,
                mask=None)
            h2 = initialization.xavier_initializer(
                shape=x_shape,
                uniform=self.normal_initializer,
                mask=None)
        elif self.hidden_init == 'zeros':
            h1 = tf.zeros(x_shape)
            h2 = tf.zeros(x_shape)
        else:
            raise RuntimeError

        # While loop
        if self.while_loop:
            # Get a dynamic h2 to store all h2s
            dh2 = tf.TensorArray(
                dtype=tf.float32,
                size=self.timesteps,
                dynamic_size=False,
                clear_after_read=True,
                name='h2_list')

            elems = [
                i0,
                x,
                h1,
                h2,
                dh2
            ]
            returned = tf.while_loop(
                self.condition,
                self.full,
                loop_vars=elems,
                back_prop=True,
                swap_memory=False)
            i0, x, h1, h2, dh2 = returned
            dh2 = dh2.stack()
            dh2 = tf.transpose(dh2, (1, 0, 2, 3, 4))
            dh2.set_shape(full_x_shape)
        else:
            dh2 = []
            timesteps = self.timesteps + self.warmup
            count, fake_count = 0, 0
            for idx in range(timesteps):
                if idx < self.warmup:
                    # Burn in the 0th frame
                    it_x = x[:, 0, :, :, :]
                    i0, _, h1, h2, _ = self.full(
                        i0=i0,
                        x=it_x,
                        h1=h1,
                        h2=h2,
                        dh2=None,
                        idx=(1 + idx) * 999)
                else:
                    it_x = x[:, count, :, :, :]
                    if self.additional_recurrence:
                        for _ in range(self.additional_recurrence + 1):
                            i0, _, h1, h2, _ = self.full(
                                i0=i0,
                                x=it_x,
                                h1=h1,
                                h2=h2,
                                dh2=None,
                                idx=fake_count)
                            fake_count += 1
                    else:
                        i0, _, h1, h2, _ = self.full(
                            i0=i0,
                            x=it_x,
                            h1=h1,
                            h2=h2,
                            dh2=None,
                            idx=count)
                    dh2 += [h2]
                    count += 1
                    fake_count += 1
            dh2 = tf.stack(dh2, axis=1)

        # Prepare output
        return dh2
