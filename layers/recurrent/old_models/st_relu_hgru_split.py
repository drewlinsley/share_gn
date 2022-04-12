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
            timesteps=1,
            h_ext=15,
            strides=[1, 1, 1, 1],
            padding='SAME',
            aux=None,
            train=True):
        """Global initializations and settings."""
        self.n, self.h, self.w, self.k = x_shape
        self.timesteps = timesteps
        self.strides = strides
        self.padding = padding
        self.train = train
        self.layer_name = layer_name

        # Sort through and assign the auxilliary variables
        default_vars = self.defaults()
        if aux is not None and isinstance(aux, dict):
            for k, v in aux.iteritems():
                default_vars[k] = v
        self.update_params(default_vars)

        # Kernel shapes
        self.h_ext = 2 * np.floor(h_ext / 2.0).astype(np.int) + 1
        self.h_shape = [self.h_ext, self.h_ext, self.k, self.k]
        self.g_shape = [self.gate_filter, self.gate_filter, self.k, self.k]
        self.m_shape = [self.gate_filter, self.gate_filter, self.k, self.k]
        self.bias_shape = [1, 1, 1, self.k]

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
            'hidden_init': 'identity',
            'gate_bias_init': 'chronos',
            'train': True,
            'recurrent_nl': tf.nn.tanh,
            'gate_nl': tf.nn.sigmoid,
            'normal_initializer': True,
            'symmetric_weights': True,
            'symmetric_gate_weights': False,
            'gate_filter': 1,  # Gate kernel size
            'gamma': True,  # Scale P
            'alpha': True,  # divisive eCRF
            'mu': True,  # subtractive eCRF
            'adapation': True,
            'multiplicative_excitation': True,
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

    def symmetric_weights(self, w, name):
        """Apply symmetric weight sharing."""
        conv_w_t = tf.transpose(w, (2, 3, 0, 1))
        conv_w_symm = 0.5 * (conv_w_t + tf.transpose(conv_w_t, (1, 0, 2, 3)))
        conv_w = tf.transpose(conv_w_symm, (2, 3, 0, 1), name=name)
        return conv_w

    def prepare_tensors(self):
        """ Prepare recurrent/forward weight matrices.
        (np.prod([h, w, k]) / 2) - k params in the surround filter
        """
        with tf.variable_scope('%s_hgru_weights' % self.layer_name):
            self.horizontal_kernels = tf.get_variable(
                name='%s_horizontal' % self.layer_name,
                dtype=self.dtype,
                initializer=initialization.xavier_initializer(
                    shape=self.h_shape,
                    uniform=self.normal_initializer),
                trainable=self.train)
            self.h_bias = tf.get_variable(
                name='%s_h_bias' % self.layer_name,
                initializer=initialization.xavier_initializer(
                    shape=self.bias_shape,
                    uniform=self.normal_initializer,
                    mask=None))
            self.gain_kernels_a = tf.get_variable(
                name='%s_gain_a' % self.layer_name,
                dtype=self.dtype,
                trainable=self.train,
                initializer=initialization.xavier_initializer(
                    shape=self.g_shape,
                    uniform=self.normal_initializer,
                    mask=None))
            self.gain_kernels_b = tf.get_variable(
                name='%s_gain_b' % self.layer_name,
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
                bias_init = tf.ones(self.bias_shape)
            self.gain_bias_a = tf.get_variable(
                name='%s_gain_bias_a' % self.layer_name,
                dtype=self.dtype,
                trainable=self.train,
                initializer=bias_init)
            self.gain_bias_b = tf.get_variable(
                name='%s_gain_bias_b' % self.layer_name,
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
                        initializer=initialization.xavier_initializer(
                            shape=self.bias_shape,
                            uniform=self.normal_initializer,
                            mask=None))
            else:
                self.kappa = tf.constant(1.)
                self.omega = tf.constant(1.)

            if self.adapation:
                self.eta = tf.get_variable(
                    trainable=self.train,
                    name='%s_eta' % self.layer_name,
                    initializer=tf.ones(self.timesteps, dtype=tf.float32))
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
                        padding=self.padding)
            else:
                activities = tf.nn.conv2d(
                    data,
                    weights,
                    self.strides,
                    padding=self.padding)
        else:
            raise RuntimeError
        return activities

    def circuit_input(self, h2):
        """Calculate gain and inh horizontal activities."""
        g1_intermediate_a = self.conv_2d_op(
            data=h2,
            weights=self.gain_kernels_a,
            symmetric_weights=self.symmetric_gate_weights)
        g1_a = self.gate_nl(g1_intermediate_a + self.gain_bias)

        # Horizontal activities
        c1_a = self.conv_2d_op(
            data=h2 * g1_a,
            weights=self.horizontal_kernels,
            symmetric_weights=self.symmetric_weights)
        c1_a += self.h_bias

        g1_intermediate_b = self.conv_2d_op(
            data=h2,
            weights=self.gain_kernels_b,
            symmetric_weights=self.symmetric_gate_weights)
        g1_b = self.gate_nl(g1_intermediate_b + self.gain_bias)

        # Horizontal activities
        c1_b = self.conv_2d_op(
            data=h2 * g1_b,
            weights=self.horizontal_kernels,
            symmetric_weights=self.symmetric_weights)
        c1_b += self.h_bias
        return c1_a, c1_b

    def circuit_output(self, h1):
        """Calculate mix and exc horizontal activities."""
        g2_intermediate = self.conv_2d_op(
            data=h1,
            weights=self.mix_kernels,
            symmetric_weights=self.symmetric_gate_weights)

        # Calculate and apply dropout if requested
        g2 = self.gate_nl(g2_intermediate + self.mix_bias)

        # Horizontal activities
        c2 = self.conv_2d_op(
            data=h1,
            weights=self.horizontal_kernels,
            symmetric_weights=self.symmetric_weights)
        c2 += self.h_bias
        return c2, g2

    def input_integration(self, x, c1, c2, h2):
        """Integration on the input."""
        linear = (self.alpha * h2 + self.mu) * c1
        multiplicative = (self.alpha * h2 + self.mu) * c2
        return tf.nn.relu(x - (linear + multiplicative))

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

    def full(self, i0, x, h1, h2):
        """hGRU body."""
        # Circuit input receives recurrent output h2
        c1, c2 = self.circuit_input(h2)

        # Calculate input (-) integration: h1 (4)
        h1 = self.input_integration(
            x=x,
            c1=c1,
            c2=c2,
            h2=h2)

        # Circuit output receives recurrent input h1
        c2, g2 = self.circuit_output(h1)

        # Calculate output (+) integration: h2 (8, 9)
        h2 = self.output_integration(
            h1=h1,
            c2=c2,
            g2=g2,
            h2=h2)

        if self.adapation:
            e = tf.gather(self.eta, i0, axis=-1)
            h2 *= e

        # Iterate loop
        i0 += 1
        return i0, x, h1, h2

    def condition(self, i0, x, h1, h2):
        """While loop halting condition."""
        return i0 < self.timesteps

    def build(self, x):
        """Run the backprop version of the CCircuit."""
        self.prepare_tensors()
        i0 = tf.constant(0)
        if self.hidden_init == 'identity':
            h1 = tf.identity(x)
            h2 = tf.identity(x)
        elif self.hidden_init == 'random':
            h1 = initialization.xavier_initializer(
                shape=[self.n, self.h, self.w, self.k],
                uniform=self.normal_initializer,
                mask=None)
            h2 = initialization.xavier_initializer(
                shape=[self.n, self.h, self.w, self.k],
                uniform=self.normal_initializer,
                mask=None)
        elif self.hidden_init == 'zeros':
            h1 = tf.zeros_like(x)
            h2 = tf.zeros_like(x)
        else:
            raise RuntimeError

        # While loop
        elems = [
            i0,
            x,
            h1,
            h2
        ]
        returned = tf.while_loop(
            self.condition,
            self.full,
            loop_vars=elems,
            back_prop=True,
            swap_memory=True)

        # Prepare output
        i0, x, h1, h2 = returned
        return h2
