"""Contextual model with partial filters."""
import warnings
import numpy as np
import tensorflow as tf
from ops import initialization
from layers.feedforward.pooling import max_pool


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
            hgru_ids,
            hgru_idx,
            timesteps=1,
            h_ext=[[15, 15]],
            strides=[1, 1],
            pool_strides=[4, 4],
            pooling_kernel=[4, 4],
            padding='SAME',
            aux=None,
            train=True):
        """Global initializations and settings."""
        self.n, self.h, self.w, self.k = x_shape
        self.hgru_ids = hgru_ids
        self.hgru_idx = hgru_idx
        self.timesteps = timesteps
        self.strides = strides
        self.pool_strides = pool_strides
        self.pool_kernel = pooling_kernel
        self.padding = padding
        self.train = train
        self.layer_name = layer_name
        self.up_kernel = [x + y for x, y in zip(
            self.pool_strides, self.pool_kernel)]

        # Sort through and assign the auxilliary variables
        default_vars = self.defaults()
        if aux is not None and isinstance(aux, dict):
            for k, v in aux.iteritems():
                default_vars[k] = v
        self.update_params(default_vars)

        # Kernel shapes
        self.h_ext = h_ext

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

    def defaults(self):
        """A dictionary containing defaults for auxilliary variables.

        These are adjusted by a passed aux dict variable."""
        return {
            'lesion_alpha': False,
            'lesion_mu': False,
            'lesion_omega': False,
            'lesion_kappa': False,
            'dtype': tf.float32,
            'hidden_init': 'zeros',
            'gate_bias_init': 'chronos',
            'train': True,
            'recurrent_nl': tf.nn.tanh,
            'gate_nl': tf.nn.sigmoid,
            'ff_nl': tf.nn.relu,
            'normal_initializer': True,
            'symmetric_weights': True,
            'symmetric_gate_weights': False,
            'gate_filter': 1,  # Gate kernel size
            'gamma': True,  # Scale P
            'alpha': True,  # divisive eCRF
            'mu': True,  # subtractive eCRF
            'adaptation': False,
            'reuse': False,
            'reuse_conv_bn': False,
            'while_loop': False,
            'multiplicative_excitation': True,
            'readout': 'fb',  # l2 or fb
            'intermediate_ff': [32, 32, 32],
            'intermediate_ks': [[7, 7], [7, 7], [7, 7]],
            'include_pooling': True,
            'hgru_2_ff_kernel': [5, 5],
            'skip': False,
            'batch_norm': True,
            'dilations': [[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]]
        }

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

    def update_params(self, kwargs):
        """Update the class attributes with kwargs."""
        if kwargs is not None:
            for k, v in kwargs.iteritems():
                setattr(self, k, v)

    def prepare_tensors(self):
        """ Prepare recurrent/forward weight matrices.
        (np.prod([h, w, k]) / 2) - k params in the surround filter
        """
        # Create FF vars
        if self.include_pooling:
            # Upsample FF layers then hgru layer
            up_filters = [self.hgru_ids['h1']] + self.intermediate_ff
            for idx in reversed(range(1, len(up_filters))):
                label = idx - 1
                setattr(
                    self,
                    'resize_kernel_%s' % label,
                    tf.get_variable(
                        name='%s_resize_kernel_%s' % (self.layer_name, label),
                        dtype=self.dtype,
                        initializer=initialization.xavier_initializer(
                            shape=self.up_kernel + [up_filters[idx - 1], up_filters[idx]],
                            uniform=self.normal_initializer),
                        trainable=True))
                setattr(
                    self,
                    'resize_bias_%s' % label,
                    tf.get_variable(
                        name='%s_resize_bias_%s' % (self.layer_name, label),
                        dtype=self.dtype,
                        initializer=tf.ones([up_filters[idx - 1]]),
                        trainable=True))

        # Create conv filters that supply top-drive
        prev_filters = self.hgru_ids['h1']
        for idx, (ff_filters, ff_kernel) in enumerate(
                zip(
                    self.intermediate_ff,
                    self.intermediate_ks)):
            setattr(
                self,
                'intermediate_kernel_%s' % idx,
                tf.get_variable(
                    name='%s_ffdrive_kernel_%s' % (self.layer_name, idx),
                    dtype=self.dtype,
                    initializer=initialization.xavier_initializer(
                        shape=ff_kernel + [prev_filters, ff_filters],
                        uniform=self.normal_initializer),
                    trainable=True))
            setattr(
                self,
                'intermediate_bias_%s' % idx,
                tf.get_variable(
                    name='%s_ffdrive_bias_%s' % (self.layer_name, idx),
                    dtype=self.dtype,
                    initializer=tf.ones([ff_filters]),
                    trainable=True))
            prev_filters = ff_filters

        # Create recurrent vars
        for idx, (layer, rk) in enumerate(self.hgru_ids.iteritems()):
            self.g_shape = [
                self.gate_filter,
                self.gate_filter,
                rk,
                rk]
            self.m_shape = [
                self.gate_filter,
                self.gate_filter,
                rk,
                rk]
            self.bias_shape = [1, 1, 1, rk]

            with tf.variable_scope(
                    '%s_hgru_weights_%s' % (self.layer_name, layer)):
                setattr(
                    self,
                    'horizontal_kernels_exc_%s' % layer,
                    tf.get_variable(
                        name='%s_horizontal_exc' % self.layer_name,
                        dtype=self.dtype,
                        initializer=initialization.xavier_initializer(
                            shape=self.h_ext[idx] + [rk, rk],
                            uniform=self.normal_initializer),
                        trainable=True))
                setattr(
                    self,
                    'horizontal_kernels_inh_%s' % layer,
                    tf.get_variable(
                        name='%s_horizontal_inh' % self.layer_name,
                        dtype=self.dtype,
                        initializer=initialization.xavier_initializer(
                            shape=self.h_ext[idx] + [rk, rk],
                            uniform=self.normal_initializer),
                        trainable=True))
                setattr(
                    self,
                    'gain_kernels_%s' % layer,
                    tf.get_variable(
                        name='%s_gain' % self.layer_name,
                        dtype=self.dtype,
                        trainable=True,
                        initializer=initialization.xavier_initializer(
                            shape=self.g_shape,
                            uniform=self.normal_initializer,
                            mask=None)))
                setattr(
                    self,
                    'mix_kernels_%s' % layer,
                    tf.get_variable(
                        name='%s_mix' % self.layer_name,
                        dtype=self.dtype,
                        trainable=True,
                        initializer=initialization.xavier_initializer(
                            shape=self.m_shape,
                            uniform=self.normal_initializer,
                            mask=None)))

                # Gain bias
                if self.gate_bias_init == 'chronos':
                    bias_init = -tf.log(
                        tf.random_uniform(
                            self.bias_shape,
                            minval=1,
                            maxval=self.timesteps - 1))
                else:
                    bias_init = tf.ones(self.bias_shape)
                setattr(
                    self,
                    'gain_bias_%s' % layer,
                    tf.get_variable(
                        name='%s_gain_bias' % self.layer_name,
                        dtype=self.dtype,
                        trainable=True,
                        initializer=bias_init))
                if self.gate_bias_init == 'chronos':
                    bias_init = -bias_init
                else:
                    bias_init = tf.ones(self.bias_shape)
                setattr(
                    self,
                    'mix_bias_%s' % layer,
                    tf.get_variable(
                        name='%s_mix_bias' % self.layer_name,
                        dtype=self.dtype,
                        trainable=True,
                        initializer=bias_init))

                # Divisive params
                if self.alpha and not self.lesion_alpha:
                    setattr(
                        self,
                        'alpha_%s' % layer,
                        tf.get_variable(
                            name='%s_alpha' % self.layer_name,
                            initializer=initialization.xavier_initializer(
                                shape=self.bias_shape,
                                uniform=self.normal_initializer,
                                mask=None)))
                elif self.lesion_alpha:
                    setattr(
                        self,
                        'alpha_%s' % layer,
                        tf.constant(0.))
                else:
                    setattr(
                        self,
                        'alpha_%s' % layer,
                        tf.constant(1.))

                if self.mu and not self.lesion_mu:
                    setattr(
                        self,
                        'mu_%s' % layer,
                        tf.get_variable(
                            name='%s_mu' % self.layer_name,
                            initializer=initialization.xavier_initializer(
                                shape=self.bias_shape,
                                uniform=self.normal_initializer,
                                mask=None)))

                elif self.lesion_mu:
                    setattr(
                        self,
                        'mu_%s' % layer,
                        tf.constant(0.))
                else:
                    setattr(
                        self,
                        'mu_%s' % layer,
                        tf.constant(1.))

                if self.gamma:
                    setattr(
                        self,
                        'gamma_%s' % layer,
                        tf.get_variable(
                            name='%s_gamma' % self.layer_name,
                            initializer=initialization.xavier_initializer(
                                shape=self.bias_shape,
                                uniform=self.normal_initializer,
                                mask=None)))
                else:
                    setattr(
                        self,
                        'gamma_%s' % layer,
                        tf.constant(1.))

                if self.multiplicative_excitation:
                    if self.lesion_kappa:
                        setattr(
                            self,
                            'kappa_%s' % layer,
                            tf.constant(0.))
                    else:
                        setattr(
                            self,
                            'kappa_%s' % layer,
                            tf.get_variable(
                                name='%s_kappa' % self.layer_name,
                                initializer=initialization.xavier_initializer(
                                    shape=self.bias_shape,
                                    uniform=self.normal_initializer,
                                    mask=None)))
                    if self.lesion_omega:
                        setattr(
                            self,
                            'omega_%s' % layer,
                            tf.constant(0.))
                    else:
                        setattr(
                            self,
                            'omega_%s' % layer,
                            tf.get_variable(
                                name='%s_omega' % self.layer_name,
                                initializer=initialization.xavier_initializer(
                                    shape=self.bias_shape,
                                    uniform=self.normal_initializer,
                                    mask=None)))
                else:
                    setattr(
                        self,
                        'kappa_%s' % layer,
                        tf.constant(1.))
                    setattr(
                        self,
                        'omega_%s' % layer,
                        tf.constant(1.))
                if self.adaptation:
                    setattr(
                        self,
                        'eta_%s' % layer,
                        tf.get_variable(
                            name='%s_eta' % self.layer_name,
                            initializer=tf.random_uniform(
                                [self.timesteps], dtype=tf.float32)))
                if self.lesion_omega:
                    setattr(
                        self,
                        'omega_%s' % layer,
                        tf.constant(0.))
                if self.lesion_kappa:
                    setattr(
                        self,
                        'kappa_%s' % layer,
                        tf.constant(0.))
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
                                    shape=[rk],
                                    collections=self.param_collections[v],
                                    initializer=self.param_initializer[v])
                    self.param_initializer = None

    def resize_x_to_y(
            self,
            x,
            y,
            name,
            mode='transpose',
            use_bias=True):
        """Resize activity x to the size of y using interpolation."""
        y_size = y.get_shape().as_list()[1:]
        if mode == 'resize':
            raise NotImplementedError
            return tf.image.resize_images(
                x,
                y_size[:-1],
                self.resize_kernel,
                align_corners=True)
        elif mode == 'transpose':
            if self.n is None:
                n = 1
                warnings.warn('Found None for batch size. Forcing to 1.')
            else:
                n = self.n
            resize_kernel = getattr(self, 'resize_kernel_%s' % name)
            resize_bias = getattr(self, 'resize_bias_%s' % name)
            resized = tf.nn.conv2d_transpose(
                value=x,
                filter=resize_kernel,
                output_shape=[n] + y_size,
                strides=[1] + self.pool_strides + [1],
                padding=self.padding,
                name='resize_x_to_y_%s' % name)
            resized = tf.nn.bias_add(
                resized,
                resize_bias)
            resized = self.ff_nl(resized)
            return resized
        else:
            raise NotImplementedError(mode)

    def conv_2d_op(
            self,
            data,
            weights,
            symmetric_weights=False,
            dilations=None):
        """2D convolutions for hgru."""
        if dilations is None:
            dilations = [1, 1, 1, 1]
        w_shape = [int(w) for w in weights.get_shape()]
        if len(w_shape) > 1 and int(w_shape[-2]) > 1:
            # Full convolutions
            if symmetric_weights:
                g = tf.get_default_graph()
                with g.gradient_override_map({'Conv2D': 'SymmetricConv2D'}):
                    activities = tf.nn.conv2d(
                        data,
                        weights,
                        self.strides,
                        padding=self.padding,
                        dilations=dilations)
            else:
                activities = tf.nn.conv2d(
                    data,
                    weights,
                    self.strides,
                    padding=self.padding,
                    dilations=dilations)
        else:
            raise RuntimeError
        return activities

    def circuit_input(self, h2, layer, var_scope, layer_idx):
        """Calculate gain and inh horizontal activities."""
        gain_kernels = getattr(self, 'gain_kernels_%s' % layer)
        gain_bias = getattr(self, 'gain_bias_%s' % layer)
        horizontal_kernels_inh = getattr(
            self, 'horizontal_kernels_inh_%s' % layer)
        # h_bias = getattr(self, 'h_bias_%s' % layer)
        g1_intermediate = self.conv_2d_op(
            data=h2,
            weights=gain_kernels,
            symmetric_weights=self.symmetric_gate_weights,
            dilations=self.dilations[layer_idx])
        with tf.variable_scope(
                '%s/g1_bn' % var_scope,
                reuse=self.scope_reuse) as scope:
            g1_intermediate = tf.contrib.layers.batch_norm(
                inputs=g1_intermediate + gain_bias,
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
        h2 *= g1

        # Horizontal activities
        c1 = self.conv_2d_op(
            data=h2,
            weights=horizontal_kernels_inh,
            symmetric_weights=self.symmetric_weights,
            dilations=self.dilations[layer_idx])
        return c1, g1

    def circuit_output(self, h1, layer, var_scope, layer_idx):
        """Calculate mix and exc horizontal activities."""
        mix_kernels = getattr(self, 'mix_kernels_%s' % layer)
        mix_bias = getattr(self, 'mix_bias_%s' % layer)
        horizontal_kernels_exc = getattr(
            self, 'horizontal_kernels_exc_%s' % layer)
        # h_bias = getattr(self, 'h_bias_%s' % layer)
        g2_intermediate = self.conv_2d_op(
            data=h1,
            weights=mix_kernels,
            symmetric_weights=self.symmetric_gate_weights,
            dilations=self.dilations[layer_idx])

        with tf.variable_scope(
                '%s/g2_bn' % var_scope,
                reuse=self.scope_reuse) as scope:
            g2_intermediate = tf.contrib.layers.batch_norm(
                inputs=g2_intermediate + mix_bias,
                scale=True,
                center=False,
                fused=True,
                renorm=False,
                param_initializers=self.param_initializer,
                updates_collections=None,
                scope=scope,
                reuse=self.reuse,
                is_training=self.train)
        g2 = self.gate_nl(g2_intermediate)

        # Horizontal activities
        c2 = self.conv_2d_op(
            data=h1,
            weights=horizontal_kernels_exc,
            symmetric_weights=self.symmetric_weights,
            dilations=self.dilations[layer_idx])
        return c2, g2

    def input_integration(self, x, c1, h2, layer):
        """Integration on the input."""
        alpha = getattr(self, 'alpha_%s' % layer)
        mu = getattr(self, 'mu_%s' % layer)
        # Nonnegative constraint on horiz interactions with x
        return self.recurrent_nl(x - self.recurrent_nl((alpha * h2 + mu) * c1))

    def output_integration(self, h1, c2, g2, h2, layer):
        """Integration on the output."""
        if self.multiplicative_excitation:
            # Multiplicative gating I * (P + Q)
            gamma = getattr(self, 'gamma_%s' % layer)
            kappa = getattr(self, 'kappa_%s' % layer)
            omega = getattr(self, 'omega_%s' % layer)
            e = gamma * c2
            a = kappa * (h1 + e)
            m = omega * (h1 * e)
            h2_hat = self.recurrent_nl(a + m)
        else:
            # Additive gating I + P + Q
            h2_hat = self.recurrent_nl(
                h1 + gamma * c2)
        return (g2 * h2) + ((1 - g2) * h2_hat)

    def hgru_ops(self, i0, x, h2, layer):
        """hGRU body."""
        layer_idx = self.hgru_idx[layer]
        assert layer_idx is not None, 'Cannot find the hgru layer id.'
        var_scope = '%s_hgru_weights' % layer
        if not self.while_loop:
            var_scope = '%s_t%s' % (var_scope, i0)

        # Circuit input receives recurrent output h2
        c1, g1 = self.circuit_input(
            h2=h2,
            layer=layer,
            var_scope=var_scope,
            layer_idx=layer_idx)
        with tf.variable_scope(
                '%s/c1_bn' % var_scope,
                reuse=self.scope_reuse) as scope:
            c1 = tf.contrib.layers.batch_norm(
                inputs=c1,
                scale=True,
                center=False,
                fused=True,
                renorm=False,
                param_initializers=self.param_initializer,
                updates_collections=None,
                scope=scope,
                reuse=self.reuse,
                is_training=self.train)

        # Calculate input (-) integration: h1 (4)
        h1 = self.input_integration(
            x=x,
            c1=c1,
            h2=h2,
            layer=layer)

        # Circuit output receives recurrent input h1
        c2, g2 = self.circuit_output(
            h1=h1,
            layer=layer,
            var_scope=var_scope,
            layer_idx=layer_idx)

        with tf.variable_scope(
                '%s/c2_bn' % var_scope,
                reuse=self.scope_reuse) as scope:
            c2 = tf.contrib.layers.batch_norm(
                inputs=c2,
                scale=True,
                center=False,
                fused=True,
                renorm=False,
                param_initializers=self.param_initializer,
                updates_collections=None,
                scope=scope,
                reuse=self.reuse,
                is_training=self.train)

        # Calculate output (+) integration: h2 (8, 9)
        h2 = self.output_integration(
            h1=h1,
            c2=c2,
            g2=g2,
            h2=h2,
            layer=layer)

        if self.adaptation:
            eta = getattr(self, 'eta_%s' % layer)
            e = tf.gather(eta, i0, axis=-1)
            h2 *= e
        return h1, h2

    def full(self, i0, x, l1_h2, l2_h2, fb_act_1):
        """hGRU body.
        Take the recurrent h2 from a low level and imbue it with
        information froma high layer. This means to treat the lower
        layer h2 as the X and the higher layer h2 as the recurrent state.
        This will serve as I/E from the high layer along with feedback
        kernels.
        """

        # # LAYER 1 hGRU
        # FF drive comes from outside recurrent loop
        l1_h1, l1_h2 = self.hgru_ops(
            i0=i0,
            x=x,
            h2=fb_act_1,
            layer='h1')
        l1_h2_scope = 'l1_h2_bn'
        if not self.while_loop:
            l1_h2_scope = '%s_t%s' % (l1_h2_scope, i0)
        if self.batch_norm:
            with tf.variable_scope(
                    l1_h2_scope,
                    reuse=self.scope_reuse) as scope:
                l1_h2 = tf.contrib.layers.batch_norm(
                    inputs=l1_h2,
                    scale=True,
                    center=False,
                    fused=True,
                    renorm=False,
                    param_initializers=self.param_initializer,
                    updates_collections=None,
                    scope=scope,
                    reuse=self.reuse,
                    is_training=self.train)

        # Pool the preceding layer's drive
        if self.include_pooling:
            l1_h2_pool = max_pool(
                bottom=l1_h2,
                k=[1] + self.pool_kernel + [1],
                s=[1] + self.pool_strides + [1],
                name='pool_h1')
        else:
            l1_h2_pool = l1_h2

        # Conv hierarchy for high-level representation
        conv_list = [l1_h2]
        activity = l1_h2_pool
        for idx, (filters, s) in enumerate(
                zip(self.intermediate_ff, self.intermediate_ks)):
            activity = tf.nn.conv2d(
                input=activity,
                filter=getattr(self, 'intermediate_kernel_%s' % idx),
                strides=self.strides,
                padding=self.padding)
            activity = tf.nn.bias_add(
                activity,
                getattr(self, 'intermediate_bias_%s' % idx))
            activity = self.ff_nl(activity)
            ff_scope = 'bn_ff_%s' % idx
            if not self.while_loop:
                ff_scope = '%s_t%s' % (ff_scope, i0)
            if self.batch_norm:
                with tf.variable_scope(
                        ff_scope,
                        reuse=self.scope_reuse) as scope:
                    activity = tf.contrib.layers.batch_norm(
                        inputs=activity,
                        scale=True,
                        center=True,
                        fused=True,
                        renorm=False,
                        param_initializers=self.param_initializer,
                        updates_collections=None,
                        scope=scope,
                        reuse=self.reuse_conv_bn,
                        is_training=self.train)
            # Gather in a list for upsample
            if idx < (len(self.intermediate_ff) - 1):
                conv_list += [activity]
                # Add pools for encoding path
                activity = max_pool(
                    bottom=activity,
                    k=[1] + self.pool_kernel + [1],
                    s=[1] + self.pool_strides + [1],
                    name='ff_pool_%s' % (idx))

        # # LAYER 2 hGRU
        # hGRU
        l2_h1, l2_h2 = self.hgru_ops(
            i0=i0,
            x=activity,
            h2=l2_h2,
            layer='h2')
        l2_h2_scope = 'l2_h2_bn'
        if not self.while_loop:
            l2_h2_scope = '%s_t%s' % (l2_h2_scope, i0)
        if self.batch_norm:
            with tf.variable_scope(
                    l2_h2_scope,
                    reuse=self.scope_reuse) as scope:
                l2_h2 = tf.contrib.layers.batch_norm(
                    inputs=l2_h2,
                    scale=True,
                    center=False,
                    fused=True,
                    renorm=False,
                    param_initializers=self.param_initializer,
                    updates_collections=None,
                    scope=scope,
                    reuse=self.reuse,
                    is_training=self.train)
        activity = l2_h2

        # Add Upsamples
        for idx, target in reversed(list(enumerate(conv_list))):
            activity = self.resize_x_to_y(x=activity, y=target, name=idx)
            if self.batch_norm:
                up_scope = 'up_bn_%s' % idx
                if not self.while_loop:
                    up_scope = '%s_t%s' % (up_scope, i0)
                with tf.variable_scope(
                        up_scope,
                        reuse=self.scope_reuse) as scope:
                    activity = tf.contrib.layers.batch_norm(
                        inputs=activity,
                        scale=True,
                        center=True,
                        fused=True,
                        renorm=False,
                        param_initializers=self.param_initializer,
                        updates_collections=None,
                        scope=scope,
                        reuse=self.reuse,
                        is_training=self.train)
            if self.skip and idx > 0:
                activity += target  # Do not skip through the hgru

        # # LAYER 1 tdGRU
        # Feedback from hgru-2 to hgru-1
        fb_inh_1, fb_act_1 = self.hgru_ops(
            i0=i0,
            x=l1_h2,
            h2=activity,
            layer='fb1')

        # TD 1 Batchnorm
        td_h1_scope = 'td_h1_bn'
        if not self.while_loop:
            td_h1_scope = '%s_t%s' % (td_h1_scope, i0)
        if self.batch_norm:
            with tf.variable_scope(
                    td_h1_scope,
                    reuse=self.scope_reuse) as scope:
                fb_act_1 = tf.contrib.layers.batch_norm(
                    inputs=fb_act_1,
                    scale=True,
                    center=False,
                    fused=True,
                    renorm=False,
                    param_initializers=self.param_initializer,
                    updates_collections=None,
                    scope=scope,
                    reuse=self.reuse,
                    is_training=self.train)

        # Peephole z-scored activities
        fb_act_1 += l1_h2

        # Iterate loop
        i0 += 1
        return i0, x, l1_h2, l2_h2, fb_act_1

    def condition(self, i0, x, l1_h2, l2_h2, fb_act_1):
        """While loop halting condition."""
        return i0 < self.timesteps

    def build(self, x):
        """Run the backprop version of the Circuit."""
        self.prepare_tensors()

        # Calculate l2 hidden state size
        x_shape = tf.shape(x)
        if self.include_pooling and len(self.intermediate_ff):
            # pooling_factor = (len(
            #     self.intermediate_ff)) * np.sum(self.pool_strides)
            pooling_factor = self.pool_strides[0] ** len(self.intermediate_ff)
            l2_shape = tf.stack(
                [
                    x_shape[0],
                    x_shape[1] / pooling_factor,
                    x_shape[2] / pooling_factor,
                    self.hgru_ids['h2']])
        else:
            l2_shape = tf.identity(x_shape)
            self.pooling_factor = 1
        np_xsh = np.array(x.get_shape().as_list())
        np_xsh[1:3] /= pooling_factor
        np_xsh[-1] = self.hgru_ids['h2']
        print '*' * 20
        print 'fgru embedding shape is: '
        print np_xsh
        print '*' * 20

        # Initialize hidden layer activities
        if self.hidden_init == 'identity':
            l1_h2 = tf.identity(x)
            l2_h2 = tf.zeros(l2_shape)
            fb_act_1 = tf.identity(x)
        elif self.hidden_init == 'random':
            l1_h2 = tf.random_normal(x_shape)
            l2_h2 = tf.random_normal(l2_shape)
            fb_act_1 = tf.random_normal(x_shape)
        elif self.hidden_init == 'zeros':
            l1_h2 = tf.zeros(x_shape)
            l2_h2 = tf.zeros(l2_shape)
            fb_act_1 = tf.zeros(x_shape)
        else:
            raise RuntimeError

        # While loop
        if self.while_loop:
            i0 = tf.constant(0)
            elems = [
                i0,
                x,
                l1_h2,
                l2_h2,
                fb_act_1
            ]
            returned = tf.while_loop(
                self.condition,
                self.full,
                loop_vars=elems,
                back_prop=True,
                swap_memory=False)

            # Prepare output
            i0, x, l1_h2, l2_h2, fb_act_1 = returned

        else:
            i0 = 0
            for idx in range(self.timesteps):
                i0, x, l1_h2, l2_h2, fb_act_1 = self.full(
                    i0=i0,
                    x=x,
                    l1_h2=l1_h2,
                    l2_h2=l2_h2,
                    fb_act_1=fb_act_1)
        if self.readout == 'fb':
            return fb_act_1
        else:
            raise NotImplementedError('Select an hGRU layer to readout from.')

