"""Contextual model with partial filters."""
import warnings
import numpy as np
import tensorflow as tf
from ops import initialization, gradients
from layers.feedforward.pooling import max_pool
from layers.feedforward.pooling import global_pool


# Dependency for symmetric weight ops is in models/layers/ff.py
class hGRU(object):
    def __getitem__(self, name):
        return getattr(self, name)

    def __contains__(self, name):
        return hasattr(self, name)

    def __init__(
            self,
            var_scope,
            timesteps,
            in_k,
            use_global_pool,
            share_ff_td_kernels,
            fgru_module_class,
            hgru_fsiz,
            hgru_fanout_factor,
            hgru_h2_k,
            ff_conv_fsiz,
            ff_conv_k,
            ff_conv_strides,
            ff_pool_fsiz,
            ff_pool_strides,
            fb_conv_fsiz,
            fb_conv_k,
            train=True,
            dtype=tf.bfloat16,
            **fgru_layer_optional_args):

        # Sanity check
        if (len(ff_conv_fsiz) != len(ff_conv_k)) | (len(ff_conv_fsiz) != len(ff_conv_strides)):
            raise ValueError('ff_conv params should have same length (num layers)')
        if (len(ff_pool_fsiz) != len(ff_pool_strides)) | (len(ff_conv_fsiz) != len(ff_pool_fsiz)):
            raise ValueError('ff_pool params should have same length (num layers) which also matches ff_conv params')
        if (len(fb_conv_fsiz) != len(fb_conv_k)) | (len(fb_conv_fsiz) != len(ff_conv_fsiz)+1):
            raise ValueError('fb_conv params should have same length (num layers) which is also one longer than ff_conv params')

        # global params
        self.var_scope = var_scope
        self.timesteps = timesteps
        self.dtype = dtype
        self.train = train
        self.in_k = in_k
        self.use_global_pool = use_global_pool
        self.share_ff_td_kernels = share_ff_td_kernels

        # hgru params
        self.hgru_fsiz = hgru_fsiz
        self.hgru_fanout = hgru_fanout_factor
        self.hgru_h2_k = hgru_h2_k
        if self.in_k*self.hgru_fanout % self.hgru_h2_k >0:
            raise ValueError('self.in_k*self.hgru_fanout must be an integer multiple of self.hgru_h2_k (e.g., in_k=8, hgru_fanout=3, hgru_h2_k=12)')

        # conv params
        self.ff_conv_fsiz = ff_conv_fsiz
        self.ff_conv_k = ff_conv_k
        self.ff_conv_strides = ff_conv_strides
        self.ff_pool_fsiz = ff_pool_fsiz
        self.ff_pool_strides = ff_pool_strides
        self.fb_conv_fsiz = fb_conv_fsiz
        self.fb_conv_k = fb_conv_k

        self.bn_param_initializer = {
                            'moving_mean': tf.constant_initializer(0., dtype=self.dtype),
                            'moving_variance': tf.constant_initializer(1., dtype=self.dtype),
                            'beta': tf.constant_initializer(0., dtype=self.dtype),
                            'gamma': tf.constant_initializer(0.1, dtype=self.dtype)
        }

        self.hgru0 = fgru_module_class.hGRU(self.var_scope + '/fgru',
                          self.in_k,
                          self.hgru_fanout,
                          self.hgru_h2_k,
                          self.hgru_fsiz,
                          use_3d=False,
                          symmetric_weights=False,
                          bn_reuse=False,
                          train=self.train,
                          dtype=self.dtype,
                          **fgru_layer_optional_args)
        self.hgru_td0 = fgru_module_class.hGRU(self.var_scope + '/fgru_td',
                          self.in_k,
                          self.hgru_fanout,
                          self.hgru_h2_k,
                          [1, 1],
                          use_3d=False,
                          symmetric_weights=False,
                          bn_reuse=False,
                          train=self.train,
                          dtype=self.dtype,
                          **fgru_layer_optional_args)

        print('>>>>>>>>>>>>>>>>>>>>>>IS_TRAINING: ' + str(self.train))

    def prepare_tensors(self):

        # HGRU KERNELS
        self.hgru0.prepare_tensors()
        self.hgru_td0.prepare_tensors()

        # FEEDFORWARD KERNELS
        lower_feats = self.hgru_h2_k
        for idx, (higher_feats, ff_dhw) in enumerate(
                zip(self.ff_conv_k, self.ff_conv_fsiz)):
            with tf.variable_scope(self.var_scope + '/ff_%s' % idx):
                setattr(
                    self,
                    'ff_%s_weights' % idx,
                    tf.get_variable(
                        name='weights',
                        dtype=self.dtype,
                        initializer=initialization.xavier_initializer(
                            shape=ff_dhw + [lower_feats, higher_feats],
                            dtype=self.dtype,
                            uniform=True),
                        trainable=True))
                lower_feats = higher_feats

        # FEEDBACK KERNELS
        lower_feats = self.in_k
        if not self.share_ff_td_kernels:
            for idx, (higher_feats, fb_dhw) in enumerate(
                    zip(self.fb_conv_k, self.fb_conv_fsiz)):
                with tf.variable_scope(self.var_scope + '/fb_%s' % idx):
                    setattr(
                        self,
                        'fb_%s_weights' % idx,
                        tf.get_variable(
                            name='weights',
                            dtype=self.dtype,
                            initializer=initialization.xavier_initializer(
                                shape=fb_dhw + [lower_feats, higher_feats],
                                dtype=self.dtype,
                                uniform=True),
                            trainable=True))
                lower_feats = higher_feats
        else:
            higher_feats = self.fb_conv_k[0]
            fb_dhw = self.fb_conv_fsiz[0]
            idx=0
            with tf.variable_scope(self.var_scope + '/fb_%s' % idx):
                setattr(
                    self,
                    'fb_%s_weights' % idx,
                    tf.get_variable(
                        name='weights',
                        dtype=self.dtype,
                        initializer=initialization.xavier_initializer(
                            shape=fb_dhw + [lower_feats, higher_feats],
                            dtype=self.dtype,
                            uniform=True),
                        trainable=True))

        for x in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.var_scope):
            fgru = 0
            ff_fb = 0
            prod = np.prod(x.get_shape().as_list())
            if ('fgru' in x.name):
                fgru += prod
            else:
                ff_fb += prod
            print('>>>>>>>>>>>>>>>>>>>>>>TRAINABLE VARS: ' + 'fgrus(' + str(fgru) + ') ff&fb(' + str(ff_fb) + ')')
            print('>>>>>>>>>>>>>>>>>>>>>>TRAINABLE VARS: ' + 'total(' + str(fgru + ff_fb) + ')')

    def resize_x_to_y(
            self,
            x,
            y,
            kernel,
            strides,
            mode='transpose'):
        """Resize activity x to the size of y using interpolation."""
        y_size = y.get_shape().as_list()
        if mode == 'resize':
            return tf.image.resize_images(
                x,
                y_size[:-1],
                kernel,
                align_corners=True)
        elif mode == 'transpose':
            resized = tf.nn.conv2d_transpose(
                value=x,
                filter=kernel,
                output_shape=y_size,
                strides=[1] + strides + [1],
                padding='SAME',
                name='resize_x_to_y')
            return resized
        else:
            raise NotImplementedError(mode)

    def generic_combine(self, tensor1, tensor2, w1, w3):
        stacked = w1*tensor1 + tensor2 + (w3*tensor1)*tensor2
        return stacked

    def full(self, i0, x, l0_h1, l0_h2, td0_h1):
        # HGRU
        l0_h1, l0_h2 = self.hgru0.run(x, l0_h1, l0_h2)
        ff = tf.contrib.layers.batch_norm(
            inputs=l0_h2,
            scale=True,
            center=True,
            fused=True,
            renorm=False,
            reuse=False,
            scope=None,
            param_initializers=self.bn_param_initializer,
            updates_collections=None,
            is_training=self.train)
        ff = tf.nn.relu(ff)
        ff_list = []
        ff_list.append(ff)

        # FEEDFORWARD
        for idx, (conv_fsiz, conv_k, conv_str, pool_fsiz, pool_str) in enumerate(zip(self.ff_conv_fsiz,
                                                                                     self.ff_conv_k,
                                                                                     self.ff_conv_strides,
                                                                                     self.ff_pool_fsiz,
                                                                                     self.ff_pool_strides)):
            with tf.variable_scope(self.var_scope + '/ff_%s' % idx, reuse=tf.AUTO_REUSE):
                weights = tf.get_variable("weights")
            # POOL
            ff = max_pool(
                bottom=ff,
                k=[1]+pool_fsiz+[1],
                s=[1]+pool_str+[1],
                name='ff_pool_hgru')
            # CONV
            ff = tf.nn.conv2d(
                input=ff,
                filter=weights,
                strides=conv_str,
                padding='SAME')
            ff = tf.contrib.layers.batch_norm(
                inputs=ff,
                scale=True,
                center=True,
                fused=True,
                renorm=False,
                reuse=False,
                scope=None,
                param_initializers=self.bn_param_initializer,
                updates_collections=None,
                is_training=self.train)
            ff = tf.nn.relu(ff)
            ff_list.append(ff)

        # GLOBAL POOL and then TILE
        if self.use_global_pool:
            top_map_shape = ff_list[-1].get_shape().as_list()
            ff = global_pool(
                    bottom=ff,
                    name='global_pool',
                    aux={})
            ff = tf.tile(tf.expand_dims(tf.expand_dims(ff, 1),1), [1] + top_map_shape[1:3] + [1])

        # TOPDOWN
        fb = ff
        if not self.share_ff_td_kernels:
            scp = 'fb_'
        else:
            scp = 'ff_'
        for idx in range(len(ff_list))[::-1]:
            if idx != 0:
                with tf.variable_scope(self.var_scope + '/' + scp + '%s' % (idx-1), reuse=True):
                    weights = tf.get_variable("weights")
                fb = self.resize_x_to_y(x=fb, y=ff_list[idx-1],
                                        kernel=weights,
                                        mode='transpose',
                                        strides=self.ff_pool_strides[idx-1])
                fb = tf.contrib.layers.batch_norm(
                    inputs=fb,
                    scale=True,
                    center=True,
                    fused=True,
                    renorm=False,
                    reuse=False,
                    scope=None,
                    param_initializers=self.bn_param_initializer,
                    updates_collections=None,
                    is_training=self.train)
                fb = tf.nn.relu(fb)
            else:
                with tf.variable_scope(self.var_scope + '/fb_0', reuse=True):
                    weights = tf.get_variable("weights")
                fb = self.resize_x_to_y(x=fb, y=x,
                                        kernel=weights,
                                        mode='transpose',
                                        strides=[1, 1])
                fb = tf.contrib.layers.batch_norm(
                    inputs=fb,
                    scale=True,
                    center=True,
                    fused=True,
                    renorm=False,
                    reuse=False,
                    scope=None,
                    param_initializers=self.bn_param_initializer,
                    updates_collections=None,
                    is_training=self.train)
                fb = tf.nn.relu(fb)

        # HGRU_TD
        td0_h1, l0_h2 = self.hgru_td0.run(fb, td0_h1, l0_h2)

        # Iterate loop
        i0 += 1
        return i0, x, l0_h1, l0_h2, td0_h1


    def just_ff(self, x, l0_h1, l0_h2):
        # HGRU
        l0_h1, l0_h2 = self.hgru0.run(x, l0_h1, l0_h2)
        ff = tf.contrib.layers.batch_norm(
            inputs=l0_h2,
            scale=True,
            center=True,
            fused=True,
            renorm=False,
            reuse=False,
            scope=None,
            param_initializers=self.bn_param_initializer,
            updates_collections=None,
            is_training=self.train)
        ff = tf.nn.relu(ff)
        ff_list = []
        ff_list.append(ff)

        # FEEDFORWARD
        for idx, (conv_fsiz, conv_k, conv_str, pool_fsiz, pool_str) in enumerate(zip(self.ff_conv_fsiz,
                                                                                     self.ff_conv_k,
                                                                                     self.ff_conv_strides,
                                                                                     self.ff_pool_fsiz,
                                                                                     self.ff_pool_strides)):
            with tf.variable_scope(self.var_scope + '/ff_%s' % idx, reuse=tf.AUTO_REUSE):
                weights = tf.get_variable("weights")
            # POOL
            ff = max_pool(
                bottom=ff,
                k=[1]+pool_fsiz+[1],
                s=[1]+pool_str+[1],
                name='ff_pool_hgru')
            # CONV
            ff = tf.nn.conv2d(
                input=ff,
                filter=weights,
                strides=conv_str,
                padding='SAME')
            ff = tf.contrib.layers.batch_norm(
                inputs=ff,
                scale=True,
                center=True,
                fused=True,
                renorm=False,
                reuse=False,
                scope=None,
                param_initializers=self.bn_param_initializer,
                updates_collections=None,
                is_training=self.train)
            ff = tf.nn.relu(ff)
            ff_list.append(ff)

        # GLOBAL POOL and then TILE
        if self.use_global_pool:
            top_map_shape = ff_list[-1].get_shape().as_list()
            ff = global_pool(
                    bottom=ff,
                    name='global_pool',
                    aux={})
        return ff

    def condition(self, i0, x, l0_h1, l0_h2, td0_h1):
        """While loop halting condition."""
        return i0 < self.timesteps

    def compute_shape(self, in_length, stride):
        if in_length % stride == 0:
            return in_length/stride
        else:
            return in_length/stride + 1

    def build(self, x):
        """Run the backprop version of the Circuit."""
        self.prepare_tensors()
        i0 = tf.constant(0)

        # Calculate l2 hidden state size
        x_shape = x.get_shape().as_list()
        l0_h1_shape = x_shape[:-1] + [x_shape[-1] * self.hgru_fanout]
        l0_h2_shape = x_shape[:-1] + [self.hgru_h2_k]
        td0_h1_shape = l0_h2_shape[:-1] + [x_shape[-1] * self.hgru_fanout]

        # Initialize hidden layer activities
        l0_h1 = tf.zeros(l0_h1_shape, dtype=self.dtype)
        l0_h2 = tf.zeros(l0_h2_shape, dtype=self.dtype)
        td0_h1 = tf.zeros(td0_h1_shape, dtype=self.dtype)

        # While loop
        # elems = [i0, x, l0_h1, l0_h2, td0_h1]
        # returned = tf.while_loop(
        #     self.condition,
        #     self.full,
        #     loop_vars=elems,
        #     back_prop=True,
        #     swap_memory=False)
        # i0, x, l0_h1, l0_h2, td0_h1 = returned

        # For loop
        for i in range(self.timesteps):
            i0, x, l0_h1, l0_h2, td0_h1 =\
                self.full(i0, x, l0_h1, l0_h2, td0_h1)

        # Prepare output
        top = self.just_ff(x, l0_h1, l0_h2)

        return l0_h2, top
