"""Contextual model with partial filters."""
import warnings
import numpy as np
import tensorflow as tf
import initialization
from pooling import max_pool3d
import gradients

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
            hgru_fsiz,
            hgru_fanout,
            hgru_h2_k,
            ff_conv_fsiz,
            ff_conv_k,
            ff_conv_strides,
            ff_kpool_multiplier,
            ff_pool_fsiz,
            ff_pool_strides,
            fb_conv_fsiz,
            fb_conv_k,
            train=True,
            dtype=tf.bfloat16):

        # global params
        self.var_scope = var_scope
        self.timesteps = timesteps
        self.dtype = dtype
        self.train = train
        self.in_k = in_k

        # hgru params
        self.hgru_fsiz = hgru_fsiz
        self.hgru_fanout = hgru_fanout
        self.hgru_h2_k = hgru_h2_k

        # conv params
        self.ff_conv_fsiz = ff_conv_fsiz
        self.ff_conv_k = ff_conv_k
        self.ff_conv_strides = ff_conv_strides
        self.ff_kpool_multiplier=ff_kpool_multiplier
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

        from .layers.v6_mely import hGRU
        self.hgru0 = hGRU(self.var_scope + '/hgru0',
                          self.in_k,
                          self.hgru_fanout,
                          self.hgru_h2_k[0],
                          self.hgru_fsiz[0],
                          use_3d=True,
                          symmetric_weights=True,
                          bn_reuse=False,
                          train=self.train,
                          dtype=self.dtype)
        self.hgru1 = hGRU(self.var_scope + '/hgru1',
                          self.ff_conv_k[0],
                          self.hgru_fanout,
                          self.hgru_h2_k[1],
                          self.hgru_fsiz[1],
                          use_3d=True,
                          symmetric_weights=True,
                          bn_reuse=False,
                          train=self.train,
                          dtype=self.dtype)
        self.hgru_td1 = hGRU(self.var_scope + '/hgru_td1',
                          self.hgru_h2_k[1],
                          self.hgru_fanout,
                          self.hgru_h2_k[1],
                          [1, 1, 1],
                          use_3d=True,
                          symmetric_weights=True,
                          bn_reuse=False,
                          train=self.train,
                          dtype=self.dtype)
        self.hgru_td0 = hGRU(self.var_scope + '/hgru_td0',
                          self.hgru_h2_k[0],
                          self.hgru_fanout,
                          self.hgru_h2_k[0],
                          [1, 1, 1],
                          use_3d=True,
                          symmetric_weights=False,
                          bn_reuse=False,
                          train=self.train,
                          dtype=self.dtype)

        print('>>>>>>>>>>>>>>>>>>>>>>IS_TRAINING: ' + str(self.train))

    def prepare_tensors(self):

        # HGRU KERNELS
        self.hgru0.prepare_tensors()
        self.hgru1.prepare_tensors()
        self.hgru_td1.prepare_tensors()
        self.hgru_td0.prepare_tensors()

        # FEEDFORWARD KERNELS
        lower_feats = self.in_k
        for idx, (higher_feats, ff_dhw) in enumerate(
                zip(self.ff_conv_k, self.ff_conv_fsiz)):
            with tf.variable_scope('ff_%s' % idx):
                if idx<2:
                    # last conv layer doesn't have spot weights
                    setattr(
                        self,
                        'ff_%s_spot_x' % idx,
                        tf.get_variable(
                            name='spot_x',
                            dtype=self.dtype,
                            initializer=initialization.xavier_initializer(
                                shape=[1,1,1,1] + [1],#[lower_feats],
                                dtype=self.dtype,
                                uniform=True),
                            trainable=True))
                    # last conv layer doesn't have spot weights
                    setattr(
                        self,
                        'ff_%s_spot_xy' % idx,
                        tf.get_variable(
                            name='spot_xy',
                            dtype=self.dtype,
                            initializer=initialization.xavier_initializer(
                                shape=[1,1,1,1] + [lower_feats],#[lower_feats],
                                dtype=self.dtype,
                                uniform=True),
                            trainable=True))
                setattr(
                    self,
                    'ff_%s_weights' % idx,
                    tf.get_variable(
                        name='weights',
                        dtype=self.dtype,
                        initializer=initialization.xavier_initializer(
                            shape=ff_dhw + [lower_feats, higher_feats*self.ff_kpool_multiplier],
                            dtype=self.dtype,
                            uniform=True),
                        trainable=True))
                lower_feats = higher_feats

        # FEEDBACK KERNELS
        lower_feats = self.in_k
        for idx, (higher_feats, fb_dhw) in enumerate(
                zip(self.fb_conv_k, self.fb_conv_fsiz)):
            with tf.variable_scope('fb_%s' % idx) as scope:
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
            resized = tf.nn.conv3d_transpose(
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

    def full(self, i0, x, l0_h1, l0_h2, l1_h1, l1_h2, td0_h1, td1_h1):
        # HGRU 0
        l0_h1, l0_h2 = self.hgru0.run(x, l0_h1, l0_h2)
        ff0 = tf.contrib.layers.batch_norm(
            inputs=l0_h2,
            scale=True,
            center=True,
            fused=True,
            renorm=False,
            param_initializers=self.bn_param_initializer,
            updates_collections=None,
            reuse=None,
            is_training=self.train)

        # FEEDFORWARD 0
        idx = 0
        with tf.variable_scope('ff_%s' % idx, reuse=tf.AUTO_REUSE):
            spot_weights_x = tf.get_variable("spot_x")
            spot_weights_xy = tf.get_variable("spot_xy")
            weights = tf.get_variable("weights")
        ff0 = self.generic_combine(
            x,
            ff0,
            spot_weights_x, spot_weights_xy)
        ff0 = tf.nn.elu(ff0) + 1
        ff0 = tf.nn.conv3d(
            input=ff0,
            filter=weights,
            strides=self.ff_conv_strides[idx],
            padding='SAME')
        if self.ff_kpool_multiplier > 1:
            low_k = 0
            running_max = ff0[:,:,:,:,low_k:low_k+self.ff_conv_k[idx]]
            for i in range(self.ff_kpool_multiplier-1):
                low_k += self.ff_conv_k[idx]
                running_max = tf.maximum(running_max, ff0[:,:,:,:,low_k:low_k+self.ff_conv_k[idx]])
            ff0 = running_max
        ff0 = tf.contrib.layers.batch_norm(
            inputs=ff0,
            scale=True,
            center=True,
            fused=True,
            renorm=False,
            param_initializers=self.bn_param_initializer,
            updates_collections=None,
            is_training=self.train)
        ff0 = tf.nn.elu(ff0) + 1

        # POOL
        ff0 = max_pool3d(
            bottom=ff0,
            k=self.ff_pool_fsiz[idx],
            s=self.ff_pool_strides[idx],
            name='ff_pool_%s' % idx)

        # HGRU 1
        l1_h1, l1_h2 = self.hgru1.run(ff0, l1_h1, l1_h2)
        ff1 = tf.contrib.layers.batch_norm(
            inputs=l1_h2,
            scale=True,
            center=True,
            fused=True,
            renorm=False,
            param_initializers=self.bn_param_initializer,
            updates_collections=None,
            reuse=None,
            is_training=self.train)

        # FEEDFORWARD 1
        idx = 1
        with tf.variable_scope('ff_%s' % idx, reuse=True):
            spot_weights_x = tf.get_variable("spot_x")
            spot_weights_xy = tf.get_variable("spot_xy")
            weights = tf.get_variable("weights")
        ff1 = self.generic_combine(
            ff0,
            ff1,
            spot_weights_x, spot_weights_xy)
        ff1 = tf.nn.elu(ff1) + 1
        ff1 = tf.nn.conv3d(
            input=ff1,
            filter=weights,
            strides=self.ff_conv_strides[idx],
            padding='SAME')
        if self.ff_kpool_multiplier > 1:
            low_k = 0
            running_max = ff1[:,:,:,:,low_k:low_k+self.ff_conv_k[idx]]
            for i in range(self.ff_kpool_multiplier-1):
                low_k += self.ff_conv_k[idx]
                running_max = tf.maximum(running_max, ff1[:,:,:,:,low_k:low_k+self.ff_conv_k[idx]])
            ff1 = running_max
        ff1 = tf.contrib.layers.batch_norm(
            inputs=ff1,
            scale=True,
            center=True,
            fused=True,
            renorm=False,
            param_initializers=self.bn_param_initializer,
            updates_collections=None,
            is_training=self.train)
        ff1 = tf.nn.elu(ff1) + 1

        # POOL
        ff1 = max_pool3d(
            bottom=ff1,
            k=self.ff_pool_fsiz[idx],
            s=self.ff_pool_strides[idx],
            name='ff_pool_%s' % idx)

        # HGRU 2
        # l2_h1, l2_h2 = self.hgru2.run(ff1, l2_h1, l2_h2)
        # ff2 = tf.contrib.layers.batch_norm(
        #     inputs=l2_h2,
        #     scale=True,
        #     center=True,
        #     fused=True,
        #     renorm=False,
        #     param_initializers=self.bn_param_initializer,
        #     updates_collections=None,
        #     reuse=None,
        #     is_training=self.train)

        # FEEDFORWARD 2
        idx = 2
        with tf.variable_scope('ff_%s' % idx, reuse=True):
            # spot_weights_x = tf.get_variable("spot_x")
            # spot_weights_xy = tf.get_variable("spot_xy")
            weights = tf.get_variable("weights")
        # ff2 = self.generic_combine(
        #     ff1,
        #     ff2,
        #     spot_weights_x, spot_weights_xy)
        # ff2 = tf.nn.elu(ff2) + 1
        ff2 = ff1
        ff2 = tf.nn.conv3d(
            input=ff2,
            filter=weights,
            strides=self.ff_conv_strides[idx],
            padding='SAME')

        if self.ff_kpool_multiplier > 1:
            low_k = 0
            running_max = ff2[:,:,:,:,low_k:low_k+self.ff_conv_k[idx]]
            for i in range(self.ff_kpool_multiplier-1):
                low_k += self.ff_conv_k[idx]
                running_max = tf.maximum(running_max, ff2[:,:,:,:,low_k:low_k+self.ff_conv_k[idx]])
            ff2 = running_max
        ff2 = tf.contrib.layers.batch_norm(
            inputs=ff2,
            scale=True,
            center=False,
            fused=True,
            renorm=False,
            param_initializers=self.bn_param_initializer,
            updates_collections=None,
            is_training=self.train)
        ff2 = tf.nn.elu(ff2) + 1

        # POOL
        ff2 = max_pool3d(
            bottom=ff2,
            k=self.ff_pool_fsiz[idx],
            s=self.ff_pool_strides[idx],
            name='ff_pool_%s' % idx)

        # FEEDBACK 2
        idx=2
        with tf.variable_scope('fb_%s' % idx, reuse=True):
            weights = tf.get_variable("weights")
        fb2 = self.resize_x_to_y(x=ff2, y=ff1,
                                  kernel=weights,
                                  mode='transpose',
                                  strides=self.ff_pool_strides[2])
        fb2 = tf.contrib.layers.batch_norm(
            inputs=fb2,
            scale=True,
            center=True,
            fused=True,
            renorm=False,
            param_initializers=self.bn_param_initializer,
            updates_collections=None,
            is_training=self.train)
        fb2 = tf.nn.elu(fb2) + 1

        # HGRU_TD 2
        # td2_h1, l2_h2 = self.hgru_td2.run(fb2, td2_h1, l2_h2)

        # FEEDBACK 1
        idx=1
        with tf.variable_scope('fb_%s' % idx, reuse=True):
            weights = tf.get_variable("weights")
        fb1 = self.resize_x_to_y(x=fb2, y=ff0,
                                  kernel=weights,
                                  mode='transpose',
                                  strides=self.ff_pool_strides[1])
        fb1 = tf.contrib.layers.batch_norm(
            inputs=fb1,
            scale=True,
            center=True,
            fused=True,
            renorm=False,
            param_initializers=self.bn_param_initializer,
            updates_collections=None,
            is_training=self.train)
        fb1 = tf.nn.elu(fb1) + 1

        # HGRU_TD 1
        td1_h1, l1_h2 = self.hgru_td1.run(fb1, td1_h1, l1_h2)

        # FEEDBACK 0
        idx=0
        with tf.variable_scope('fb_%s' % idx, reuse=True):
            weights = tf.get_variable("weights")
        fb0 = self.resize_x_to_y(x=fb1, y=x,
                                  kernel=weights,
                                  mode='transpose',
                                  strides=self.ff_pool_strides[0])
        fb0 = tf.contrib.layers.batch_norm(
            inputs=fb0,
            scale=True,
            center=True,
            fused=True,
            renorm=False,
            param_initializers=self.bn_param_initializer,
            updates_collections=None,
            is_training=self.train)
        fb0 = tf.nn.elu(fb0) + 1

        # HGRU_TD 0
        td0_h1, l0_h2 = self.hgru_td0.run(fb0, td0_h1, l0_h2)

        # Iterate loop
        i0 += 1
        return i0, x, l0_h1, l0_h2, l1_h1, l1_h2, td0_h1, td1_h1


    def condition(self, i0, x, l0_h1, l0_h2, l1_h1, l1_h2, td0_h1, td1_h1):
        """While loop halting condition."""
        return i0 < self.timesteps

    def compute_shape(self, in_length, stride):
        if in_length % stride == 0:
            return in_length/stride
        else:
            return in_length/stride + 1

    def build(self, x, seed):
        """Run the backprop version of the Circuit."""
        self.prepare_tensors()
        i0 = tf.constant(0)

        # Calculate l2 hidden state size
        x_shape = x.get_shape().as_list()
        l0_h1_shape = x_shape[:-1] + [x_shape[-1] * self.hgru_fanout]
        l0_h2_shape = x_shape[:-1] + [self.hgru_h2_k[0]]
        l1_h1_shape = [
                x_shape[0],
                self.compute_shape(l0_h2_shape[1], self.ff_pool_strides[0][0]),
                self.compute_shape(l0_h2_shape[2], self.ff_pool_strides[0][1]),
                self.compute_shape(l0_h2_shape[3], self.ff_pool_strides[0][2]),
                self.ff_conv_k[0] * self.hgru_fanout]
        l1_h2_shape = l1_h1_shape[:-1] + [self.hgru_h2_k[1]]
        # l2_h1_shape = [
        #         x_shape[0],
        #         self.compute_shape(l1_h2_shape[1], self.ff_pool_strides[1][0]),
        #         self.compute_shape(l1_h2_shape[2], self.ff_pool_strides[1][1]),
        #         self.compute_shape(l1_h2_shape[3], self.ff_pool_strides[1][2]),
        #         self.ff_conv_k[1] * self.hgru_fanout]
        # l2_h2_shape = l2_h1_shape[:-1] + [self.hgru_h2_k[2]]
        # td2_h1_shape = l2_h2_shape[:-1] + [self.hgru_h2_k[2] * self.hgru_fanout]
        td1_h1_shape = l1_h2_shape[:-1] + [self.hgru_h2_k[1] * self.hgru_fanout]
        td0_h1_shape = l0_h2_shape[:-1] + [self.hgru_h2_k[0] * self.hgru_fanout]

        # Initialize hidden layer activities
        l0_h1 = tf.zeros(l0_h1_shape, dtype=self.dtype)
        l0_h2 = tf.ones(l0_h2_shape, dtype=self.dtype)*seed*2 - 1
        l1_h1 = tf.zeros(l1_h1_shape, dtype=self.dtype)
        l1_h2 = tf.zeros(l1_h2_shape, dtype=self.dtype)
        # l2_h1 = tf.zeros(l2_h1_shape, dtype=self.dtype)
        # l2_h2 = tf.zeros(l2_h2_shape, dtype=self.dtype)
        # td2_h1 = tf.zeros(td2_h1_shape, dtype=self.dtype)
        td1_h1 = tf.zeros(td1_h1_shape, dtype=self.dtype)
        td0_h1 = tf.zeros(td0_h1_shape, dtype=self.dtype)

        # While loop
        elems = [i0, x, l0_h1, l0_h2, l1_h1, l1_h2, td0_h1, td1_h1]

        returned = tf.while_loop(
            self.condition,
            self.full,
            loop_vars=elems,
            back_prop=True,
            swap_memory=False)

        # Prepare output
        i0, x, l0_h1, l0_h2, l1_h1, l1_h2, td0_h1, td1_h1 = returned

        return l0_h2
