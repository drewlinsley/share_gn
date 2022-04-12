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

        from .layers.v6_mely import hGRU
        self.hgru = hGRU(self.var_scope,
                          self.in_k,
                          self.hgru_fanout,
                          self.hgru_h2_k[0],
                          self.hgru_fsiz[0],
                          use_3d=True,
                          symmetric_weights=True,
                          bn_reuse=False,
                          train=self.train,
                          dtype=self.dtype)

        print('>>>>>>>>>>>>>>>>>>>>>>IS_TRAINING: ' + str(self.train))

    def prepare_tensors(self):

        # HGRU KERNELS
        self.hgru.prepare_tensors()

    def full(self, i0, x, l0_h1, l0_h2):
        # HGRU 0
        l0_h1, l0_h2 = self.hgru0.run(x, l0_h1, l0_h2)

        # Iterate loop
        i0 += 1
        return i0, x, l0_h1, l0_h2, l1_h1, l1_h2, td0_h1, td1_h1


    def condition(self, i0, x, l0_h1, l0_h2):
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
        l0_h2_shape = x_shape[:-1] + [self.hgru_h2_k[0]]

        # Initialize hidden layer activities
        l0_h1 = tf.zeros(l0_h1_shape, dtype=self.dtype)
        l0_h2 = tf.zeros(l0_h2_shape, dtype=self.dtype)

        # While loop
        elems = [i0, x, l0_h1, l0_h2]

        returned = tf.while_loop(
            self.condition,
            self.full,
            loop_vars=elems,
            back_prop=True,
            swap_memory=False)

        # Prepare output
        i0, x, l0_h1, l0_h2 = returned

        return l0_h2
