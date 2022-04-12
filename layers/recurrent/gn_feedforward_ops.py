import warnings
import numpy as np
import tensorflow as tf
from layers.feedforward import normalization
from layers.feedforward import pooling


class GNFFOps(object):
    """Methods for initializing gammanet parameters."""
    def __init__(self):
        pass

    def __call__(self):
        pass

    def apply_skips(self, activity, layer_id, encoder_layer):
        """Apply skips if requested."""
        if (
            self.skip_connections and
                self.fgru_connectivity is not 'all_to_all' and
                hasattr(self, '%s_for_skip' % encoder_layer)):
            print(
                'Skipping from %s to %s' % (
                    layer_id, '%s_for_skip' % encoder_layer))
            skip_activity = getattr(
                self,
                '%s_for_skip' % encoder_layer)
            activity += skip_activity
        return activity

    def resize_x_to_y(
            self,
            x,
            y,
            name,
            encoder_layer,
            i0,
            reps,
            mode='transpose',
            use_bias=True):
        """Resize activity x to the size of y using interpolation."""
        if isinstance(y, np.ndarray):
            y = y.tolist()

        def tf_resize(im, size):
            """Wrapper for resizing."""
            if isinstance(
                    self.image_resize, str) and self.image_resize is 'unpool':
                pool_idx = getattr(
                    self, 'pool_idx_%s' % encoder_layer)
                return pooling.unpool_with_argmax_layer(
                    bottom=im,
                    ind=pool_idx,
                    name='unpool_%s' % name,
                    filter_size=self.pool_kernel,
                    data_format=self.data_format)
            else:
                if self.data_format is 'NCHW':
                    return tf.transpose(
                        self.image_resize(
                            images=tf.transpose(im, (0, 2, 3, 1)),
                            size=size[1:],
                            align_corners=True), (0, 3, 1, 2))
                elif self.data_format is 'NHWC':
                    return self.image_resize(
                        images=im,
                        size=size[:-1],
                        align_corners=True)

        if mode == 'resize':
            raise NotImplementedError
            return tf.image.resize_images(
                x,
                y[:-1],
                self.resize_kernel,
                align_corners=True)
        elif mode == 'transpose':
            if self.N is None:
                N = 1
                warnings.warn('Found None for batch size. Forcing to 1.')
            else:
                N = self.N
            if self.separable_upsample:
                # Pair bilinear resize w/ 1x1 conv
                if isinstance(x, list):
                    resized = []
                    for ix in x:
                        resized += [tf_resize(im=ix, size=y)]
                    resized = tf.concat(resized, axis=-1)
                else:
                    resized = tf_resize(im=x, size=y)
            else:
                if isinstance(x, list):
                    raise NotImplementedError(
                        'No all-to-all upsample support.')
                resize_bias = getattr(self, 'resize_bias_%s' % name)
                resize_kernel = getattr(self, 'resize_kernel_%s' % name)
                resized = tf.nn.conv2d_transpose(
                    value=x,
                    filter=resize_kernel,
                    output_shape=[N] + y,
                    strides=self.up_strides,
                    padding='SAME',
                    data_format=self.data_format,
                    name='resize_x_to_y_%s' % name)
                resized = self.bias_add(
                    resized,
                    resize_bias,
                    name='resize_bias_%s' % name)

            if self.time_skips:
                raise NotImplementedError

            if self.upsample_convs:
                resized = self.conv_tower(
                    activity=resized,
                    reps=reps,
                    layer_id=name,
                    compression=False,
                    i0=i0,
                    tag='conv_resize',
                    use_nl=self.upsample_nl,
                    skip=encoder_layer)
            return resized
        else:
            raise NotImplementedError(mode)

    def nl_bn(
            self,
            activity,
            ff_name,
            nl,
            normalization_type,
            scope=None,
            gamma=None,
            beta=None,
            center=False,
            scale=False,
            pre_nl_bn=False):
        """Wrapper for applying normalization and nonlinearity."""
        assert center is not None, 'No center provided.'
        assert scale is not None, 'No scale provided.'
        if pre_nl_bn:
            if scope is None:
                with tf.variable_scope(
                        ff_name,
                        reuse=self.scope_reuse) as scope:
                    activity = normalization.apply_normalization(
                        activity=activity,
                        normalization_type=normalization_type,
                        data_format=self.data_format,
                        training=self.train_norm_moments,
                        trainable=self.train_norm_params,
                        scale=scale,
                        center=center,
                        scope=scope,
                        name=ff_name,
                        reuse=self.scope_reuse)
            else:
                activity = normalization.apply_normalization(
                    activity=activity,
                    normalization_type=normalization_type,
                    data_format=self.data_format,
                    training=self.train_norm_moments,
                    trainable=self.train_norm_params,
                    scale=scale,
                    center=center,
                    scope=scope,
                    name=ff_name,
                    reuse=self.scope_reuse)
            if gamma is not None:
                activity *= gamma
            if beta is not None:
                activity += beta
            if not nl:
                return activity
            else:
                return nl(activity)
        else:
            if nl:
                activity = nl(activity)
            if scope is None:
                with tf.variable_scope(
                        ff_name,
                        reuse=self.scope_reuse) as scope:
                    activity = normalization.apply_normalization(
                        activity=activity,
                        normalization_type=normalization_type,
                        data_format=self.data_format,
                        training=self.train_norm_moments,
                        trainable=self.train_norm_params,
                        scale=scale,
                        center=center,
                        scope=scope,
                        name=ff_name,
                        reuse=self.scope_reuse)
            else:
                activity = normalization.apply_normalization(
                    activity=activity,
                    normalization_type=normalization_type,
                    data_format=self.data_format,
                    training=self.train_norm_moments,
                    trainable=self.train_norm_params,
                    scale=scale,
                    center=center,
                    scope=scope,
                    name=ff_name,
                    reuse=self.scope_reuse)
            if gamma is not None:
                activity = activity * gamma
            if beta is not None:
                activity = activity + beta
            return activity

    def conv_tower(
            self,
            activity,
            reps,
            layer_id,
            compression,
            i0,
            use_nl=True,
            skip=None,
            tag='ffdrive'):
        """Build the intermediate conv tower to expand RF size."""
        for il in range(reps):
            if self.recurrent_ff:
                # Calculate the gate activity if requested
                prev_activity = getattr(self, '%s_%s_%s' % (tag, layer_id, il))
                ff_gate_kernel = getattr(
                    self,
                    '%s_gate_kernel_prev_%s_%s' % (tag, layer_id, il))
                ff_cand_gate_kernel = getattr(
                    self,
                    '%s_gate_kernel_cand_%s_%s' % (tag, layer_id, il))
                ff_cand_hidden_kernel = getattr(
                    self,
                    '%s_kernel_hidden_%s_%s' % (tag, layer_id, il))
                ff_gate_bias = getattr(
                    self,
                    '%s_gate_bias_%s_%s' % (tag, layer_id, il))
                beta = getattr(self, '%s_gate_prev_bn_beta_%s_%s' % (
                    tag, layer_id, il))
                gamma = getattr(self, '%s_gate_prev_bn_gamma_%s_%s' % (
                    tag, layer_id, il))
                ff_gate_activity = self.nl_bn(
                    self.conv_op(
                        input=prev_activity,
                        filter=ff_gate_kernel,
                        strides=self.strides,
                        padding=self.ff_padding,
                        data_format=self.data_format,
                        name='gate_prev_%s_%s_%s' % (tag, layer_id, il)),
                    ff_name='%s_gate_prev_bn_%s_%s_%s' % (
                        tag, layer_id, il, i0),
                    normalization_type=self.ff_normalization_type,
                    gamma=gamma,
                    beta=beta,
                    nl=None)

            # Calculate FF drive
            if reps == 4 and self.residual and il == 0:
                activity = self.conv_op(
                    input=activity,
                    filter=getattr(
                        self, '%s_projection_kernel_%s_%s' % (
                            tag, layer_id, il)),
                    strides=self.strides,
                    padding=self.ff_padding,
                    data_format=self.data_format,
                    name='conv_projection_%s_%s_%s' % (tag, layer_id, il))
                beta = getattr(self, '%s_branch_bn_beta_%s_%s' % (
                    tag, layer_id, il))
                gamma = getattr(self, '%s_branch_bn_gamma_%s_%s' % (
                    tag, layer_id, il))
                activity = self.nl_bn(
                    activity=activity,
                    ff_name='%s_branch_bn_%s_%s_%s' % (tag, layer_id, il, i0),
                    nl=None,
                    normalization_type=self.ff_normalization_type,
                    gamma=gamma,
                    beta=beta)
                activity = self.ff_nl(beta + gamma * activity)
                skip_path = tf.identity(activity)

            # Apply skips
            if reps == 4 and self.residual and il == (reps - 1):
                activity += skip_path
            elif il == 0 and isinstance(skip, int):
                activity = self.apply_skips(
                    activity=activity,
                    layer_id=layer_id,
                    encoder_layer=skip)

            # if il == 0 and isinstance(skip, int):
            #     activity = self.apply_skips(
            #         activity=activity,
            #         layer_id=layer_id,
            #         encoder_layer=skip)

            if self.separable_convs:
                activity = tf.nn.separable_conv2d(
                    input=activity,
                    depthwise_filter=getattr(
                        self,
                        '%s_kernel_spatial_%s_%s' % (tag, layer_id, il)),
                    pointwise_filter=getattr(
                        self,
                        '%s_kernel_channel_%s_%s' % (tag, layer_id, il)),
                    strides=self.strides,
                    padding=self.ff_padding,
                    data_format=self.data_format,
                    name='conv_%s_%s_%s' % (tag, layer_id, il))
            else:
                activity = self.conv_op(
                    input=activity,
                    filter=getattr(
                        self, '%s_kernel_%s_%s' % (tag, layer_id, il)),
                    strides=self.strides,
                    padding=self.ff_padding,
                    data_format=self.data_format,
                    name='conv_%s_%s_%s' % (tag, layer_id, il))
            bias = getattr(self, '%s_bias_%s_%s' % (tag, layer_id, il))
            activity = self.bias_add(
                activity,
                bias,
                name='bias_%s_%s_%s' % (tag, layer_id, il))

            # EXPERIMENTAL: Add skip connections through time
            if self.time_skips:
                raise NotImplementedError

            # Allow for skips
            ff_name = 'bn_%s_%s_%s' % (tag, layer_id, il)
            if not self.while_loop:
                ff_name = '%s_t%s' % (ff_name, i0)

            # Integrate with previous FF drive if requested
            if self.recurrent_ff:
                beta = getattr(
                    self, '%s_gate_cand_bn_beta_%s_%s' % (tag, layer_id, il))
                gamma = getattr(
                    self, '%s_gate_cand_bn_gamma_%s_%s' % (tag, layer_id, il))
                ff_cand_gate_activity = self.nl_bn(
                    self.conv_op(
                        input=activity,
                        filter=ff_cand_gate_kernel,
                        strides=self.strides,
                        padding=self.ff_padding,
                        data_format=self.data_format,
                        name='gate_cand_%s_%s_%s' % (tag, layer_id, il)),
                    ff_name='%s_gate_cand_bn_%s_%s_%s' % (
                        tag, layer_id, il, i0),
                    nl=None,
                    normalization_type=self.ff_normalization_type,
                    gamma=gamma,
                    beta=beta)
                ff_gate_activity = self.gate_nl(self.bias_add(
                    ff_gate_activity + ff_cand_gate_activity,
                    ff_gate_bias))
                hidden_activity = self.conv_op(
                    input=prev_activity * ff_gate_activity,
                    filter=ff_cand_hidden_kernel,
                    strides=self.strides,
                    padding=self.ff_padding,
                    data_format=self.data_format,
                    name='hidden_cand_%s_%s_%s' % (tag, layer_id, il))

                if 0:
                    # To try: move bn/nonlin here
                    beta = getattr(self, '%s_hidden_bn_beta_%s_%s' % (
                        tag, layer_id, il))
                    gamma = getattr(self, '%s_hidden_bn_gamma_%s_%s' % (
                        tag, layer_id, il))
                    hidden_activity = self.nl_bn(
                        activity=hidden_activity,
                        ff_name='%s_%s_hidden_ff' % (tag, ff_name),
                        nl=None,
                        normalization_type=self.ff_normalization_type,
                        gamma=gamma,
                        beta=beta)
                    beta = getattr(self, '%s_bn_beta_%s_%s' % (
                        tag, layer_id, il))
                    gamma = getattr(self, '%s_bn_gamma_%s_%s' % (
                        tag, layer_id, il))
                    activity = self.nl_bn(
                        activity=activity,
                        ff_name='%s_%s_ff' % (tag, ff_name),
                        nl=None,
                        normalization_type=self.ff_normalization_type,
                        gamma=gamma,
                        beta=beta)
                    if use_nl:
                        activity = self.ff_nl(activity)
                activity = (ff_gate_activity) * prev_activity + (
                    1 - ff_gate_activity) * self.ff_nl(
                    activity + hidden_activity)
                setattr(
                    self,
                    '%s_%s_%s' % (tag, layer_id, il),
                    activity)
            if 1:
                activity = self.nl_bn(
                    activity=activity,
                    ff_name='%s_%s_ff' % (tag, ff_name),
                    normalization_type=self.ff_normalization_type,
                    beta=getattr(self, '%s_bn_beta_%s_%s' % (
                        tag, layer_id, il)),
                    gamma=getattr(self, '%s_bn_gamma_%s_%s' % (
                        tag, layer_id, il)),
                    nl=self.ff_nl)
                # if use_nl:
                #     activity = self.ff_nl(activity)

        # Store for upsample
        if self.skip_connections and not isinstance(skip, int):
            setattr(
                self,
                '%s_for_skip' % layer_id,
                activity)
        return activity

    def apply_pool(self, activity, layer_id):
        """Wrapper for pooling."""
        if isinstance(
                self.image_resize, str) and self.image_resize is 'unpool':
            activity, idx = pooling.max_pool_with_indices(
                bottom=activity,
                data_format=self.data_format,
                k=self.prepared_pool_kernel,
                s=self.prepared_pool_stride,
                p=self.ff_padding,
                name='ff_pool_%s' % layer_id)
            setattr(self, 'pool_idx_%s' % layer_id, idx)
        else:
            activity = pooling.max_pool(
                bottom=activity,
                data_format=self.data_format,
                k=self.prepared_pool_kernel,
                s=self.prepared_pool_stride,
                p=self.ff_padding,
                name='ff_pool_%s' % layer_id)
        return activity

    def upsample_router(self, activity, layer_id, encoder_layer, reps, i0):
        """Wrapper for applying fgru upsamples."""
        # if (self.force_horizontal and (i0 < (self.timesteps - 1))):
        #     # Only upsample on the final timestep
        #     return activity
        return self.upsample_ops(
            activity=activity,
            layer_id=layer_id,
            encoder_layer=encoder_layer,
            reps=reps,
            i0=i0)

    def upsample_ops(
            self,
            activity,
            layer_id,
            encoder_layer,
            reps,
            i0):
        """Apply upsampling."""
        # target = self.layer_shapes[layer_id]
        target = self.layer_shapes[encoder_layer]
        activity = self.resize_x_to_y(
            x=activity,
            y=target[1:],
            name=layer_id,
            encoder_layer=encoder_layer,
            reps=reps,
            i0=i0)

        # if self.ff_normalization_type:
        #     up_name = 'up_bn_%s' % layer_id
        #     if not self.while_loop:
        #         up_name = '%s_t%s' % (up_name, i0)
        #     activity = self.nl_bn(
        #         activity=activity,
        #         ff_name=up_name,
        #         nl=self.ff_nl)

        return activity

    def td_router(self, high_h, low_h, layer_id, i0):
        """Route to the appropriate topdown ops."""
        if self.force_bottom_up:  # or self.force_horizontal:
            # Only upsample on the final timestep
            return high_h
        return self.td_ops(
            high_h=high_h,
            low_h=low_h,
            layer_id=layer_id,
            i0=i0)

    def create_pad_mask(
            self,
            h,
            w,
            y,
            x,
            eps=1e-8):
        """Create a mask for padding according to
        https://arxiv.org/abs/1811.11718."""
        slide_window = x * x
        mask = tf.ones(shape=[1, h, w, 1])
        update_mask = tf.layers.conv2d(
            mask,
            filters=1,
            kernel_size=(y, x),
            kernel_initializer=tf.constant_initializer(1.0),
            strides=(1, 1),
            padding='same',
            use_bias=False,
            trainable=False)
        mask_ratio = slide_window / (update_mask + 1e-8)
        update_mask = tf.clip_by_value(update_mask, 0.0, 1.0)
        mask_ratio = mask_ratio * update_mask
        return update_mask, mask_ratio

    def conv_op(
            self,
            input,
            filter,
            strides,
            padding,
            data_format,
            name):
        """Wrapper for convolution."""
        if self.partial_padding:
            assert data_format is not 'NCHW'
            _, h, w, _ = input.get_shape().as_list()
            y, x, _, _ = filter.get_shape().as_list()
            _, mask_ratio = self.create_pad_mask(
                h=h,
                w=w,
                y=y,
                x=x)
        activity = tf.nn.conv2d(
            input=input,
            filter=filter,
            strides=strides,
            padding=padding,
            data_format=data_format,
            name=name)
        if self.partial_padding:
            return activity * mask_ratio
        else:
            return activity

    def td_ops(self, high_h, low_h, layer_id, i0):
        """Apply top-down operations."""
        error, fb_act = self.fgru_ops(
            i0=i0,
            ff_drive=low_h,
            h2=high_h,
            layer_id=layer_id)

        # Peephole activities
        if self.use_homunculus:
            if self.gate_homunculus:
                pre_homunculus = getattr(self, 'pre_homunculus_%s' % layer_id)
                post_homunculus = getattr(
                    self, 'post_homunculus_%s' % layer_id)
                homunculus_bias = getattr(
                    self, 'homunculus_bias_%s' % layer_id)
                pre_activity = self.conv_op(
                    input=low_h,
                    filter=pre_homunculus,
                    strides=self.strides,
                    padding=self.ff_padding,
                    data_format=self.data_format,
                    name='pre_homunculus_act_%s' % layer_id)
                post_activity = self.conv_op(
                    input=fb_act,
                    filter=post_homunculus,
                    strides=self.strides,
                    padding=self.ff_padding,
                    data_format=self.data_format,
                    name='post_homunculus_act_%s' % layer_id)
                homunculus = tf.sigmoid(
                    pre_activity + post_activity + homunculus_bias)
            else:
                homunculus = getattr(self, 'homunculus_%s' % layer_id)
                if self.time_homunculus:
                    homunculus = tf.sigmoid(homunculus[i0])
                else:
                    homunculus = tf.sigmoid(homunculus)
            fb_act = (homunculus * fb_act) + ((1 - homunculus) * low_h)
        else:
            # Skip low with inhibited low
            fb_act += low_h
        return fb_act, error

