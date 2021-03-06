import numpy as np
import tensorflow as tf
from layers.feedforward.conv import get_bilinear_filter


class CreateGNParams(object):
    """Methods for initializing gammanet parameters."""
    def __init__(self):
        pass

    def __call__(self):
        pass

    def get_input_gate_shape(
            self,
            compression,
            fan_in,
            fan_out,
            layer,
            spatial_output=1):
        """Determine the shape of gates given model options.

        Adjust for the following conditions:
        (1) Top-down gate (concat top-down activity with layer's recurrent)
        (2) Squeeze-fb, in which the top-down dimensionality is preserved
        (3) Attention
            (i) Squeeze and excite
            (ii) Gala

        """
        s_shape = None
        if self.td_gate and compression is 'upsample':
            fan_in *= 2
        g_shape = [
            self.gate_filter,
            self.gate_filter,
            fan_in * 2,
            fan_out]
        if (
                self.attention is 'se' or
                self.attention is 'gala' and
                self.attention_layers > 1):
            # Create global attention
            g_shape[-1] //= 2
            depth_g_shape = [g_shape]
            # Downsample
            prev_fan_out = g_shape[-1]
            for idx in range(1, self.attention_layers // 2):
                it_fan_out = prev_fan_out // 2
                depth_g_shape += [[
                    self.gate_filter,
                    self.gate_filter,
                    prev_fan_out,
                    it_fan_out]]
                prev_fan_out = it_fan_out

            # Upsample
            for idx in range(
                    self.attention_layers // 2, self.attention_layers):
                it_fan_out = prev_fan_out * 2
                depth_g_shape += [[
                    self.gate_filter,
                    self.gate_filter,
                    prev_fan_out,
                    it_fan_out]]
                prev_fan_out = it_fan_out
            g_shape = depth_g_shape
            assert it_fan_out == fan_out,\
                'Global attention fan_out is incorrect.'
        else:
            g_shape = [g_shape]

        if self.attention is 'gala':
            # Create spatial attention
            if self.attention_layers == 1:
                s_shape = [[
                    self.saliency_filter,
                    self.saliency_filter,
                    fan_in,
                    spatial_output]]
            else:
                prev_fan_out = fan_out // 2
                depth_s_shape = [[
                    self.saliency_filter,
                    self.saliency_filter,
                    fan_in,
                    prev_fan_out]]

                # Downsample
                for idx in range(1, self.attention_layers - 1):
                    it_fan_out = prev_fan_out // 2
                    depth_s_shape += [[
                        self.saliency_filter,
                        self.saliency_filter,
                        prev_fan_out,
                        it_fan_out]]
                    prev_fan_out = it_fan_out

                # Force saliency
                depth_s_shape += [[
                    self.saliency_filter,
                    self.saliency_filter,
                    prev_fan_out,
                    spatial_output]]
                s_shape = depth_s_shape
        return g_shape, s_shape

    def symmetric_init(self, w, symmetry=None):
        """Initialize symmetric weight sharing."""
        if symmetry is None:
            symmetry = self.symmetric_weights
        init = w
        if ('channel' in symmetry) and ('space' in symmetry):
            init = tf.cast(
                0.5 * (init + init[::-1, ::-1]), self.dtype)
            init = tf.cast(
                0.5 * (init + tf.transpose(
                    init, (0, 1, 3, 2))), self.dtype)
        elif 'channel' in symmetry:
            init = tf.cast(
                0.5 * (init + tf.transpose(init, (0, 1, 3, 2))), self.dtype)
        elif 'space' in symmetry:
            init = tf.cast(
                0.5 * (init + init[::-1, ::-1]), self.dtype)
        return init

    def create_conv_weights(
            self,
            reps,
            layer_id,
            fan_in,
            fan_out,
            kernels,
            prev_layer=None,
            bias_gate_init=1.):  # 0.):
        """Create convolutional filters and biases."""
        if not kernels[0]:
            return
        if reps == 4:
            proj_fan_in = fan_in
            fan_in = fan_out
        for il in range(reps):
            if self.combine_fgru_output and il == 0 and prev_layer['fgru_kernels'][0]:
                store_fan_in = fan_in
                fan_in *= 2
            elif self.combine_fgru_output and prev_layer['fgru_kernels'][0]:
                fan_in = store_fan_in
            if self.recurrent_ff:
                # Create minimal input/output gate + recurrent weight
                weight = tf.get_variable(
                    name='ffdrive_gate_kernel_prev_%s_%s' % (
                        layer_id, il),
                    dtype=self.dtype,
                    shape=[1, 1, fan_out, fan_out],
                    initializer=self.gate_initializer,
                    trainable=self.train_ff_gate)
                self.setattr(
                    self,
                    'ffdrive_gate_kernel_prev_%s_%s' % (layer_id, il),
                    weight)
                weight = tf.get_variable(
                    name='ffdrive_gate_kernel_cand_%s_%s' % (
                        layer_id, il),
                    dtype=self.dtype,
                    shape=[1, 1, fan_out, fan_out],
                    initializer=self.gate_initializer,
                    trainable=self.train_ff_gate)
                self.setattr(
                    self,
                    'ffdrive_gate_kernel_cand_%s_%s' % (layer_id, il),
                    weight)
                self.setattr(  # zeros or 1s?
                    self,
                    'ffdrive_gate_bias_%s_%s' % (layer_id, il),
                    tf.get_variable(
                        name='ffdrive_gate_bias_%s_%s' % (
                            layer_id, il),
                        dtype=self.dtype,
                        shape=[fan_out],
                        initializer=tf.initializers.constant(bias_gate_init),
                        trainable=self.train_ff_gate))
                weight = tf.get_variable(
                    name='ffdrive_kernel_hidden_%s_%s' % (
                        layer_id, il),
                    dtype=self.dtype,
                    shape=[1, 1, fan_out, fan_out],
                    initializer=self.kernel_initializer,
                    trainable=self.train)
                self.setattr(
                    self,
                    'ffdrive_kernel_hidden_%s_%s' % (layer_id, il),
                    weight)

                # Normalization parameters
                # self.setattr(
                #     self,
                #     'ffdrive_hidden_bn_beta_%s_%s' % (layer_id, il),
                #     tf.get_variable(
                #         name='ffdrive_hidden_bn_beta_%s_%s' % (
                #             layer_id, il),
                #         dtype=self.dtype,
                #         shape=[1, 1, 1, fan_out],
                #         initializer=tf.initializers.zeros(),
                #         trainable=self.train))
                # self.setattr(
                #     self,
                #     'ffdrive_hidden_bn_gamma_%s_%s' % (layer_id, il),
                #     tf.get_variable(
                #         name='ffdrive_hidden_bn_gamma_%s_%s' % (
                #             layer_id, il),
                #         dtype=self.dtype,
                #         shape=[1, 1, 1, fan_out],
                #         initializer=tf.initializers.ones(),
                #         trainable=self.train))
                self.setattr(
                    self,
                    'ffdrive_gate_prev_bn_beta_%s_%s' % (layer_id, il),
                    tf.get_variable(
                        name='ffdrive_gate_prev_bn_beta_%s_%s' % (
                            layer_id, il),
                        dtype=self.dtype,
                        shape=[1, 1, 1, fan_out],
                        initializer=tf.initializers.zeros(),
                        trainable=self.train_norm_params))
                self.setattr(
                    self,
                    'ffdrive_gate_prev_bn_gamma_%s_%s' % (layer_id, il),
                    tf.get_variable(
                        name='ffdrive_gate_prev_bn_gamma_%s_%s' % (
                            layer_id, il),
                        dtype=self.dtype,
                        shape=[1, 1, 1, fan_out],
                        initializer=self.gamma_init,
                        trainable=self.train_norm_params))
                self.setattr(
                    self,
                    'ffdrive_gate_cand_bn_beta_%s_%s' % (layer_id, il),
                    tf.get_variable(
                        name='ffdrive_gate_cand_bn_beta_%s_%s' % (
                            layer_id, il),
                        dtype=self.dtype,
                        shape=[1, 1, 1, fan_out],
                        initializer=tf.initializers.zeros(),
                        trainable=self.train_norm_params))
                self.setattr(
                    self,
                    'ffdrive_gate_cand_bn_gamma_%s_%s' % (layer_id, il),
                    tf.get_variable(
                        name='ffdrive_gate_cand_bn_gamma_%s_%s' % (
                            layer_id, il),
                        dtype=self.dtype,
                        shape=[1, 1, 1, fan_out],
                        initializer=self.gamma_init,
                        trainable=self.train_norm_params))

            # Create kernel
            if self.separable_convs:
                self.setattr(
                    self,
                    'ffdrive_kernel_spatial_%s_%s' % (layer_id, il),
                    tf.get_variable(
                        name='ffdrive_kernel_spatial_%s_%s' % (
                            layer_id, il),
                        dtype=self.dtype,
                        shape=kernels + [
                            fan_in, self.separable_convs],
                        initializer=self.kernel_initializer,
                        trainable=self.train))
                self.setattr(
                    self,
                    'ffdrive_kernel_channel_%s_%s' % (layer_id, il),
                    tf.get_variable(
                        name='ffdrive_kernel_channel_%s_%s' % (
                            layer_id, il),
                        dtype=self.dtype,
                        shape=[
                            1,
                            1,
                            fan_in * self.separable_convs,
                            fan_out],
                        initializer=self.kernel_initializer,
                        trainable=self.train))
            else:
                self.setattr(
                    self,
                    'ffdrive_kernel_%s_%s' % (layer_id, il),
                    tf.get_variable(
                        name='ffdrive_kernel_%s_%s' % (
                            layer_id, il),
                        dtype=self.dtype,
                        shape=kernels + [fan_in, fan_out],
                        initializer=self.kernel_initializer,
                        trainable=self.train))
            self.setattr(
                self,
                'ffdrive_bias_%s_%s' % (layer_id, il),
                tf.get_variable(
                    name='ffdrive_bias_%s_%s' % (
                        layer_id, il),
                    dtype=self.dtype,
                    shape=[fan_out],
                    initializer=tf.initializers.ones(),
                    trainable=self.train))

            # Normalization parameters
            self.setattr(
                self,
                'ffdrive_bn_beta_%s_%s' % (layer_id, il),
                tf.get_variable(
                    name='ffdrive_bn_beta_%s_%s' % (
                        layer_id, il),
                    dtype=self.dtype,
                    shape=[1, 1, 1, fan_out],
                    initializer=tf.initializers.zeros(),
                    trainable=self.train_norm_params))
            self.setattr(
                self,
                'ffdrive_bn_gamma_%s_%s' % (layer_id, il),
                tf.get_variable(
                    name='ffdrive_bn_gamma_%s_%s' % (
                        layer_id, il),
                    dtype=self.dtype,
                    shape=[1, 1, 1, fan_out],
                    initializer=self.gamma_init,
                    trainable=self.train_norm_params))
            if il == 0 and reps == 4:
                # Initial resnet-style shortcut and BN
                self.setattr(
                    self,
                    'ffdrive_projection_kernel_%s_%s' % (layer_id, il),
                    tf.get_variable(
                        name='ffdrive_projection_kernel_%s_%s' % (
                            layer_id, il),
                        dtype=self.dtype,
                        shape=[1, 1, proj_fan_in, fan_out],
                        initializer=self.kernel_initializer,
                        trainable=self.train))
                self.setattr(
                    self,
                    'ffdrive_branch_bn_beta_%s_%s' % (layer_id, il),
                    tf.get_variable(
                        name='ffdrive_branch_bn_beta_%s_%s' % (
                            layer_id, il),
                        dtype=self.dtype,
                        shape=[1, 1, 1, fan_out],
                        initializer=tf.initializers.zeros(),
                        trainable=self.train_norm_params))
                self.setattr(
                    self,
                    'ffdrive_branch_bn_gamma_%s_%s' % (layer_id, il),
                    tf.get_variable(
                        name='ffdrive_branch_bn_gamma_%s_%s' % (
                            layer_id, il),
                        dtype=self.dtype,
                        shape=[1, 1, 1, fan_out],
                        initializer=self.gamma_init,
                        trainable=self.train_norm_params))
            fan_in = fan_out

    def create_resize_weights(
            self,
            reps,
            layer_id,
            fan_in,
            fan_out,
            kernels,
            layer=None,
            prev_layer=None,
            bias_gate_init=1.,  # 0.,
            bias_init=tf.initializers.zeros):
        """Create kernels and biases for resizing feature maps.

        Flags for bilinear initialization of kernels,
        using separable upsamples (fixing resize to bilinear).
        """
        # if self.separable_upsample:
        #     shape = [1, 1] + [fan_in, fan_out]
        # else:
        #     shape = kernels + [fan_in, fan_out]

        # Create resize kernels
        if reps == 4:
            proj_fan_in = fan_in
            fan_in = fan_out
        if self.combine_fgru_output and prev_layer['fgru_kernels'][0]:
            store_fan_in = fan_in
            fan_in *= 2
        if not self.separable_upsample:
            assert reps and kernels[0], 'Pass a repeat + kernel size.'
            if self.bilinear_init and not self.separable_upsample:
                kernel = get_bilinear_filter(
                    name='resize_kernel_%s' % layer_id,
                    filter_shape=kernels + [fan_out, fan_in],
                    upscale_factor=self.pool_strides[1],
                    dtype=self.dtype,
                    trainable=self.train)
                self.setattr(
                    self,
                    'resize_kernel_%s' % layer_id,
                    kernel)
            else:
                initializer = self.kernel_initializer
                self.setattr(
                    self,
                    'resize_kernel_%s' % layer_id,
                    tf.get_variable(
                        name='resize_kernel_%s' % layer_id,
                        dtype=self.dtype,
                        shape=kernels + [fan_out, fan_in],
                        initializer=initializer,
                        trainable=self.train))
            self.setattr(
                self,
                'resize_bias_%s' % layer_id,
                tf.get_variable(
                    name='resize_bias_%s' % layer_id,
                    dtype=self.dtype,
                    shape=[fan_out],
                    initializer=bias_init,
                    trainable=self.train))
        if self.combine_fgru_output and prev_layer['fgru_kernels'][0]:
            fan_in = store_fan_in

        # Post-upsample conv kernel
        for il in range(reps):
            # Create recurrent vars if requested
            if self.combine_fgru_output and il == 0 and isinstance(prev_layer['fgru_kernels'], list):
                store_fan_in = fan_in
                fan_in *= 2
            elif self.combine_fgru_output and isinstance(layer['fgru_kernels'], list):
                fan_in = store_fan_in
            if il == 0 and self.image_resize and self.separable_upsample:
                ch_in, ch_out = fan_in, fan_out
            else:
                ch_in, ch_out = fan_out, fan_out
            if self.recurrent_ff:
                # Create minimal input/output gate + recurrent weight
                weight = tf.get_variable(
                    name='conv_resize_gate_kernel_prev_%s_%s' % (
                        layer_id, il),
                    dtype=self.dtype,
                    shape=[1, 1, ch_out, ch_out],
                    initializer=self.gate_initializer,
                    trainable=self.train_ff_gate)
                self.setattr(
                    self,
                    'conv_resize_gate_kernel_prev_%s_%s' % (layer_id, il),
                    weight)
                weight = tf.get_variable(
                    name='conv_resize_gate_kernel_cand_%s_%s' % (
                        layer_id, il),
                    dtype=self.dtype,
                    shape=[1, 1, ch_out, ch_out],
                    initializer=self.gate_initializer,
                    trainable=self.train_ff_gate)
                self.setattr(
                    self,
                    'conv_resize_gate_kernel_cand_%s_%s' % (layer_id, il),
                    weight)
                weight = tf.get_variable(
                    name='conv_resize_kernel_hidden_%s_%s' % (
                        layer_id, il),
                    dtype=self.dtype,
                    shape=[1, 1, ch_out, ch_out],
                    initializer=self.kernel_initializer,
                    trainable=self.train)
                self.setattr(
                    self,
                    'conv_resize_kernel_hidden_%s_%s' % (layer_id, il),
                    weight)
                self.setattr(  # zeros or 1s?
                    self,
                    'conv_resize_gate_bias_%s_%s' % (layer_id, il),
                    tf.get_variable(
                        name='conv_resize_gate_bias_%s_%s' % (
                            layer_id, il),
                        dtype=self.dtype,
                        shape=[ch_out],
                        initializer=tf.initializers.constant(bias_gate_init),
                        trainable=self.train_ff_gate))

                # Normalization parameters
                # self.setattr(
                #     self,
                #     'conv_resize_hidden_bn_beta_%s_%s' % (layer_id, il),
                #     tf.get_variable(
                #         name='conv_resize_hidden_bn_beta_%s_%s' % (
                #             layer_id, il),
                #         dtype=self.dtype,
                #         shape=[1, 1, 1, fan_out],
                #         initializer=tf.initializers.zeros(),
                #         trainable=self.train))
                # self.setattr(
                #     self,
                #     'conv_resize_hidden_bn_gamma_%s_%s' % (layer_id, il),
                #     tf.get_variable(
                #         name='conv_resize_hidden_bn_gamma_%s_%s' % (
                #             layer_id, il),
                #         dtype=self.dtype,
                #         shape=[1, 1, 1, fan_out],
                #         initializer=tf.initializers.ones(),
                #         trainable=self.train))
                self.setattr(
                    self,
                    'conv_resize_gate_prev_bn_beta_%s_%s' % (layer_id, il),
                    tf.get_variable(
                        name='conv_resize_gate_prev_bn_beta_%s_%s' % (
                            layer_id, il),
                        dtype=self.dtype,
                        shape=[1, 1, 1, fan_out],
                        initializer=tf.initializers.zeros(),
                        trainable=self.train_norm_params))
                self.setattr(
                    self,
                    'conv_resize_gate_prev_bn_gamma_%s_%s' % (layer_id, il),
                    tf.get_variable(
                        name='conv_resize_gate_prev_bn_gamma_%s_%s' % (
                            layer_id, il),
                        dtype=self.dtype,
                        shape=[1, 1, 1, fan_out],
                        initializer=self.gamma_init,
                        trainable=self.train_norm_params))
                self.setattr(
                    self,
                    'conv_resize_gate_cand_bn_beta_%s_%s' % (layer_id, il),
                    tf.get_variable(
                        name='conv_resize_gate_cand_bn_beta_%s_%s' % (
                            layer_id, il),
                        dtype=self.dtype,
                        shape=[1, 1, 1, fan_out],
                        initializer=tf.initializers.zeros(),
                        trainable=self.train_norm_params))
                self.setattr(
                    self,
                    'conv_resize_gate_cand_bn_gamma_%s_%s' % (layer_id, il),
                    tf.get_variable(
                        name='conv_resize_gate_cand_bn_gamma_%s_%s' % (
                            layer_id, il),
                        dtype=self.dtype,
                        shape=[1, 1, 1, fan_out],
                        initializer=self.gamma_init,
                        trainable=self.train_norm_params))

            if self.upsample_convs and self.separable_convs:
                self.setattr(  # Add additional spatial res
                    self,
                    'conv_resize_kernel_spatial_%s_%s' % (layer_id, il),
                    tf.get_variable(
                        name='conv_resize_kernel_spatial_%s_%s' % (
                            layer_id, il),
                        dtype=self.dtype,
                        shape=[kernels[0], kernels[1]] + [
                            ch_in, self.separable_convs],
                        initializer=self.kernel_initializer,
                        trainable=self.train))
                self.setattr(
                    self,
                    'conv_resize_kernel_channel_%s_%s' % (layer_id, il),
                    tf.get_variable(
                        name='conv_resize_kernel_channel_%s_%s' % (
                            layer_id, il),
                        dtype=self.dtype,
                        shape=[
                            1,
                            1,
                            self.separable_convs * ch_in,
                            fan_out],
                        initializer=self.kernel_initializer,
                        trainable=self.train))
                self.setattr(
                    self,
                    'conv_resize_bias_%s_%s' % (layer_id, il),
                    tf.get_variable(
                        name='conv_resize_bias_%s_%s' % (layer_id, il),
                        dtype=self.dtype,
                        shape=ch_out,
                        initializer=bias_init,
                        trainable=self.train))
            elif self.upsample_convs and not self.separable_convs:
                self.setattr(
                    self,
                    'conv_resize_kernel_%s_%s' % (layer_id, il),
                    tf.get_variable(
                        name='conv_resize_kernel_%s_%s' % (layer_id, il),
                        dtype=self.dtype,
                        shape=[
                            self.up_kernel[0] + 1,
                            self.up_kernel[1] + 1] + [ch_in, ch_out],
                        initializer=self.kernel_initializer,
                        trainable=self.train))
                self.setattr(
                    self,
                    'conv_resize_bias_%s_%s' % (layer_id, il),
                    tf.get_variable(
                        name='conv_resize_bias_%s_%s' % (layer_id, il),
                        dtype=self.dtype,
                        shape=ch_out,
                        initializer=bias_init,
                        trainable=self.train))
            # Normalization parameters
            self.setattr(
                self,
                'conv_resize_bn_beta_%s_%s' % (layer_id, il),
                tf.get_variable(
                    name='conv_resize_bn_beta_%s_%s' % (
                        layer_id, il),
                    dtype=self.dtype,
                    shape=[1, 1, 1, fan_out],
                    initializer=tf.initializers.zeros(),
                    trainable=self.train_norm_params))
            self.setattr(
                self,
                'conv_resize_bn_gamma_%s_%s' % (layer_id, il),
                tf.get_variable(
                    name='conv_resize_bn_gamma_%s_%s' % (
                        layer_id, il),
                    dtype=self.dtype,
                    shape=[1, 1, 1, fan_out],
                    initializer=self.gamma_init,
                    trainable=self.train_norm_params))
            if il == 0 and reps == 4:
                # Initial resnet-style shortcut and BN
                self.setattr(
                    self,
                    'conv_resize_projection_kernel_%s_%s' % (layer_id, il),
                    tf.get_variable(
                        name='conv_resize_projection_kernel_%s_%s' % (
                            layer_id, il),
                        dtype=self.dtype,
                        shape=[1, 1, proj_fan_in, fan_out],
                        initializer=self.kernel_initializer,
                        trainable=self.train))
                self.setattr(
                    self,
                    'conv_resize_branch_bn_beta_%s_%s' % (layer_id, il),
                    tf.get_variable(
                        name='conv_resize_branch_bn_beta_%s_%s' % (
                            layer_id, il),
                        dtype=self.dtype,
                        shape=[1, 1, 1, fan_out],
                        initializer=tf.initializers.zeros(),
                        trainable=self.train_norm_params))
                self.setattr(
                    self,
                    'conv_resize_branch_bn_gamma_%s_%s' % (layer_id, il),
                    tf.get_variable(
                        name='conv_resize_branch_bn_gamma_%s_%s' % (
                            layer_id, il),
                        dtype=self.dtype,
                        shape=[1, 1, 1, fan_out],
                        initializer=self.gamma_init,
                        trainable=self.train_norm_params))

    def create_homunculus(self, layer_id, fgru_kernel, fan_in, fan_out):
        """Create router homunculus if requested."""
        if not self.use_homunculus or not fgru_kernel[0]:
            return

        if self.force_horizontal:
            # homu_train = False
            homu_init = np.inf
        elif self.gate_homunculus:
            homu_init = self.gate_initializer
        else:
            # homu_train = True
            if self.time_homunculus:
                homu_init = tf.zeros(self.timesteps, dtype=self.dtype)
            elif self.force_horizontal:
                homu_init = tf.constant(
                    (np.zeros([]) * np.inf).astype(
                        np.float32),
                    dtype=self.dtype)
            elif self.single_homunculus:
                homu_init = tf.constant(
                    (np.zeros([])).astype(
                        np.float32),
                    dtype=self.dtype)
            else:
                homu_init = tf.constant(
                    (np.zeros(fan_out)).astype(
                        np.float32),
                    dtype=self.dtype)
        if self.gate_homunculus:
            self.setattr(
                self,
                'pre_homunculus_%s' % layer_id,
                tf.get_variable(
                    name='pre_homunculus_%s' % layer_id,
                    dtype=self.dtype,
                    shape=[1, 1, fan_out, fan_out],
                    initializer=homu_init,
                    trainable=self.train_fgru_gate))
            self.setattr(
                self,
                'post_homunculus_%s' % layer_id,
                tf.get_variable(
                    name='post_homunculus_%s' % layer_id,
                    dtype=self.dtype,
                    shape=[1, 1, fan_out, fan_out],
                    initializer=homu_init,
                    trainable=self.train_fgru_gate))
            self.setattr(
                self,
                'homunculus_bias_%s' % layer_id,
                tf.get_variable(
                    name='homunculus_bias_%s' % layer_id,
                    dtype=self.dtype,
                    shape=[1, 1, 1, fan_out],
                    initializer=tf.initializers.zeros(),
                    trainable=self.train_fgru_gate))
        else:
            self.setattr(
                self,
                'homunculus_%s' % layer_id,
                tf.get_variable(
                    name='homunculus_%s' % layer_id,
                    dtype=self.dtype,
                    initializer=homu_init,
                    trainable=self.train_fgru_gate))

    def create_fgru_weights(
            self,
            fan_in,
            fan_out,
            kernels,
            layer_id,
            compression):
        """Create fgru kernels and biases."""
        if not kernels[0]:
            return
        self.symm_k_tag = 'symm' if self.symmetric_weights else 'full'
        self.symm_g_tag = 'symm' if self.symmetric_gate_weights else 'full'
        g_shape, s_shape = self.get_input_gate_shape(
            fan_in=fan_in,
            fan_out=fan_out,
            compression=compression,
            layer=layer_id)
        m_shape = [
            self.gate_filter,
            self.gate_filter,
            fan_in * 2,
            fan_out]  # reverse features for output
        if self.data_format is 'NHWC':
            self.bias_shape = [1, 1, 1, fan_out]
        else:
            self.bias_shape = [1, fan_out, 1, 1]
        with tf.variable_scope(
                'fgru_weights_%s' % layer_id):
            if self.weight_norm:
                V = self.horizontal_kernel_initializer(  # weight_norm_initializer(
                    kernels + [fan_in, fan_in])
                g = self.horizontal_kernel_initializer(  # weight_norm_initializer(
                    [1, 1, 1, fan_in])
                self.setattr(
                    self,
                    '%s_horizontal_kernels_inh_V_%s' % (
                        self.symm_k_tag,
                        layer_id),
                    tf.get_variable(
                        name='%s_horizontal_kernels_inh_V_%s' % (
                            self.symm_k_tag, layer_id),
                        dtype=self.dtype,
                        initializer=V,
                        trainable=self.train_fgru_kernels))
                self.setattr(
                    self,
                    '%s_horizontal_kernels_inh_g_%s' % (
                        self.symm_k_tag,
                        layer_id),
                    tf.get_variable(
                        name='%s_horizontal_kernels_inh_g_%s' % (
                            self.symm_k_tag, layer_id),
                        dtype=self.dtype,
                        initializer=g,
                        trainable=self.train_fgru_kernels))
                self.setattr(
                    self,
                    '%s_horizontal_kernels_exc_V_%s' % (
                        self.symm_k_tag, layer_id),
                    tf.get_variable(
                        name='%s_horizontal_kernels_exc_V_%s' % (
                             self.symm_k_tag, layer_id),
                        dtype=self.dtype,
                        initializer=V,
                        trainable=self.train_fgru_kernels))
                self.setattr(
                    self,
                    '%s_horizontal_kernels_exc_g_%s' % (
                        self.symm_k_tag, layer_id),
                    tf.get_variable(
                        name='%s_horizontal_kernels_exc_g_%s' % (
                             self.symm_k_tag, layer_id),
                        dtype=self.dtype,
                        initializer=g,
                        trainable=self.train_fgru_kernels))
            else:
                iv = self.horizontal_kernel_initializer(
                    kernels + [fan_in, fan_in])
                ev = self.horizontal_kernel_initializer(
                    kernels + [fan_in, fan_in])
                if self.symmetric_weights and self.symmetric_inits:
                    iv = self.symmetric_init(iv)
                    ev = self.symmetric_init(ev)
                self.setattr(
                    self,
                    '%s_horizontal_kernels_inh_%s' % (
                        self.symm_k_tag,
                        layer_id),
                    tf.get_variable(
                        name='%s_horizontal_kernels_inh_%s' % (
                            self.symm_k_tag, layer_id),
                        dtype=self.dtype,
                        initializer=iv,
                        trainable=self.train_fgru_kernels))
                self.setattr(
                    self,
                    '%s_horizontal_kernels_exc_%s' % (
                        self.symm_k_tag, layer_id),
                    tf.get_variable(
                        name='%s_horizontal_kernels_exc_%s' % (
                             self.symm_k_tag, layer_id),
                        dtype=self.dtype,
                        initializer=ev,
                        trainable=self.train_fgru_kernels))

            # Create the 4 sets of normalization params
            if self.fgru_batchnorm:
                if self.c1_c2_norm:
                    self.setattr(
                        self,
                        'c1_bn_beta_%s' % layer_id,
                        tf.get_variable(
                            name='c1_bn_beta_%s' % layer_id,
                            dtype=self.dtype,
                            shape=[1, 1, 1, fan_out],
                            initializer=tf.initializers.zeros(),
                            trainable=self.train_norm_params))
                    self.setattr(
                        self,
                        'c1_bn_gamma_%s' % layer_id,
                        tf.get_variable(
                            name='c1_bn_gamma_%s' % layer_id,
                            dtype=self.dtype,
                            shape=[1, 1, 1, fan_out],
                            initializer=self.gamma_init,
                            trainable=self.train_norm_params))
                    self.setattr(
                        self,
                        'c2_bn_beta_%s' % layer_id,
                        tf.get_variable(
                            name='c2_bn_beta_%s' % layer_id,
                            dtype=self.dtype,
                            shape=[1, 1, 1, fan_out],
                            initializer=tf.initializers.zeros(),
                            trainable=self.train_norm_params))
                    self.setattr(
                        self,
                        'c2_bn_gamma_%s' % layer_id,
                        tf.get_variable(
                            name='c2_bn_gamma_%s' % layer_id,
                            dtype=self.dtype,
                            shape=[1, 1, 1, fan_out],
                            initializer=self.gamma_init,
                            trainable=self.train_norm_params))
                # self.setattr(
                #     self,
                #     'g1_bn_beta_%s' % layer_id,
                #     tf.get_variable(
                #         name='g1_bn_beta_%s' % layer_id,
                #         dtype=self.dtype,
                #         shape=[1, 1, 1, fan_out],
                #         initializer=tf.initializers.zeros(),
                #         trainable=self.train_norm_params))
                # self.setattr(
                #     self,
                #     'g1_bn_gamma_%s' % layer_id,
                #     tf.get_variable(
                #         name='g1_bn_gamma_%s' % layer_id,
                #         dtype=self.dtype,
                #         shape=[1, 1, 1, fan_out],
                #         initializer=self.gamma_init,
                #         trainable=self.train_norm_params))
                # self.setattr(
                #     self,
                #     'g2_bn_beta_%s' % layer_id,
                #     tf.get_variable(
                #         name='g2_bn_beta_%s' % layer_id,
                #         dtype=self.dtype,
                #         shape=[1, 1, 1, fan_out],
                #         initializer=tf.initializers.zeros(),
                #         trainable=self.train_norm_params))
                # self.setattr(
                #     self,
                #     'g2_bn_gamma_%s' % layer_id,
                #     tf.get_variable(
                #         name='g2_bn_gamma_%s' % layer_id,
                #         dtype=self.dtype,
                #         shape=[1, 1, 1, fan_out],
                #         initializer=self.gamma_init,
                #         trainable=self.train_norm_params))

            # if self.combine_fgru_output:
            #     self.setattr(
            #         self,
            #         '%s_combine_kernels_%s' % (self.symm_g_tag, layer_id),
            #         tf.get_variable(
            #             name='%s_combine_kernels_%s' % (
            #                 self.symm_g_tag, layer_id),
            #             dtype=self.dtype,
            #             shape=[1, 1, fan_out * 2, fan_out],
            #             initializer=self.kernel_initializer,
            #             trainable=self.train))
            #     self.setattr(
            #         self,
            #         'combine_bias_%s' % layer_id,
            #         tf.get_variable(
            #             name='combine_bias_%s' % layer_id,
            #             dtype=self.dtype,
            #             shape=[fan_out],
            #             initializer=tf.initializers.zeros(),
            #             trainable=self.train))

            if self.fgru_output_normalization:
                self.setattr(
                    self,
                    'h2_bn_beta_%s' % layer_id,
                    tf.get_variable(
                        name='h2_bn_beta_%s' % layer_id,
                        dtype=self.dtype,
                        shape=[1, 1, 1, fan_out],
                        initializer=tf.initializers.zeros(),
                        trainable=self.train_norm_params))
                self.setattr(
                    self,
                    'h2_bn_gamma_%s' % layer_id,
                    tf.get_variable(
                        name='h2_bn_gamma_%s' % layer_id,
                        dtype=self.dtype,
                        shape=[1, 1, 1, fan_out],
                        initializer=self.gamma_init,
                        trainable=self.train_norm_params))

            # Create input/gain-gate and add attention if requested
            if self.attention is 'se' or self.attention is 'gala':
                assert isinstance(g_shape, list), \
                    'Gain kernel shapes were incorrectly created.'
                # Prepare channel gain kernels from a list
                for g_idx, it_g_shape in enumerate(g_shape):
                    self.setattr(
                        self,
                        '%s_%s_%s_global_gain' % (
                            self.symm_g_tag, g_idx, layer_id),
                        tf.get_variable(
                            name='%s_%s_%s_global_gain' % (
                                self.symm_g_tag,
                                g_idx,
                                layer_id),
                            dtype=self.dtype,
                            trainable=self.train_fgru_gate,
                            shape=it_g_shape,
                            initializer=self.gate_initializer))
                    self.setattr(
                        self,
                        '%s_%s_%s_global_bias' % (
                            self.symm_g_tag, g_idx, layer_id),
                        tf.get_variable(
                            name='%s_%s_%s_global_bias' % (
                                self.symm_g_tag,
                                g_idx,
                                layer_id),
                            dtype=self.dtype,
                            trainable=self.train_fgru_gate,
                            shape=it_g_shape[-1],
                            initializer=tf.initializers.zeros))
                    if g_idx < len(g_shape) and self.norm_attention:
                        self.setattr(
                            self,
                            '%s_%s_global_beta' % (g_idx, layer_id),
                            tf.get_variable(
                                name='%s_%s_global_beta' % (g_idx, layer_id),
                                dtype=self.dtype,
                                shape=[1, 1, 1, it_g_shape[-1]],
                                initializer=tf.initializers.zeros(),
                                trainable=self.train_norm_params))
                        self.setattr(
                            self,
                            '%s_%s_global_gamma' % (g_idx, layer_id),
                            tf.get_variable(
                                name='%s_%s_global_gamma' % (g_idx, layer_id),
                                dtype=self.dtype,
                                shape=[1, 1, 1, it_g_shape[-1]],
                                initializer=self.gamma_init,
                                trainable=self.train_norm_params))
            else:
                self.setattr(
                    self,
                    '%s_gain_kernels_%s' % (
                        self.symm_g_tag, layer_id),
                    tf.get_variable(
                        name='%s_gain_kernels_%s' % (
                            self.symm_g_tag, layer_id),
                        dtype=self.dtype,
                        trainable=self.train_fgru_gate,
                        shape=g_shape[0],
                        initializer=self.gate_initializer))
            if self.attention is 'gala':
                assert isinstance(s_shape, list), \
                    'Spatial kernel shapes were incorrectly created.'
                # Prepare spatial gain kernels from a list
                for s_idx, it_s_shape in enumerate(s_shape):
                    self.setattr(
                        self,
                        '%s_%s_%s_spatial_gain' % (
                            self.symm_g_tag, s_idx, layer_id),
                        tf.get_variable(
                            name='%s_%s_%s_spatial_gain' % (
                                self.symm_g_tag,
                                s_idx,
                                layer_id),
                            dtype=self.dtype,
                            trainable=self.train_fgru_gate,
                            shape=it_s_shape,
                            initializer=self.gate_initializer))
                    self.setattr(
                        self,
                        '%s_%s_%s_spatial_bias' % (
                            self.symm_g_tag, s_idx, layer_id),
                        tf.get_variable(
                            name='%s_%s_%s_spatial_bias' % (
                                self.symm_g_tag,
                                s_idx,
                                layer_id),
                            dtype=self.dtype,
                            trainable=self.train_fgru_gate,
                            shape=it_s_shape[-1],
                            initializer=tf.initializers.zeros))
                    if s_idx < len(s_shape) and self.norm_attention:
                        self.setattr(
                            self,
                            '%s_%s_spatial_beta' % (s_idx, layer_id),
                            tf.get_variable(
                                name='%s_%s_spatial_beta' % (s_idx, layer_id),
                                dtype=self.dtype,
                                shape=[1, 1, 1, it_s_shape[-1]],
                                initializer=tf.initializers.zeros(),
                                trainable=self.train_norm_params))
                        self.setattr(
                            self,
                            '%s_%s_spatial_gamma' % (s_idx, layer_id),
                            tf.get_variable(
                                name='%s_%s_spatial_gamma' % (s_idx, layer_id),
                                dtype=self.dtype,
                                shape=[1, 1, 1, it_s_shape[-1]],
                                initializer=tf.initializers.ones(),
                                trainable=self.train_norm_params))

            # Create output/mix-gate
            self.setattr(
                self,
                '%s_mix_kernels_%s' % (self.symm_g_tag, layer_id),
                tf.get_variable(
                    name='%s_mix_kernels_%s' % (
                        self.symm_g_tag, layer_id),
                    dtype=self.dtype,
                    trainable=self.train_fgru_gate,
                    shape=m_shape,
                    initializer=self.gate_initializer))

            # Gain/mix bias
            if self.gate_bias_init == 'chronos':
                pre_bias_init = -tf.log(
                    tf.random_uniform(
                        shape=self.bias_shape,
                        dtype=self.dtype,
                        minval=1,
                        maxval=np.maximum(self.timesteps - 1, 1)))
                bias_init = tf.cast(pre_bias_init, self.dtype)
            else:
                bias_init = tf.ones(self.bias_shape, dtype=self.dtype)

            # If using attention, use normal biases + an additional chronos
            self.setattr(
                self,
                'gain_bias_%s' % layer_id,
                tf.get_variable(
                    name='gain_bias_%s' % layer_id,
                    dtype=self.dtype,
                    trainable=self.train_fgru_gate,
                    initializer=bias_init))
            if self.gate_bias_init == 'chronos':
                bias_init = tf.cast(-pre_bias_init, self.dtype)
            else:
                bias_init = tf.ones(self.bias_shape, dtype=self.dtype)
            self.setattr(
                self,
                'mix_bias_%s' % layer_id,
                tf.get_variable(
                    name='mix_bias_%s' % layer_id,
                    dtype=self.dtype,
                    trainable=self.train_fgru_gate,
                    initializer=bias_init))

            # Divisive params
            if self.alpha and not self.lesion_alpha:
                self.setattr(
                    self,
                    'alpha_%s' % layer_id,
                    tf.get_variable(
                        name='alpha_%s' % layer_id,
                        dtype=self.dtype,
                        shape=self.bias_shape,
                        trainable=self.train_fgru_params,
                        # initializer=self.alpha_initializer()))
                        initializer=self.alpha_initializer))
            elif self.lesion_alpha:
                self.setattr(
                    self,
                    'alpha_%s' % layer_id,
                    tf.constant(0., dtype=self.dtype))
            else:
                self.setattr(
                    self,
                    'alpha_%s' % layer_id,
                    tf.constant(1., dtype=self.dtype))

            if self.mu and not self.lesion_mu:
                self.setattr(
                    self,
                    'mu_%s' % layer_id,
                    tf.get_variable(
                        name='%s_mu' % self.layer_name,
                        dtype=self.dtype,
                        shape=self.bias_shape,
                        trainable=self.train_fgru_params,
                        # initializer=self.mu_initializer()))
                        initializer=self.mu_initializer))
            else:
                self.setattr(
                    self,
                    'mu_%s' % layer_id,
                    tf.constant(0., dtype=self.dtype))

            if self.multiplicative_excitation:
                if self.lesion_kappa:
                    self.setattr(
                        self,
                        'kappa_%s' % layer_id,
                        tf.constant(0., dtype=self.dtype))
                else:
                    self.setattr(
                        self,
                        'kappa_%s' % layer_id,
                        tf.get_variable(
                            name='kappa_%s' % layer_id,
                            dtype=self.dtype,
                            shape=self.bias_shape,
                            trainable=self.train_fgru_params,
                            # initializer=self.kappa_initializer())
                            initializer=self.kappa_initializer)
                    )
                if self.lesion_omega:
                    self.setattr(
                        self,
                        'omega_%s' % layer_id,
                        tf.constant(0., dtype=self.dtype))
                else:
                    self.setattr(
                        self,
                        'omega_%s' % layer_id,
                        tf.get_variable(
                            name='omega_%s' % layer_id,
                            dtype=self.dtype,
                            shape=self.bias_shape,
                            trainable=self.train_fgru_params,
                            # initializer=self.omega_initializer())
                            initializer=self.omega_initializer)
                    )
            else:
                self.setattr(
                    self,
                    'kappa_%s' % layer_id,
                    tf.constant(1., dtype=self.dtype))
                self.setattr(
                    self,
                    'omega_%s' % layer_id,
                    tf.constant(1., dtype=self.dtype))
            if self.adaptation is 'eta':
                self.setattr(
                    self,
                    'eta_%s' % layer_id,
                    tf.get_variable(
                        name='eta_%s' % self.layer_name,
                        dtype=self.dtype,
                        shape=self.bias_shape,
                        trainable=self.train_fgru_params,
                        initializer=self.kernel_initializer()))

    def create_hidden_states(
            self,
            constructor,
            shapes,
            recurrent_ff,
            init,
            dtype):
        """Create all requested hidden states."""
        def return_tensor(shape):
            """Return a tensor of shape with init."""
            if init == 'random':
                return tf.random_normal(shape, dtype=dtype)
            elif init == 'zeros' or init == 'identity':
                if np.any(shape == None):
                    shape = [shape[0], 320, 320, shape[-1]]
                return tf.zeros(shape, dtype=dtype)
            else:
                raise RuntimeError
        print('*' * 20)
        for idx, (layer_id, layer_info) in enumerate(constructor.items()):
        # for idx, (layer_id, layer_info) in enumerate(constructor.iteritems()):
            layer_shape = shapes[layer_id]
            if layer_info['ff_kernels'][0] and recurrent_ff:
                if layer_info['compression'] is not 'upsample':
                    tag = 'ffdrive'
                else:
                    tag = 'conv_resize'
                for rep in range(layer_info['ff_repeats']):
                    # Create recurrent FF states
                    self.setattr(
                        self,
                        '%s_%s_%s' % (tag, layer_id, rep),
                        return_tensor(
                            shape=layer_shape))
                    self.hidden_dict[
                        '%s_%s_%s' % (tag, layer_id, rep)] = layer_shape
            if (
                layer_info['fgru_kernels'][0] and
                    layer_info['compression'] != 'upsample'):
                # Create fGRU states
                setattr(
                    self,
                    'fgru_%s' % layer_id,
                    return_tensor(
                        shape=layer_shape))
                self.hidden_dict['fgru_%s' % layer_id] = layer_shape

                print(
                    'Layer %s, %s: %s' % (
                        layer_id,
                        layer_info['compression'],
                        layer_shape))
        print('*' * 20)


def defaults():
    """A dictionary containing defaults for auxilliary variables.
    These are adjusted by a passed aux dict variable."""
    return {
        'hidden_init': 'identity',
        'gate_bias_init': 'chronos',
        'recurrent_nl': tf.nn.relu,
        'gate_nl': tf.nn.sigmoid,
        'combine_fgru_output': False,
        'lesion_alpha': False,
        'lesion_mu': False,
        'lesion_omega': False,
        'lesion_kappa': False,
        'c1_c2_norm': True,
        'fgru_output_normalization': False,
        'alpha': True,  # divisive eCRF
        'mu': True,  # subtractive eCRF
        'timestep_output': False,  # Return list w/ all timesteps activity
        'time_skips': False,  # Skip time for convs: False/full/final
        'attention': 'se',  # 'gala', 'se', or False
        'upsample_convs': True,
        'bilinear_init': False,
        'symmetric_weights': True,
        'symmetric_gate_weights': False,
        'symmetric_inits': True,
        'force_alpha_divisive': False,
        'force_omega_nonnegative': False,
        'fgru_batchnorm': False,
        'excite_se': False,  # if not false, set to a reduction ratio
        'td_gate': False,  # Introduce TD activity into h1 gate
        'td_cell_state': False,
        'gate_filter': 1,  # Gate kernel size
        'saliency_filter': 5,
        'norm_attention': False,
        'attention_layers': 2,  # Use this depth in the attention layers
        'adaptation': False,  # 'norm',
        'use_homunculus': False,
        'time_homunculus': False,
        'gate_homunculus': True,
        'single_homunculus': False,
        'reuse_conv_bn': False,
        'while_loop': False,
        'multiplicative_excitation': True,
        'residual': True,
        'skip_connections': False,
        'force_bottom_up': False,
        'force_horizontal': False,
        'weight_norm': False,
        'gamma_init': tf.initializers.constant(0.1),
        'image_resize': tf.image.resize_nearest_neighbor,
        'partial_padding': False,
        'dilations': [1, 1, 1, 1]
    }
