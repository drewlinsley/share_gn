import warnings
import numpy as np
import tensorflow as tf
from collections import OrderedDict
from ops import tf_fun
from layers.recurrent.gn_params_minimal import CreateGNParams
from layers.recurrent.gn_params_minimal import defaults
from layers.recurrent.gn_recurrent_ops_alt_bn_sym_minimal import GNRnOps
from layers.recurrent.gn_feedforward_ops import GNFFOps


def update_layer_shape(
        layer_shape,
        pool_strides,
        data_format,
        padding,
        kernel,
        features,
        direction):
    """Return a new shape of an activity."""
    copy_layer = np.copy(layer_shape)

    def op(x, y):
        if direction is 'upsample':
            return x * y
        elif direction is 'pool' or direction is 'embedding':
            return x // y
        elif direction is None or direction is 'none':
            return x
        else:
            raise NotImplementedError(direction)
    valid_offset = 0
    if padding == 'VALID':
        valid_offset = kernel
    if np.any([x is None for x in copy_layer]):
        return copy_layer
    if data_format is 'NCHW':
        copy_layer[2] = op(
            copy_layer[2],
            pool_strides[0]) - valid_offset
        copy_layer[3] = op(
            copy_layer[3],
            pool_strides[0]) - valid_offset
        copy_layer[1] = features
    elif data_format is 'NHWC':
        copy_layer[1] = op(
            copy_layer[1],
            pool_strides[0]) - valid_offset
        copy_layer[2] = op(
            copy_layer[2],
            pool_strides[0]) - valid_offset
        copy_layer[3] = features
    else:
        raise NotImplementedError(data_format)
    return copy_layer


class GN(CreateGNParams, GNRnOps, GNFFOps):
    """Construct the gammanet."""
    def __getitem__(self, name):
        return getattr(self, name)

    def __contains__(self, name):
        return hasattr(self, name)

    def setattr(self, obj, key, value):
        """Wrapper for setattr that stores the variable name in a list."""
        setattr(obj, key, value)
        self.variable_list[key] = [value.get_shape().as_list(), value.dtype]

    def __init__(
            self,
            layer_name,
            gammanet_constructor,
            fgru_normalization_type,
            ff_normalization_type,
            train,
            reuse,
            fgru_connectivity,
            ff_nl=tf.nn.relu,
            additional_readouts=None,
            horizontal_kernel_initializer=tf.initializers.variance_scaling(),
            kernel_initializer=tf.initializers.variance_scaling(),
            gate_initializer=tf.contrib.layers.xavier_initializer(),
            train_ff_gate=None,
            train_fgru_gate=None,
            train_norm_moments=None,
            train_norm_params=None,
            train_fgru_kernels=None,
            train_fgru_params=None,
            up_kernel=None,
            stop_loop=False,
            recurrent_ff=False,
            timesteps=1,
            strides=[1, 1, 1, 1],
            pool_strides=[2, 2],
            pool_kernel=[4, 4],
            data_format='NHWC',
            horizontal_padding='SAME',
            ff_padding='SAME',
            aux=None):
        """Global initializations and settings."""
        self.timesteps = timesteps
        self.strides = strides
        self.pool_strides = pool_strides
        self.pool_kernel = pool_kernel
        self.horizontal_padding = horizontal_padding
        self.ff_padding = ff_padding
        self.train = train
        self.layer_name = layer_name
        self.data_format = data_format
        self.horizontal_kernel_initializer = horizontal_kernel_initializer
        self.kernel_initializer = kernel_initializer
        self.gate_initializer = gate_initializer
        self.additional_readouts = additional_readouts
        self.fgru_normalization_type = fgru_normalization_type
        self.ff_normalization_type = ff_normalization_type
        self.recurrent_ff = recurrent_ff
        self.stop_loop = stop_loop
        self.ff_nl = ff_nl
        self.fgru_connectivity = fgru_connectivity
        self.gammanet_constructor = gammanet_constructor
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
        if up_kernel is None:
            self.up_kernel = [h + w for h, w in zip(
                self.pool_strides, self.pool_kernel)]
            print('No up-kernel provided. Derived: %s' % self.up_kernel)
        else:
            self.up_kernel = up_kernel

        # Sort through and assign the auxilliary variables
        default_vars = defaults()
        if aux is not None and isinstance(aux, dict):
            for k, v in aux.iteritems():
                default_vars[k] = v
        self.update_params(default_vars)
        if self.time_skips:
            self.conv_dict = OrderedDict()

        # Store variables in the order they were created. Hack for python 2.x.
        self.variable_list = OrderedDict()
        self.hidden_dict = OrderedDict()

        # Kernel info
        if data_format is 'NHWC':
            self.prepared_pool_kernel = [1] + self.pool_kernel + [1]
            self.prepared_pool_stride = [1] + self.pool_strides + [1]
            self.up_strides = [1] + self.pool_strides + [1]
        else:
            self.prepared_pool_kernel = [1, 1] + self.pool_kernel
            self.prepared_pool_stride = [1, 1] + self.pool_strides
            self.up_strides = [1, 1] + self.pool_strides
        self.sanity_check()
        if self.symmetric_weights:
            self.symmetric_weights = self.symmetric_weights.split('_')

        # Nonlinearities and initializations
        if isinstance(self.recurrent_nl, basestring):
            self.recurrent_nl = tf_fun.interpret_nl(self.recurrent_nl)

        # Set initializers for greek letters
        if self.force_alpha_divisive:
            self.alpha_initializer = tf.initializers.variance_scaling
        else:
            self.alpha_initializer = tf.ones_initializer
        self.mu_initializer = tf.zeros_initializer
        # self.omega_initializer = tf.initializers.variance_scaling
        self.omega_initializer = tf.ones_initializer
        self.kappa_initializer = tf.zeros_initializer

        # Handle BN scope reuse
        self.scope_reuse = reuse

    def sanity_check(self):
        """Check that inputs are correct."""
        assert self.data_format == 'NHWC' or self.data_format == 'NCHW', \
            'Data format must be entered as NHWC or NCHW'
        if self.force_bottom_up:
            print('Forcing a bottom up version of the model. No recurrence!')
            self.timesteps = 1
            self.skip = False
        if self.force_horizontal:
            assert not self.while_loop, \
                'Forcing horizontal is incompatible with while loop.'
        assert not (self.timestep_output and self.while_loop), \
            'Cannot return per timestep output with the while-loop.'
        assert (
            self.attention is 'se' or
            self.attention is 'gala' or
            not self.attention)
        assert self.symmetric_weights is not True,\
            'Symmetric weights can be spatial channel spatial_channel or False'

    def update_params(self, kwargs):
        """Update the class attributes with kwargs."""
        if kwargs is not None:
            try:
                for k, v in kwargs.iteritems():
                    assert not hasattr(self, k), 'Overwriting attribute.'
                    setattr(self, k, v)
            except:
                for k, v in kwargs.items():
                    assert not hasattr(self, k), 'Overwriting attribute.'
                    setattr(self, k, v)

    def prepare_tensors(self, X_shape, allow_resize=True):
        """ Prepare recurrent/forward weight matrices from the constructor."""
        fan_in = X_shape[1] if self.data_format is 'NCHW' else X_shape[-1]
        self.layer_shapes = OrderedDict()
        layer_shape = X_shape
        prev_layer = None
        for idx, (layer_id, layer_info) in enumerate(
                self.gammanet_constructor.items()):
                # self.gammanet_constructor.iteritems()):
            # Update layer shape for layer
            layer_shape = update_layer_shape(
                layer_shape=layer_shape,
                pool_strides=self.pool_strides,
                data_format=self.data_format,
                features=layer_info['features'],
                direction=layer_info['compression'] if idx > 0 else None,
                padding=self.ff_padding,
                kernel=layer_info['ff_kernels'][0])
            self.layer_shapes[layer_id] = layer_shape

            if layer_info['compression'] is 'upsample' and allow_resize:
                # Add resizes and convs
                if self.fgru_connectivity is 'all_to_all':
                    encoder_layer = len(self.gammanet_constructor) - layer_id
                    layers_to_emb = range(
                        len(self.gammanet_constructor) // 2,
                        encoder_layer,
                        -1)
                    resize_features = [
                        self.gammanet_constructor[x]['features']
                        for x in layers_to_emb]
                    fan_in = int(fan_in + np.sum(
                        resize_features))
                self.create_resize_weights(
                    layer_id=layer_id,
                    fan_in=fan_in,
                    fan_out=layer_info['features'],
                    reps=layer_info['ff_repeats'],
                    layer=layer_info,
                    prev_layer=prev_layer,
                    kernels=layer_info['ff_kernels'])
                self.create_homunculus(
                    layer_id,
                    layer_info['fgru_kernels'],
                    fan_in=fan_in,
                    fan_out=layer_info['features'])
            else:
                # Add convs
                self.create_conv_weights(
                    layer_id=layer_id,
                    fan_in=fan_in,
                    fan_out=layer_info['features'],
                    reps=layer_info['ff_repeats'],
                    prev_layer=prev_layer,
                    kernels=layer_info['ff_kernels'])

            # Add fgrus
            self.create_fgru_weights(
                layer_id=layer_id,
                fan_in=layer_info['features'],
                fan_out=layer_info['features'],
                kernels=layer_info['fgru_kernels'],
                compression=layer_info['compression'])
            fan_in = layer_info['features']
            prev_layer = layer_info

    def bias_add(self, activity, bias, name=None):
        """Wrapper for adding a bias."""
        return tf.nn.bias_add(
            activity,
            bias,
            data_format=self.data_format,
            name=name)

    def interpret_symmetry(self, symmetric_weights):
        """Return a grad op for symmetric weights."""
        if not symmetric_weights:
            return False
        elif 'channel' in symmetric_weights and 'spatial' in symmetric_weights:
            return 'ChannelSymmetricConv'
        elif 'channel' in symmetric_weights:
            return 'ChannelSymmetricConv'
        elif 'spatial' in symmetric_weights:
            return 'SpatialSymmetricConv'
        else:
            raise NotImplementedError(symmetric_weights)

    def build_model(self, i0, activity, constructor):
        """Wrapper for building model in the constructor with input X."""
        for layer_id, layer_info in constructor.iteritems():
            if layer_info['compression'] is 'upsample':
                # Find corresponding encoder layer
                encoder_layer = len(constructor) - 1 - layer_id
                if layer_info['ff_kernels'][0]:
                    if self.fgru_connectivity is 'all_to_all':
                        emb = len(self.gammanet_constructor) // 2
                        layers_to_emb = range(
                            emb,
                            encoder_layer + 1,
                            -1)
                        extra_activities = [
                            getattr(self, 'fgru_%s' % x)
                            for x in layers_to_emb]
                        activity = [activity] + extra_activities
                    activity = self.upsample_router(
                        activity=activity,
                        layer_id=layer_id,
                        encoder_layer=encoder_layer,
                        reps=layer_info['ff_repeats'],
                        i0=i0)

                # Apply TD-fGRU if requested
                if layer_info['fgru_kernels'][0]:
                    fgru_activity = getattr(self, 'fgru_%s' % encoder_layer)
                    fgru_activity, error = self.td_router(
                        high_h=activity,
                        low_h=fgru_activity,
                        layer_id=layer_id,
                        i0=i0)
                    setattr(
                        self,
                        'fgru_%s' % encoder_layer,
                        fgru_activity)
                    activity = tf.identity(fgru_activity)
                    activity = self.fgru_postprocess(
                        activity=activity,
                        error=error,
                        layer_id=layer_id)
            else:
                if layer_info['ff_kernels'][0]:
                    # Always apply fGRUs to FF connections
                    activity = self.conv_tower(
                        activity=activity,
                        reps=layer_info['ff_repeats'],
                        compression=layer_info['compression'],
                        layer_id=layer_id,
                        i0=i0)

                # Apply H-fGRU if requested
                if layer_info['fgru_kernels'][0]:
                    fgru_label = 'fgru_%s' % layer_id
                    fgru_activity = getattr(self, fgru_label)
                    if self.hidden_init == 'identity' and i0 == 0:
                        # Set hidden to activity
                        fgru_activity = activity
                    error, fgru_activity = self.fgru_ops(  # h^(1), h^(2)
                        ff_drive=activity,
                        h2=fgru_activity,
                        layer_id=layer_id,
                        i0=i0)
                    setattr(
                        self,
                        fgru_label,
                        fgru_activity)
                    activity = tf.identity(fgru_activity) + activity
                    activity = self.fgru_postprocess(
                        activity=activity,
                        error=error,
                        layer_id=layer_id)

                # Pool if requested
                if layer_info['compression'] is 'pool':
                    activity = self.apply_pool(
                        activity=activity,
                        layer_id=layer_id)

                # Stop loop if requested
                if (
                    self.stop_loop and
                    fgru_label == self.stop_loop and
                        (i0 == self.timesteps - 1)):
                    return activity

        # Iterate timestep
        i0 += 1
        return activity  # , i0

    def __call__(self, X):
        """Run the backprop version of the Circuit."""
        X_shape = X.get_shape().as_list()
        self.N = X_shape[0]
        self.dtype = X.dtype
        self.prepare_tensors(X_shape)
        self.create_hidden_states(
            constructor=self.gammanet_constructor,
            shapes=self.layer_shapes,
            recurrent_ff=self.recurrent_ff,
            init=self.hidden_init,
            dtype=self.dtype)

        # For loop
        if self.timestep_output:
            self.timestep_list = [X]
        if self.while_loop:
            warnings.warn('while-loop not implemented', DeprecationWarning)
        for idx in range(self.timesteps):
            activity = self.build_model(
                i0=idx,
                activity=X,  # Always pass FF drive
                constructor=self.gammanet_constructor)
            tf.add_to_collection('checkpoints', activity)
            if self.timestep_output:
                for l in self.timestep_output:
                    if not hasattr(self.timestep_list, l):
                        self.timestep_list[l] = []
                    self.timestep_list[l] += getattr(self, l)
                self.timestep_list += [activity]
            num_vars = len(tf.trainable_variables())
            if idx > 0:
                assert num_vars == prev_var_count
            prev_var_count = num_vars
        if self.additional_readouts is not None:
            readouts = [activity]
            for r in self.additional_readouts:
                readouts += [getattr(self, r)]
            return readouts
        else:
            return activity
