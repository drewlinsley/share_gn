import os
import tensorflow as tf
from tqdm import tqdm
try:
    from ops import memory_saving_gradients as mem_grads
except Exception as e:
    print(
        'Failed to import memory_saving_gradients in ops/optimizers.py: %s'
        % e)


def superconvergence_schedule(
        base_lr,
        current_step,
        max_lr=3.,
        min_lr=1e-5,
        stepsize=20000.):
    """Implement the superconvergence learning rate schedule."""
    current_step = tf.cast(current_step, tf.float32)
    cycle = tf.floor(tf.cast(1 + current_step / (2 * stepsize), tf.float32))
    x = tf.abs(current_step / stepsize - 2. * cycle + 1.)
    lr = min_lr + (max_lr - min_lr) * tf.maximum(0.0, 1.0 - x)
    return lr


def ilsvrc12_learning_rate_schedule(
        base_lr,
        current_epoch,
        train_batch_size,
        LR_SCHEDULE=[(1.0, 5), (0.1, 30), (0.01, 60), (0.001, 80)]):
    """Handles linear scaling rule, gradual warmup, and LR decay.
    The learning rate starts at 0, then it increases linearly per step.
    After 5 epochs we reach the base learning rate (scaled to account
    for batch size).
    After 30, 60 and 80 epochs the learning rate is divided by 10.
    After 90 epochs training stops and the LR is set to 0. This ensures
    that we train for exactly 90 epochs for reproducibility.
    Args:
    current_epoch: `Tensor` for current epoch.
    Returns:
    A scaled `Tensor` for current learning rate.
    """
    scaled_lr = base_lr * (train_batch_size / 256.0)
    decay_rate = (
        scaled_lr * LR_SCHEDULE[0][0] *
        current_epoch / LR_SCHEDULE[0][1])
    for mult, start_epoch in LR_SCHEDULE:
        decay_rate = tf.where(
            current_epoch < start_epoch,
            decay_rate, scaled_lr * mult)
    return decay_rate


def bsds_learning_rate_schedule(
        base_lr,
        global_step,
        current_epoch,
        num_train_images,
        adjust_epoch=60.,
        decay_rate=0.1):
    """Decay the initial by 10 every 30 epochs."""
    return tf.train.exponential_decay(
        learning_rate=base_lr,
        global_step=global_step,
        decay_steps=num_train_images * adjust_epoch,
        decay_rate=decay_rate,
        staircase=True,
        name='bsds_decay')


def get_lr_schedule(lr, lr_schedule):
    """Return a learning rate schedule if requested."""
    if lr_schedule is None:
        return lr
    elif isinstance(lr_schedule, dict):
        k, v = lr_schedule.items()[0]
        batches_epoch = v[0] / v[1]  # num_train_images / batch size
        global_step = tf.train.get_or_create_global_step()
        current_epoch = tf.cast(global_step, tf.float32) / batches_epoch
        if k is 'ilsvrc12' or k is 'ilsvrc':
            return ilsvrc12_learning_rate_schedule(
                base_lr=lr,
                current_epoch=current_epoch,
                train_batch_size=v[1])
        elif k is 'bsds' or k is 'multicue':
            return bsds_learning_rate_schedule(
                base_lr=lr,
                global_step=global_step,
                num_train_images=batches_epoch,
                current_epoch=current_epoch)
        elif k is 'connectomics' or k is 'connectomics_learning_rate_schedule':
            return bsds_learning_rate_schedule(
                base_lr=lr,
                global_step=global_step,
                num_train_images=batches_epoch,
                adjust_epoch=30.,
                current_epoch=current_epoch)
        elif k is 'superconvergence':
            return superconvergence_schedule(
                base_lr=lr,
                current_step=global_step)
        else:
            raise NotImplementedError(k)
    else:
        raise NotImplementedError(lr_schedule)


def apply_gradients(opt, grads, global_step):
    """Apply gradients."""
    if isinstance(opt, dict):
        var = tf.trainable_variables()
        for k, v in opt:
            import ipdb;ipdb.set_trace()

    else:
        apply_gradient_op = opt.apply_gradients(
            grads,
            global_step=global_step)
    return [apply_gradient_op]


def average_gradients(tower_grads):
    """Calculate the average gradient for each shared variable across all towers.
    Note that this function provides a synchronization point across all towers.
    Args:
      tower_grads: List of lists of (gradient, variable) tuples. The outer list
        is over individual gradients. The inner list is over the gradient
        calculation for each tower.
    Returns:
       List of pairs of (gradient, variable) gradients have been averaged
       across all towers.
    """
    average_grads = []
    if len(tower_grads) == 1:
        return tower_grads[0]
    for grad_and_vars in tqdm(zip(*tower_grads), desc='Merging gradients'):
        # Note that each grad_and_vars looks like the following:
        #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        grads = []
        for g, _ in grad_and_vars:
            # Add 0 dimension to the gradients to represent the tower.
            expanded_g = tf.expand_dims(g, 0)

            # Append on a 'tower' dimension which we will average over below.
            grads.append(expanded_g)


        # Average over the 'tower' dimension.
        grad = tf.concat(axis=0, values=grads)
        grad = tf.reduce_mean(grad, 0)

        # Keep in mind that the Variables are redundant because they are shared
        # across towers. So .. we will just return the first tower's pointer to
        # the Variable.
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads


def get_optimizers(optimizer, lr, dtype):
    """Parse optimizer and return an optimizer object."""
    if isinstance(lr, dict):
        opts = {}
        for k, v in lr.iteritems():
            opts[k] = return_optimizer(optimizer, v, dtype)
    else:
        return return_optimizer(optimizer, lr, dtype)


def return_optimizer(optimizer, lr, dtype):
    """Parse optimizer and return an optimizer object."""
    if dtype == tf.float32 or dtype == tf.bfloat16:
        eps = 1e-08
    elif dtype == tf.float16:  # or dtype == tf.bfloat16:
        eps = 1e-04
    else:
        raise NotImplementedError(
            'Need to figure out eps for dtype: %s' % dtype)
    if optimizer == 'adam':
        optim = lambda x: tf.train.AdamOptimizer(
            x, epsilon=eps)
    elif optimizer == 'adam_keras':
        raise NotImplementedError
        optim = lambda x: tf.keras.optimizers.Adam(
            x, amsgrad=True, epsilon=eps)
    elif optimizer == 'adam_w':
        optim = lambda x: tf.contrib.opt.AdamWOptimizer(
            weight_decay=1e-3, learning_rate=x, epsilon=eps)
    elif optimizer == 'adam_eps':
        optim = lambda x: tf.train.AdamOptimizer(
            x, epsilon=0.1)
    elif optimizer == 'nadam':
        optim = lambda x: tf.contrib.opt.NadamOptimizer(
            x, epsilon=eps)
    elif optimizer == 'power':
        optim = tf.contrib.opt.PowerSignOptimizer
    elif optimizer == 'sgd':
        optim = tf.train.GradientDescentOptimizer
    elif optimizer == 'momentum':
        optim = lambda x: tf.train.MomentumOptimizer(
            learning_rate=x, momentum=0.99, use_nesterov=True)
    elif optimizer == 'rmsprop':
        optim = tf.train.RMSPropOptimizer
    else:
        raise RuntimeError(
            'Cannot understand your loss function: %s' % optimizer)
    optim = optim(lr)
    return optim


def get_optimizer(
        loss,
        lr,
        optimizer,
        lr_schedule=None,
        dtype=tf.float32,
        model=None,
        clip_gradients=None,
        restore_scope=None,
        var_list=None,
        plot_gradients=True,
        save_memory=False,
        constraints=None):
    """Return the optimizer."""
    # Prepare the optimizer
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    optim = return_optimizer(
        optimizer=optimizer,
        lr=lr,
        dtype=dtype,
        loss=loss,
        var_list=var_list,
        clip_gradients=clip_gradients)

    if save_memory:
        tf.__dict__["gradients"] = mem_grads.gradients_memory
        os.environ['TF_CUDNN_USE_AUTOTUNE'] = '0'
        if var_list is None:
            var_list = tf.trainable_variables()
        gvs = [(x, y) for x, y in zip(
            mem_grads.gradients(
                loss,
                var_list,
                checkpoints='memory'),
            var_list)]
    else:
        if var_list:
            gvs = optim.compute_gradients(loss, var_list=var_list)
        else:
            gvs = optim.compute_gradients(loss)
    train_op = check_and_clip_grads(gvs, optim, clip_gradients)
    train_op = tf.group([train_op, update_ops])

    # Prepare learning rate if requested
    lr = get_lr_schedule(lr=lr, lr_schedule=lr_schedule)
    return train_op, lr


def check_grads(gvs):
    """Make sure all variables are in the graph."""
    null_grads = [x for x in gvs if x[0] is None]
    if len(null_grads):
        null_names = [x[1].name for x in null_grads]
        raise RuntimeError(
            'The following vars are not in the backprop graph: %s' %
            null_names)


def apply_grad_clip(grads, clip_gradients):
    """Clip gradients by norm clip_gradients.""" 
    capped_grads, variables = [], []
    for grad, v in grads:
        capped_grads += [tf.clip_by_norm(grad, clip_gradients)]
        variables += [v]
    print('Clipped %s variables at %s norm.' % (len(capped_grads), clip_gradients))
    return zip(capped_grads, variables)


def check_and_clip_grads(
        gvs,
        optim,
        clip_gradients,
        visualize_gradients=False):
    """Check gradients for None and clip if requested."""
    null_grads = [x for x in gvs if x[0] is None]
    if len(null_grads):
        null_names = [x[1].name for x in null_grads]
        raise RuntimeError(
            'The following vars are not in the backprop graph: %s' %
            null_names)
    gradients, variables = zip(*gvs)
    if visualize_gradients:
        for g, v in zip(gradients, variables):
            if 'horizontal' in v.name:
                tf.summary.histogram('grad_%s' % v.name, g)
                tf.summary.histogram('activity_%s' % v.name, v)
    if clip_gradients:
        grads = apply_grad_clip(zip(gradients, variables), clip_gradients)
        return optim.apply_gradients(
            grads,
            global_step=tf.train.get_or_create_global_step())
    else:
        return optim.apply_gradients(
            gvs,
            global_step=tf.train.get_or_create_global_step())

