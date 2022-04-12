import numpy as np
import tensorflow as tf
from utils import py_utils
from ops import training
# from ops import metrics
# from ops import data_structure
from ops import data_loader
from ops import optimizers
from ops import losses
from ops import gradients
from ops import tf_fun


def initialize_tf(
        config,
        directories,
        placeholders,
        restore_global_step=False,
        default_restore=True,
        ckpt=None):
    """Initialize tensorflow model variables."""
    saver = tf.train.Saver(
        var_list=tf.global_variables(),
        max_to_keep=config.save_checkpoints)

    # Filter global_variables for exclusion scope in config
    exclusion_scope, restore_variables = None, tf.global_variables()
    if ckpt is not None:
        ckpt_vars = set([x[0] for x in tf.train.list_variables(ckpt)])
        model_vars = tf.global_variables()
        restore_variables = list(set(
            [
                x.name.split(':')[0]
                for x in model_vars]).intersection(ckpt_vars))
        excluded_variables = list(set(
            [
                x.name.split(':')[0]
                for x in model_vars]).symmetric_difference(ckpt_vars))
        restore_variables = [
            x
            for x in tf.global_variables()
            if x.name.split(':')[0] in restore_variables]
        print(
            'Will restore %s/%s variables. Excluding: %s' % (
                len(restore_variables),
                len(restore_variables) + len(excluded_variables),
                excluded_variables))
    if not len(restore_variables):
        raise RuntimeError('Something went wrong with variable restore.')

    # # Removed this, which is a vestige of training a new readout on a previously trained model
    # # Need to change the config name.
    # if hasattr(config, 'exclusion_scope'):
    #     exclusion_scope = config.exclusion_scope
    #     restore_variables = [
    #         x
    #         for x in restore_variables
    #         if exclusion_scope not in x.name]
    # elif hasattr(config, 'inclusion_scope'):
    #     inclusion_scope = config.inclusion_scope
    #     restore_variables = [
    #         x
    #         for x in restore_variables
    #         if inclusion_scope in x.name]
    if default_restore:
        restore_variables = [
            x
            for x in restore_variables
            if 'Adam' not in x.name]
        restore_variables = [
            x
            for x in restore_variables
            if 'power' not in x.name]
        restore_variables = [
            x
            for x in restore_variables
            if 'moving_mean' not in x.name]
        restore_variables = [
            x
            for x in restore_variables
            if 'moving_variance' not in x.name]
        restore_variables = [
            x
            for x in restore_variables
            if '/beta' not in x.name]
        restore_variables = [
            x
            for x in restore_variables
            if '/gamma' not in x.name]
    if not restore_global_step:
        restore_variables = [
            x
            for x in restore_variables
            if 'global_step' not in x.name]
    restore_saver = tf.train.Saver(
        var_list=restore_variables)
    summary_op = tf.summary.merge_all()
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
    sess.run(
        tf.group(
            tf.global_variables_initializer(),
            tf.local_variables_initializer()))
    summary_writer = tf.summary.FileWriter(
        directories['summaries'],
        sess.graph)
    if not placeholders:
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    else:
        coord, threads = None, None
    return (
        sess,
        saver,
        summary_op,
        summary_writer,
        coord,
        threads,
        restore_saver)


def get_placeholders(train_dataset, val_dataset, config):
    """Create placeholders and apply augmentations."""
    train_images = tf.placeholder(
        dtype=train_dataset.tf_reader['image']['dtype'],
        shape=[config.train_batch_size] + train_dataset.im_size,
        name='train_images')
    train_labels = tf.placeholder(
        dtype=train_dataset.tf_reader['label']['dtype'],
        shape=[config.train_batch_size] + train_dataset.label_size,
        name='train_labels')
    val_images = tf.placeholder(
        dtype=val_dataset.tf_reader['image']['dtype'],
        shape=[config.val_batch_size] + val_dataset.im_size,
        name='val_images')
    val_labels = tf.placeholder(
        dtype=val_dataset.tf_reader['label']['dtype'],
        shape=[config.val_batch_size] + val_dataset.label_size,
        name='val_labels')
    aug_train_ims, aug_train_labels = [], []
    aug_val_ims, aug_val_labels = [], []
    split_train_ims = tf.split(
        train_images, config.train_batch_size, axis=0)
    split_train_labels = tf.split(
        train_labels, config.train_batch_size, axis=0)
    split_val_ims = tf.split(
        val_images, config.val_batch_size, axis=0)
    split_val_labels = tf.split(
        val_labels, config.val_batch_size, axis=0)
    for tr_im, tr_la, va_im, va_la in zip(
            split_train_ims,
            split_train_labels,
            split_val_ims,
            split_val_labels):
        if not np.any(
            np.array(
                tr_im.get_shape().as_list()) == None):
            tr_im = tf.squeeze(tr_im)
            tr_la = tf.squeeze(tr_la)
            va_im = tf.squeeze(va_im)
            va_la = tf.squeeze(va_la)
        else:
            tr_im = tf.squeeze(tr_im, 0)
            tr_la = tf.squeeze(tr_la, 0)
            va_im = tf.squeeze(va_im, 0)
            va_la = tf.squeeze(va_la, 0)
        tr_im, tr_la = data_loader.image_augmentations(
            image=tr_im,
            label=tr_la,
            model_input_image_size=train_dataset.model_input_image_size,
            data_augmentations=config.train_augmentations)
        va_im, va_la = data_loader.image_augmentations(
            image=va_im,
            label=va_la,
            model_input_image_size=val_dataset.model_input_image_size,
            data_augmentations=config.val_augmentations)
        aug_train_ims += [tr_im]
        aug_train_labels += [tr_la]
        aug_val_ims += [va_im]
        aug_val_labels += [va_la]
    aug_train_ims = tf.stack(aug_train_ims, axis=0)
    aug_train_labels = tf.stack(aug_train_labels, axis=0)
    aug_val_ims = tf.stack(aug_val_ims, axis=0)
    aug_val_labels = tf.stack(aug_val_labels, axis=0)
    # return aug_train_ims, aug_train_labels, aug_val_ims, aug_val_labels
    return (
        train_images,
        train_labels,
        val_images,
        val_labels,
        aug_train_ims,
        aug_train_labels,
        aug_val_ims,
        aug_val_labels)


def get_placeholders_test(test_dataset, config):
    """Create test placeholders and apply augmentations."""
    test_images = tf.placeholder(
        dtype=test_dataset.tf_reader['image']['dtype'],
        shape=[config.test_batch_size] + test_dataset.im_size,
        name='test_images')
    test_labels = tf.placeholder(
        dtype=test_dataset.tf_reader['label']['dtype'],
        shape=[config.test_batch_size] + test_dataset.label_size,
        name='test_labels')
    aug_test_ims, aug_test_labels = [], []
    split_test_ims = tf.split(test_images, config.test_batch_size, axis=0)
    split_test_labels = tf.split(test_labels, config.test_batch_size, axis=0)
    for te_im, te_la in zip(
            split_test_ims,
            split_test_labels):
        te_im, te_la = data_loader.image_augmentations(
            image=tf.squeeze(te_im),
            label=tf.squeeze(te_la),
            model_input_image_size=test_dataset.model_input_image_size,
            data_augmentations=config.test_augmentations)
        aug_test_ims += [te_im]
        aug_test_labels += [te_la]
    aug_test_ims = tf.stack(aug_test_ims, axis=0)
    aug_test_labels = tf.stack(aug_test_labels, axis=0)
    return test_images, test_labels, aug_test_ims, aug_test_labels


def build_model(
        exp_params,
        config,
        log,
        dt_string,
        gpu_device,
        cpu_device,
        use_db=True,
        add_config=None,
        placeholders=False,
        checkpoint=None,
        test=False,
        map_out=None,
        num_batches=None,
        tensorboard_images=False):
    """Standard model building routines."""
    config = py_utils.add_to_config(
        d=exp_params,
        config=config)
    if not hasattr(config, 'force_path'):
        config.force_path = False
    exp_label = '%s_%s_%s' % (
        exp_params['model'],
        exp_params['experiment'],
        py_utils.get_dt_stamp())
    directories = py_utils.prepare_directories(config, exp_label)
    dataset_module = py_utils.import_module(
        pre_path=config.dataset_classes,
        module=config.train_dataset)
    train_dataset_module = dataset_module.data_processing()
    if not config.force_path:
        (
            train_data,
            _,
            _) = py_utils.get_data_pointers(
            dataset=train_dataset_module.output_name,
            base_dir=config.tf_records,
            local_dir=config.local_tf_records,
            cv='train')
    else:
        train_data = train_dataset_module.train_path
    dataset_module = py_utils.import_module(
        pre_path=config.dataset_classes,
        module=config.val_dataset)
    val_dataset_module = dataset_module.data_processing()
    if not config.force_path:
        val_data, _, _ = py_utils.get_data_pointers(
            dataset=val_dataset_module.output_name,
            base_dir=config.tf_records,
            local_dir=config.local_tf_records,
            cv='val')
    else:
        val_data = train_dataset_module.val_path
        # val_means_image, val_means_label = None, None

    # Create data tensors
    if hasattr(train_dataset_module, 'aux_loss'):
        train_aux_loss = train_dataset_module.aux_loss
    else:
        train_aux_loss = None
    with tf.device(cpu_device):
        if placeholders and not test:
            # Train with placeholders
            (
                pl_train_images,
                pl_train_labels,
                pl_val_images,
                pl_val_labels,
                train_images,
                train_labels,
                val_images,
                val_labels) = get_placeholders(
                    train_dataset=train_dataset_module,
                    val_dataset=val_dataset_module,
                    config=config)
            train_module_data = train_dataset_module.get_data()
            val_module_data = val_dataset_module.get_data()
            placeholders = {
                'train': {
                    'images': train_module_data[0]['train'],
                    'labels': train_module_data[1]['train']
                },
                'val': {
                    'images': val_module_data[0]['val'],
                    'labels': val_module_data[1]['val']
                },
            }
            train_aux, val_aux = None, None
        elif placeholders and test:
            test_dataset_module = train_dataset_module
            # Test with placeholders
            (
                pl_test_images,
                pl_test_labels,
                test_images,
                test_labels) = get_placeholders_test(
                    test_dataset=test_dataset_module,
                    config=config)
            test_module_data = test_dataset_module.get_data()
            placeholders = {
                'test': {
                    'images': test_module_data[0]['test'],
                    'labels': test_module_data[1]['test']
                },
            }
            train_aux, val_aux = None, None
        else:
            train_images, train_labels, train_aux = data_loader.inputs(
                dataset=train_data,
                batch_size=config.train_batch_size,
                model_input_image_size=train_dataset_module.model_input_image_size,
                tf_dict=train_dataset_module.tf_dict,
                data_augmentations=config.train_augmentations,
                num_epochs=config.epochs,
                aux=train_aux_loss,
                tf_reader_settings=train_dataset_module.tf_reader,
                shuffle=config.shuffle_train)
            if hasattr(val_dataset_module, 'val_model_input_image_size'):
                val_dataset_module.model_input_image_size = val_dataset_module.val_model_input_image_size
            val_images, val_labels, val_aux = data_loader.inputs(
                dataset=val_data,
                batch_size=config.val_batch_size,
                model_input_image_size=val_dataset_module.model_input_image_size,
                tf_dict=val_dataset_module.tf_dict,
                data_augmentations=config.val_augmentations,
                num_epochs=None,
                tf_reader_settings=val_dataset_module.tf_reader,
                shuffle=config.shuffle_val)

    # Build training and val models
    model_spec = py_utils.import_module(
        module=config.model,
        pre_path=config.model_classes)
    if hasattr(train_dataset_module, 'force_output_size'):
        train_dataset_module.output_size = train_dataset_module.force_output_size
    if hasattr(val_dataset_module, 'force_output_size'):
        val_dataset_module.output_size = val_dataset_module.force_output_size
    if hasattr(config, 'loss_function'):
        train_loss_function = config.loss_function
        val_loss_function = config.loss_function
    else:
        train_loss_function = config.train_loss_function
        val_loss_function = config.val_loss_function

    # Route test vs train/val
    h_check = [
        x
        for x in tf.trainable_variables()
        if 'homunculus' in x.name or 'humonculus' in x.name]
    if not hasattr(config, 'default_restore'):
        config.default_restore = False
    if test:
        assert len(gpu_device) == 1, 'Testing only works with 1 gpu.'
        gpu_device = gpu_device[0]
        with tf.device(gpu_device):
            if not placeholders:
                test_images = val_images
                test_labels = val_labels
                test_dataset_module = val_dataset_module
            test_logits, test_vars = model_spec.build_model(
                data_tensor=test_images,
                reuse=None,
                training=False,
                output_shape=test_dataset_module.output_size)
        if test_logits.dtype is not tf.float32:
            test_logits = tf.cast(test_logits, tf.float32)

        # Derive loss
        if not hasattr(config, 'test_loss_function'):
            test_loss_function = val_loss_function
        else:
            test_loss_function = config.test_loss_function
        test_loss = losses.derive_loss(
            labels=test_labels,
            logits=test_logits,
            images=test_images,
            loss_type=test_loss_function)

        # Derive score
        test_score = losses.derive_score(
            labels=test_labels,
            logits=test_logits,
            loss_type=test_loss_function,
            images=test_images,
            score_type=config.score_function)

        # Initialize model
        (
            sess,
            saver,
            summary_op,
            summary_writer,
            coord,
            threads,
            restore_saver) = initialize_tf(
            config=config,
            placeholders=placeholders,
            ckpt=checkpoint,
            default_restore=config.default_restore,
            directories=directories)

        if placeholders:
            proc_images = test_images
            proc_labels = test_labels
            test_images = pl_test_images
            test_labels = pl_test_labels

        # _, H, W, _ = test_vars['model_output_y'].shape
        # H = H // 2
        # W = W // 2
        # jacobian = tf.gradients(test_logits, test_vars['model_output_x'])[0]  # g.batch_jacobian(test_vars['model_output_x'], test_images)
        test_dict = {
            'test_loss': test_loss,
            'test_score': test_score,
            'test_images': test_images,
            'test_labels': test_labels,
            'test_logits': test_logits,
            # 'test_jacobian': jacobian
        }
        if placeholders:
            test_dict['test_proc_images'] = proc_images
            test_dict['test_proc_labels'] = proc_labels
        if len(h_check):
            test_dict['homunculus'] = h_check[0]
        if isinstance(test_vars, dict):
            try:
                for k, v in test_vars.iteritems():
                    test_dict[k] = v
            except:
                for k, v in test_vars.items():
                    test_dict[k] = v
        else:
            test_dict['activity'] = test_vars
    else:
        train_losses, val_losses, tower_grads, norm_updates = [], [], [], []
        train_scores, val_scores = [], []
        train_image_list, train_label_list = [], []
        val_image_list, val_label_list = [], []
        train_reuse = None
        if not hasattr(config, 'lr_schedule'):
            config.lr_schedule = None
        if hasattr(config, 'loss_function'):
            train_loss_function = config.loss_function
            val_loss_function = config.loss_function
        else:
            train_loss_function = config.train_loss_function
            val_loss_function = config.val_loss_function

        # Prepare loop
        if not placeholders:
            train_batch_queue = tf_fun.get_batch_queues(
                images=train_images,
                labels=train_labels,
                gpu_device=gpu_device)
            val_batch_queue = tf_fun.get_batch_queues(
                images=val_images,
                labels=val_labels,
                gpu_device=gpu_device)

        config.lr = optimizers.get_lr_schedule(
            lr=config.lr,
            lr_schedule=config.lr_schedule)
        opt = optimizers.get_optimizers(
            optimizer=config.optimizer,
            lr=config.lr,
            dtype=train_images.dtype)
        with tf.device(cpu_device):
            global_step = tf.train.get_or_create_global_step()
            for i, gpu in enumerate(gpu_device):
                # rs = tf.AUTO_REUSE if i > 0 else None
                with tf.device(gpu):
                    with tf.name_scope('tower_%d' % i) as scope:
                        # Prepare tower data
                        if placeholders:
                            # Multi-gpu: will have to split
                            # train_images per gpu by hand
                            train_image_batch = train_images
                            val_image_batch = val_images
                            train_label_batch = train_labels
                            val_label_batch = val_labels
                        else:
                            (
                                train_image_batch,
                                train_label_batch) = train_batch_queue.dequeue()
                            (
                                val_image_batch,
                                val_label_batch) = val_batch_queue.dequeue()
                        train_image_list += [train_image_batch]
                        train_label_list += [train_label_batch]
                        val_image_list += [val_image_batch]
                        val_label_list += [val_label_batch]

                        # Build models
                        train_logits, train_vars = model_spec.build_model(
                            data_tensor=train_image_batch,
                            reuse=train_reuse,
                            training=True,
                            output_shape=train_dataset_module.output_size)
                        num_training_vars = len(tf.trainable_variables())
                        val_logits, val_vars = model_spec.build_model(
                            data_tensor=val_image_batch,
                            reuse=True,
                            training=False,
                            output_shape=val_dataset_module.output_size)
                        num_validation_vars = len(tf.trainable_variables())
                        assert num_training_vars == num_validation_vars, \
                            'Found a different # of train and val variables.'
                        train_reuse = True

                        # Derive losses
                        if train_logits.dtype is not tf.float32:
                            train_logits = tf.cast(train_logits, tf.float32)
                        if val_logits.dtype is not tf.float32:
                            val_logits = tf.cast(val_logits, tf.float32)
                        train_loss = losses.derive_loss(
                            labels=train_label_batch,
                            logits=train_logits,
                            images=train_image_batch,
                            loss_type=train_loss_function)
                        val_loss = losses.derive_loss(
                            labels=val_label_batch,
                            logits=val_logits,
                            images=val_image_batch,
                            loss_type=val_loss_function)

                        # Derive score
                        train_score = losses.derive_score(
                            labels=train_labels,
                            logits=train_logits,
                            loss_type=train_loss_function,
                            images=train_image_batch,
                            score_type=config.score_function)
                        val_score = losses.derive_score(
                            labels=val_labels,
                            logits=val_logits,
                            loss_type=val_loss_function,
                            images=val_image_batch,
                            score_type=config.score_function)

                        # Add aux losses if requested
                        if hasattr(model_spec, 'weight_decay'):
                            wd = (model_spec.weight_decay() * tf.add_n(
                                [
                                    tf.nn.l2_loss(v)
                                    for v in tf.trainable_variables()
                                    if 'batch_normalization' not in v.name and 'horizontal' not in v.name and 'mu' not in v.name and 'beta' not in v.name and 'intercept' not in v.name]))
                            tf.summary.scalar('weight_decay', wd)
                            train_loss += wd

                        if hasattr(model_spec, 'bsds_weight_decay'):
                            wd = (model_spec.bsds_weight_decay()['l2'] * tf.add_n(
                                [
                                    tf.nn.l2_loss(v)
                                    for v in tf.trainable_variables()
                                    if 'horizontal' not in v.name and 'norm' not in v.name]))
                            tf.summary.scalar('weight_decay_readout', wd)
                            train_loss += wd
                            wd = (model_spec.bsds_weight_decay()['l1'] * tf.add_n(
                                [
                                    tf.reduce_sum(tf.abs(v))
                                    for v in tf.trainable_variables()
                                    if 'horizontal' in v.name]))
                            tf.summary.scalar('weight_decay_horizontal', wd)
                            train_loss += wd

                        if hasattr(model_spec, 'orthogonal'):
                            weights = [
                                v
                                for v in tf.trainable_variables()
                                if 'horizontal' in v.name]
                            assert len(weights) is not None, \
                                'No horizontal weights for laplace.'
                            wd = model_spec.orthogonal() * tf.add_n(
                                [tf_fun.orthogonal(w) for w in weights])
                            tf.summary.scalar('weight_decay', wd)
                            train_loss += wd

                        if hasattr(model_spec, 'laplace'):
                            weights = [
                                v
                                for v in tf.trainable_variables()
                                if 'horizontal' in v.name]
                            assert len(weights) is not None, \
                                'No horizontal weights for laplace.'
                            wd = model_spec.laplace() * tf.add_n(
                                [tf_fun.laplace(w) for w in weights])
                            tf.summary.scalar('weight_decay', wd)
                            train_loss += wd

                        # Derive auxilary losses
                        if hasattr(config, 'aux_loss'):
                            import pdb;pdb.set_trace()
                            aux_loss_type, scale = config.aux_loss.items()[0]
                            for k, v in train_vars.iteritems():
                                # if k in train_dataset_module.aux_loss.keys():
                                # (
                                #     aux_loss_type,
                                #     scale
                                # ) = train_dataset_module.aux_loss[k]
                                train_loss += (losses.derive_loss(
                                    labels=train_labels,
                                    logits=v,
                                    images=train_image_batch,
                                    loss_type=aux_loss_type) * scale)

                        # Gather everything
                        train_losses += [train_loss]
                        val_losses += [val_loss]
                        train_scores += [train_score]
                        val_scores += [val_score]

                        # Compute and store gradients
                        with tf.variable_scope(
                                tf.get_variable_scope(),
                                reuse=tf.AUTO_REUSE):
                            grads = opt.compute_gradients(train_loss)
                        optimizers.check_grads(grads)
                        tower_grads += [grads]

                        # Gather normalization variables
                        norm_updates += [tf.get_collection(
                            tf.GraphKeys.UPDATE_OPS,
                            scope=scope)]

        # Recompute and optimize gradients
        grads = optimizers.average_gradients(tower_grads)
        if hasattr(config, 'clip_gradients') and config.clip_gradients:
            grads = optimizers.apply_grad_clip(grads, config.clip_gradients)
        op_vars = []
        if hasattr(config, 'exclusion_lr') and hasattr(
                config, 'exclusion_scope'):
            grads_0 = [
                x for x in grads if config.exclusion_scope not in x[1].name]
            grads_1 = [
                x for x in grads if config.exclusion_scope in x[1].name]

            if hasattr(config, 'special_scope'):
                grads_2 = [
                    x for x in grads_0 if config.special_scope in x[1].name]
                grads_0 = [
                    x for x in grads_0 if config.special_scope not in x[1].name]

            op_vars_0 = optimizers.apply_gradients(
                opt=opt,
                grads=grads_0,
                global_step=global_step)
            opt_1 = optimizers.get_optimizers(
                optimizer=config.optimizer,
                lr=config.exclusion_lr,
                dtype=train_images.dtype)
            op_vars_1 = optimizers.apply_gradients(
                opt=opt_1,
                grads=grads_1,
                global_step=global_step)
            op_vars += [op_vars_0]
            op_vars += [op_vars_1]
            if hasattr(config, 'special_scope'):
                opt_2 = optimizers.get_optimizers(
                    optimizer=config.optimizer,
                    lr=config.special_lr,
                    dtype=train_images.dtype)
                op_vars_2 = optimizers.apply_gradients(
                    opt=opt_2,
                    grads=grads_2,
                    global_step=global_step)
                op_vars += [op_vars_2]
        else:
            op_vars += [optimizers.apply_gradients(
                opt=opt,
                grads=grads,
                global_step=global_step)]
        if not hasattr(config, 'variable_moving_average'):
            config.variable_moving_average = False
        if config.variable_moving_average:
            variable_averages = tf.train.ExponentialMovingAverage(
                config.variable_moving_average,
                global_step)
            op_vars += [variable_averages.apply(tf.trainable_variables())]
        if len(norm_updates):
            op_vars += [tf.group(*norm_updates)]
        train_op = tf.group(*op_vars)

        # Summarize losses and scores
        train_loss = tf.reduce_mean(train_losses)
        val_loss = tf.reduce_mean(val_losses)
        train_score = tf.reduce_mean(train_scores)
        val_score = tf.reduce_mean(val_scores)
        if len(train_image_list) > 1:
            train_image_list = tf.stack(train_image_list, axis=0)
            train_label_list = tf.stack(train_label_list, axis=0)
        else:
            train_image_list = train_image_list[0]
            train_label_list = train_label_list[0]
        if len(val_image_list) > 1:
            val_image_list = tf.stack(val_image_list, axis=0)
            val_label_list = tf.stack(val_label_list, axis=0)
        else:
            val_image_list = val_image_list[0]
            val_label_list = val_label_list[0]

        tf.summary.scalar('train_loss', train_loss)
        tf.summary.scalar('val_loss', val_loss)
        if tensorboard_images:
            tf.summary.image('train_images', train_images)
            tf.summary.image('val_images', val_images)

        # Initialize model
        (
            sess,
            saver,
            summary_op,
            summary_writer,
            coord,
            threads,
            restore_saver) = initialize_tf(
            config=config,
            placeholders=placeholders,
            ckpt=checkpoint,
            default_restore=config.default_restore,
            directories=directories)

        # Create dictionaries of important training and validation information
        if placeholders:
            proc_train_images = train_images
            proc_train_labels = train_labels
            proc_val_images = val_images
            proc_val_labels = val_labels
            train_images = pl_train_images
            train_labels = pl_train_labels
            val_images = pl_val_images
            val_labels = pl_val_labels

        train_dict = {
            'train_loss': train_loss,
            'train_score': train_score,
            'train_images': train_image_list,
            'train_labels': train_label_list,
            'train_logits': train_logits,
            'train_op': train_op
        }

        if placeholders:
            train_dict['proc_train_images'] = proc_train_images
            train_dict['proc_train_labels'] = proc_train_labels
        if train_aux is not None:
            train_dict['train_aux'] = train_aux
        if tf.contrib.framework.is_tensor(config.lr):
            train_dict['lr'] = config.lr
        else:
            train_dict['lr'] = tf.constant(config.lr)

        if isinstance(train_vars, dict):
            try:
                for k, v in train_vars.iteritems():
                    train_dict[k] = v
            except:
                for k, v in train_vars.items():
                    train_dict[k] = v
        else:
            train_dict['activity'] = train_vars
        if hasattr(config, 'save_gradients') and config.save_gradients:
            grad = tf.gradients(train_logits, train_images)[0]
            if grad is not None:
                train_dict['gradients'] = grad
            else:
                log.warning('Could not calculate val gradients.')

        val_dict = {
            'val_loss': val_loss,
            'val_score': val_score,
            'val_images': val_image_list,
            'val_logits': val_logits,
            'val_labels': val_label_list,
        }
        if placeholders:
            val_dict['proc_val_images'] = proc_val_images
            val_dict['proc_val_labels'] = proc_val_labels
        if val_aux is not None:
            val_dict['aux'] = val_aux

        if isinstance(val_vars, dict):
            try:
                for k, v in val_vars.iteritems():
                    val_dict[k] = v
            except:
                for k, v in val_vars.items():
                    val_dict[k] = v
        else:
            val_dict['activity'] = val_vars
        if hasattr(config, 'save_gradients') and config.save_gradients:
            grad = tf.gradients(val_logits, val_images)[0]
            if grad is not None:
                val_dict['gradients'] = grad
            else:
                log.warning('Could not calculate val gradients.')
        if len(h_check):
            val_dict['homunculus'] = h_check[0]

    # Add optional info to the config
    if add_config is not None:
        extra_list = add_config.split(',')
        for eidx, extra in enumerate(extra_list):
            setattr(config, 'extra_%s' % eidx, extra)

    # Count parameters
    num_params = tf_fun.count_parameters(var_list=tf.trainable_variables())
    print('Model has approximately %s trainable params.' % num_params)
    if test:
        return training.test_loop(
            log=log,
            config=config,
            sess=sess,
            summary_op=summary_op,
            summary_writer=summary_writer,
            saver=saver,
            restore_saver=restore_saver,
            directories=directories,
            test_dict=test_dict,
            exp_label=exp_label,
            num_params=num_params,
            checkpoint=checkpoint,
            num_batches=num_batches,
            save_weights=config.save_weights,
            save_checkpoints=config.save_checkpoints,
            save_activities=config.save_activities,
            save_gradients=config.save_gradients,
            map_out=map_out,
            placeholders=placeholders)
    else:
        # Start training loop
        training.training_loop(
            log=log,
            config=config,
            coord=coord,
            sess=sess,
            summary_op=summary_op,
            summary_writer=summary_writer,
            saver=saver,
            restore_saver=restore_saver,
            threads=threads,
            directories=directories,
            train_dict=train_dict,
            val_dict=val_dict,
            exp_label=exp_label,
            num_params=num_params,
            checkpoint=checkpoint,
            use_db=use_db,
            save_weights=config.save_weights,
            save_checkpoints=config.save_checkpoints,
            save_activities=config.save_activities,
            save_gradients=config.save_gradients,
            placeholders=placeholders)
