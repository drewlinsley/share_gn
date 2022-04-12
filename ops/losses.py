import numpy as np
import tensorflow as tf
from ops.data_loader import rgb_to_lab


def derive_loss(labels, logits, loss_type, images=None):
    """Derive loss_type between labels and logits."""
    assert loss_type is not None, 'No loss_type declared'
    if loss_type == 'sparse_ce':  # or loss_type == 'cce':
        logits = tf.cast(logits, tf.float32)
        return tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=tf.reshape(labels, [-1]),
                logits=logits))
    elif loss_type == 'contrastive':
        exp_logits = tf.math.exp(logits)
        pos = exp_logits[:, 0]
        total = tf.reduce_sum(exp_logits, -1)
        return -tf.reduce_sum(tf.log(pos / (total)))  #  * 100000
        # return -tf.reduce_sum((pos / (total + 1e-8)))
    elif loss_type == 'coco_ce':
        logits = tf.cast(logits, tf.float32)
        label_shape = labels.get_shape().as_list()
        if label_shape[-1] > 1:
            raise RuntimeError('Label shape is %s.' % label_shape)
        labels = tf.squeeze(labels, -1)
        labels = tf.cast(tf.round(labels), tf.int32)
        oh_labels = tf.one_hot(
            labels, logits.get_shape().as_list()[-1], axis=-1)
        weights = np.load(
            '/media/data_cifs/lakshmi/coco_class_weights.npz')[
                'class_weights'][None, None, None, :]
        class_weights = tf.reduce_sum(
            tf.multiply(oh_labels, weights), -1)
        return tf.losses.softmax_cross_entropy(
            onehot_labels=oh_labels, weights=class_weights, logits=logits)
    elif loss_type == 'sparse_ce_image' or loss_type == 'sparse_cce_image':
        logits = tf.cast(logits, tf.float32)
        label_shape = labels.get_shape().as_list()
        if label_shape[-1] > 1:
            raise RuntimeError('Label shape is %s.' % label_shape)
        labels = tf.squeeze(labels, -1)
        labels = tf.cast(labels, tf.int32)
        return tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=labels,
                logits=logits))
    elif loss_type == 'timestep_sparse_ce_image':
        labels = tf.squeeze(labels, -1)
        labels = tf.cast(labels, tf.int32)
        losses = []
        for logit in logits:
            losses += [
                tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=labels,
                    logits=logit)]
        return tf.reduce_mean(losses)
    elif loss_type == 'cce_image':
        logits = tf.cast(logits, tf.float32)
        labels = tf.cast(labels, tf.float32)
        return tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits_v2(
                labels=labels,
                logits=logits,
                dim=-1))
    elif loss_type == 'bce':
        # logits = tf.contrib.layers.flatten(tf.cast(logits, tf.float32))
        # labels = tf.contrib.layers.flatten(tf.cast(labels, tf.float32))
        logit_shape = np.array(logits.get_shape().as_list())
        label_shape = np.array(labels.get_shape().as_list())
        logits = tf.cast(logits, tf.float32)
        labels = tf.cast(labels, tf.float32)
        assert np.all(logit_shape == label_shape), 'Logit/label shape mismatch'
        return tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                labels=labels,
                logits=logits))
    elif loss_type == 'weighted_bce':
        # logits = tf.contrib.layers.flatten(tf.cast(logits, tf.float32))
        # labels = tf.contrib.layers.flatten(tf.cast(labels, tf.float32))
        logit_shape = np.array(logits.get_shape().as_list())
        label_shape = np.array(labels.get_shape().as_list())
        labels = tf.cast(tf.greater(labels, 0), tf.float32)
        count_neg = tf.reduce_sum(tf.cast(tf.equal(labels, 0), tf.float32))
        total = np.prod(labels.get_shape().as_list()[1:])
        # num_neg = total - num_pos
        beta = tf.cast(count_neg / (total), tf.float32)
        pos_weight = beta / (1 - beta)
        assert np.all(logit_shape == label_shape), 'Logit/label shape mismatch'
        return tf.reduce_mean(
            tf.nn.weighted_cross_entropy_with_logits(
                targets=labels,
                logits=logits,
                pos_weight=pos_weight))
    elif loss_type == 'weighted_cce':
        logits = tf.cast(logits, tf.float32)
        labels = tf.cast(tf.greater(labels, 1e-4), tf.float32)
        count_neg = tf.reduce_sum(tf.cast(tf.equal(labels, 0), tf.float32))
        total = np.prod(labels.get_shape().as_list()[1:])
        # num_neg = total - num_pos
        beta = tf.cast(count_neg / (total), tf.float32)
        pos_weight = beta / (1 - beta)
        labels = tf.cast(labels, tf.int64)
        return tf.losses.sparse_softmax_cross_entropy(
            labels=labels,  # tf.expand_dims(labels, -1),
            logits=logits, weights=pos_weight)
    elif loss_type == 'bsds_weighted_bce':
        # logits = tf.contrib.layers.flatten(tf.cast(logits, tf.float32))
        # labels = tf.contrib.layers.flatten(tf.cast(labels, tf.float32))
        logit_shape = np.array(logits.get_shape().as_list())
        label_shape = np.array(labels.get_shape().as_list())
        labels = tf.cast(tf.greater(labels, 0.15), tf.float32)
        count_neg = tf.reduce_sum(tf.cast(tf.equal(labels, 1), tf.float32))
        total = np.prod(labels.get_shape().as_list()[1:])
        # num_neg = total - num_pos
        beta = tf.cast(count_neg / (total), tf.float32)
        pos_weight = beta / (1 - beta)
        assert np.all(logit_shape == label_shape), 'Logit/label shape mismatch'
        return tf.reduce_mean(
            tf.nn.weighted_cross_entropy_with_logits(
                targets=labels,
                logits=logits,
                pos_weight=pos_weight))
    elif loss_type == 'snemi_bce':
        logit_shape = np.array(logits.get_shape().as_list())
        label_shape = np.array(labels.get_shape().as_list())
        assert np.all(logit_shape == label_shape), 'Logit/label shape mismatch'
        pos_weight = np.array([
            2.418742781547052,
            1.8065865879883696,
            2.0661867016503397,
            1.5265990514073233,
            1.4459855433953799,
            1.3630231018175727,
            1.1505136412052275,
            1.261153265450892])[None, None, None, :]
        return tf.reduce_mean(
            tf.nn.weighted_cross_entropy_with_logits(
                targets=labels,
                logits=logits,
                pos_weight=pos_weight))
    elif loss_type == 'berson_single_bce':
        logit_shape = np.array(logits.get_shape().as_list())
        label_shape = np.array(labels.get_shape().as_list())
        assert np.all(logit_shape == label_shape), 'Logit/label shape mismatch'
        pos_weight = np.array([
            0.5])
        return tf.reduce_mean(
            tf.nn.weighted_cross_entropy_with_logits(
                targets=labels,
                logits=logits,
                pos_weight=pos_weight))
    elif loss_type == 'berson_bce':
        logit_shape = np.array(logits.get_shape().as_list())
        label_shape = np.array(labels.get_shape().as_list())
        assert np.all(logit_shape == label_shape), 'Logit/label shape mismatch'
        pos_weight = 1 / (tf.cast(
            tf.reduce_sum(labels), tf.float32) / np.prod(label_shape))
        # pos_weight = np.array([
        #     1.996855538310967,
        #     1.649734254927337,
        #     1.5926006782333113,
        #     1.4752676526044315])[None, None, None, :]
        #     0.9751423370980603,
        #     1.3418085446925088,
        #     0.8214196863251048,
        #     1.2433403543158281])[None, None, None, :]
        return tf.reduce_mean(
            tf.nn.weighted_cross_entropy_with_logits(
                targets=labels,
                logits=logits,
                pos_weight=pos_weight))
    elif loss_type == 'ae_l2_loss':
        images = images + [103.94, 116.78, 123.68]  # [123.68, 116.78, 103.94]
        # images = tf.stack(
        #     [images[..., 2], images[..., 1], images[..., 0]], axis=-1)
        images = images / 255
        # images = tf.clip_by_value(images, 0, 1)
        # images = (images * 2) - 1
        # logits = tf.tanh(logits)
        # return tf.reduce_mean(tf.pow((logits - images), 2))
        return tf.nn.l2_loss(logits - images)
    elif loss_type == 'ae_cce_loss':
        images = images + [103.94, 116.78, 123.68]  # [123.68, 116.78, 103.94]
        images = tf.clip_by_value(images, 0, 255)
        # images = tf.stack(
        #     [images[..., 2], images[..., 1], images[..., 0]], axis=-1)
        images = tf.cast(images, tf.int64)
        # images = tf.clip_by_value(images, 0, 1)
        # images = (images * 2) - 1
        # logits = tf.tanh(logits)
        return tf.losses.sparse_softmax_cross_entropy(labels=images, logits=logits)
    elif loss_type == 'ae_bce_loss':
        images = images + [103.94, 116.78, 123.68]  # [123.68, 116.78, 103.94]
        images = tf.stack(
            [images[..., 2], images[..., 1], images[..., 0]], axis=-1)
        images = images / 255
        images = tf.clip_by_value(images, 0, 1)
        # Now convert to lab
        images = tf.squeeze(images)
        images = rgb_to_lab(images)
        images = images[None]
        # images = tf.stop_gradient(images)
        images = (1 + images) / 2
        images = tf.clip_by_value(images, 0, 1)
        import pdb;pdb.set_trace()
        return tf.losses.sigmoid_cross_entropy(images, logits)
    elif loss_type == 'connectomics_bce':
        logit_shape = np.array(logits.get_shape().as_list())
        label_shape = np.array(labels.get_shape().as_list())
        assert np.all(logit_shape == label_shape), 'Logit/label shape mismatch'
        return tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                labels=labels[..., :2],
                logits=logits[..., :2]))
    elif loss_type == 'dice':
        gamma = 0.15
        pos_labels = tf.cast(tf.greater(labels, gamma), tf.float32)
        neg_labels = tf.cast(tf.equal(labels, 0.), tf.float32)
        mask = pos_labels + neg_labels
        dice = dice_loss(
            logits=logits,
            labels=mask,  # labels,
            mask=mask,
            gamma=False)
    elif loss_type == 'dice_bce':
        gamma = 0.4
        pos_labels = tf.cast(tf.greater(labels, gamma), tf.float32)
        neg_labels = tf.cast(tf.less_equal(labels, 0.), tf.float32)
        y = tf.cast(
            tf.where(
                tf.greater(labels, gamma),
                tf.ones_like(labels),
                labels), tf.float32)
        mask = pos_labels + neg_labels
        dice = dice_loss(
            logits=logits,
            labels=y,
            mask=mask,
            # labels=labels,
            gamma=False)
        bce = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=y,
            logits=logits) * mask
        return dice + (0.001 * tf.reduce_mean(bce))
    elif loss_type == 'bi_bce':
        return hed_loss(
            logits=logits,
            labels=labels,
            gamma=.5)  # 0.5
    elif loss_type == 'pfc_bi_bce':
        self_sup = images
        return hed_loss(
            logits=logits,
            labels=labels,
            gamma=.5)  # 0.5
    elif loss_type == 'occlusion_bi_bce':
        return occlusion_hed_loss(
            logits=logits,
            labels=labels,
            gamma=.5)  # 0.5
    elif loss_type == 'impute_bi_bce':
        loss = hed_loss(
            logits=logits,
            labels=labels,
            reduction=False,
            gamma=.5)
        mask = tf.cast(tf.greater_equal(labels, 0.), tf.float32)
        return tf.reduce_sum(loss * mask)
    elif loss_type == 'bi_bce_hed':
        loss = hed_loss(
            logits=logits,
            labels=labels,
            reduction=False,
            gamma=.5)
        mask = tf.cast(tf.greater_equal(labels, 0), tf.float32)
        return tf.reduce_sum(loss * mask)
    elif loss_type == 'bi_bce_hed_g4':
        loss = hed_loss(
            logits=logits,
            labels=labels,
            reduction=False,
            gamma=.4)
        mask = tf.cast(tf.greater_equal(labels, 0), tf.float32)
        return tf.reduce_sum(loss * mask)
    elif loss_type == 'bi_bce_edges' or loss_type == 'bi_bce_edge':
        return hed_loss(
            logits=logits,
            labels=labels,
            gamma=.3)
    elif loss_type == 'bi_bce_boundaries':
        return hed_loss(
            logits=logits,
            labels=labels,
            gamma=.4)
    elif loss_type == 'bi_bce_boundaries_bdcn':
        return hed_loss(
            logits=logits,
            labels=labels,
            gamma=.4)
    elif loss_type == 'bi_bce_edges_bdcn':
        return hed_loss(
            logits=logits,
            labels=labels,
            gamma=.3)
    elif loss_type == 'hed_bce':
        return hed_loss_bak(
            logits=logits,
            labels=labels,
            gamma=.5)
    elif loss_type == 'bsds_bce':
        return bsds_bce_loss(logits=logits, labels=labels, gamma=1e-3)
    elif loss_type == 'multicue_bce':
        gamma, balance = 0.4, 1.1
        logits = tf.contrib.layers.flatten(tf.cast(logits, tf.float32))
        labels = tf.contrib.layers.flatten(tf.cast(labels, tf.float32))
        logit_shape = np.array(logits.get_shape().as_list())
        label_shape = np.array(labels.get_shape().as_list())
        assert np.all(logit_shape == label_shape), 'Logit/label shape mismatch'
        # Select dispositive 0s and values > gamma
        pos_mask = tf.greater(labels, gamma)
        neg_mask = tf.equal(labels, 0.)
        float_pos_mask = tf.cast(pos_mask, tf.float32)
        float_neg_mask = tf.cast(neg_mask, tf.float32)
        pos_weight = tf.reduce_sum(float_pos_mask)
        neg_weight = tf.reduce_sum(float_neg_mask)
        total = pos_weight + neg_weight
        pos_weights = float_pos_mask * (pos_weight * balance / total)
        neg_weights = float_neg_mask * (neg_weight * 1 / total)
        weights = pos_weights + neg_weights
        return tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                labels=labels,
                logits=logits) * weights)
    elif loss_type == 'cce':
        logits = tf.cast(logits, tf.float32)
        labels = tf.cast(labels, tf.int64)
        return tf.losses.sparse_softmax_cross_entropy(
            labels=labels,  # tf.expand_dims(labels, -1),
            logits=logits)
    elif loss_type == 'l2':
        logits = tf.cast(logits, tf.float32)
        labels = tf.cast(labels, tf.float32)
        return tf.nn.l2_loss(labels - logits)
    elif loss_type == 'mse':
        logits = tf.cast(logits, tf.float32)
        labels = tf.cast(labels, tf.float32)
        return tf.sqrt(tf.reduce_mean((labels - logits) ** 2))
    elif loss_type == 'masked_mse':
        logits = tf.cast(logits, tf.float32)
        labels = tf.cast(labels, tf.float32)
        mask = tf.cast(tf.equal(labels, -999.), tf.float32)
        distance = ((labels - logits) * mask) ** 2
        distance = tf.where(
            tf.is_nan(distance),
            tf.zeros_like(distance), distance)
        return tf.sqrt(tf.reduce_mean(distance))
    elif loss_type == 'mirror_invariant_l2_grating':
        logits = tf.cast(logits, tf.float32)
        labels = tf.cast(labels, tf.float32)
        return tf.minimum(
            tf.nn.l2_loss(labels - logits),
            tf.nn.l2_loss(-labels - logits))
    elif loss_type == 'l2_image':
        logits = tf.cast(logits, tf.float32)
        return tf.nn.l2_loss(labels - logits)
    elif loss_type == 'snemi_bce_loss':
        logits = tf.cast(logits, tf.float32)
        return snemi_bce_loss(
            labels=labels,
            logits=logits)
    elif loss_type == 'pearson' or loss_type == 'correlation':
        logits = tf.cast(logits, tf.float32)
        return pearson_dissimilarity(
            labels=labels,
            logits=logits,
            REDUCE=tf.reduce_mean)
    elif loss_type == 'sigmoid_pearson':
        logits = tf.cast(logits, tf.float32)
        return pearson_dissimilarity(
            labels=labels,
            logits=tf.nn.sigmoid(logits),
            REDUCE=tf.reduce_mean)
    elif loss_type == 'pixel_error':
        pred = tf.cast(tf.greater(tf.nn.sigmoid(logits), .5), tf.int32)
        error = tf.cast(
            tf.not_equal(pred, tf.cast(labels, tf.int32)), tf.float32)
        return tf.reduce_mean(error, name='pixel_error')
    else:
        raise NotImplementedError(loss_type)


def derive_score(labels, logits, score_type, loss_type, images=None):
    """Derive score_type between labels and logits."""
    assert score_type is not None, 'No score_type declared'
    if score_type == 'sparse_ce' or score_type == 'cce':
        logits = tf.cast(logits, tf.float32)
        return tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=tf.reshape(labels, [-1]),
                logits=logits))
    elif score_type == 'bce':
        # logits = tf.contrib.layers.flatten(tf.cast(logits, tf.float32))
        # labels = tf.contrib.layers.flatten(tf.cast(labels, tf.float32))
        logit_shape = np.array(logits.get_shape().as_list())
        label_shape = np.array(labels.get_shape().as_list())
        assert np.all(logit_shape == label_shape), 'Logit/label shape mismatch'
        return tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                labels=labels,
                logits=logits))
    elif score_type == 'l2':
        logits = tf.cast(logits, tf.float32)
        labels = tf.cast(labels, tf.float32)
        return tf.nn.l2_loss(labels - logits)
    elif score_type == 'mse':
        logits = tf.cast(logits, tf.float32)
        labels = tf.cast(labels, tf.float32)
        return tf.sqrt(tf.reduce_mean((labels - logits) ** 2))
    elif score_type == 'masked_mse':
        logits = tf.cast(logits, tf.float32)
        labels = tf.cast(labels, tf.float32)
        mask = tf.cast(tf.equal(labels, -999.), tf.float32)
        distance = ((labels - logits) * mask) ** 2
        distance = tf.where(
            tf.is_nan(distance),
            tf.zeros_like(distance), distance)
        return tf.sqrt(tf.reduce_mean(distance))
    elif score_type == 'pixel_error':
        pred = tf.cast(tf.greater(tf.nn.sigmoid(logits), .5), tf.int32)
        error = tf.cast(
            tf.not_equal(pred, tf.cast(labels, tf.int32)), tf.float32)
        return tf.reduce_mean(error, name='pixel_error')
    elif score_type == 'pearson' or score_type == 'correlation':
        logits = tf.cast(logits, tf.float32)
        return pearson_dissimilarity(
            labels=labels,
            logits=logits,
            REDUCE=tf.reduce_mean)
    elif score_type == 'sigmoid_pearson':
        logits = tf.cast(logits, tf.float32)
        return pearson_dissimilarity(
            labels=labels,
            logits=tf.nn.sigmoid(logits),
            REDUCE=tf.reduce_mean)
    elif score_type == 'snemi_bce_loss':
        logits = tf.cast(logits, tf.float32)
        return snemi_bce_loss(
            labels=labels,
            logits=logits)
    elif score_type == 'connectomics_bce':
        logit_shape = np.array(logits.get_shape().as_list())
        label_shape = np.array(labels.get_shape().as_list())
        assert np.all(logit_shape == label_shape), 'Logit/label shape mismatch'
        return tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                labels=labels[..., :2],
                logits=logits[..., :2]))
    elif loss_type == 'mirror_invariant_l2_grating':
        logits = tf.cast(logits, tf.float32)
        labels = tf.cast(labels, tf.float32)
        return tf.minimum(
            tf.nn.l2_loss(labels - logits),
            tf.nn.l2_loss(-labels - logits))
    elif score_type == 'prop_positives':
        total_pos = tf.cast(
            tf.reduce_sum(
                labels), tf.float32)
        correct = tf.cast(
            tf.equal(
                labels, tf.cast(
                    tf.greater(logits, 0.5), tf.float32)), tf.float32)
        return tf.reduce_sum(correct) / total_pos
    elif score_type == 'weighted_cce':
        logits = tf.cast(logits, tf.float32)
        labels = tf.cast(tf.greater(labels, 1e-4), tf.float32)
        count_neg = tf.reduce_sum(tf.cast(tf.equal(labels, 0), tf.float32))
        total = np.prod(labels.get_shape().as_list()[1:])
        # num_neg = total - num_pos
        beta = tf.cast(count_neg / (total), tf.float32)
        pos_weight = beta / (1 - beta)
        labels = tf.cast(labels, tf.int64)
        return tf.losses.sparse_softmax_cross_entropy(
            labels=labels,  # tf.expand_dims(labels, -1),
            logits=logits, weights=pos_weight)
    elif score_type == 'f1':
        return f1_score(logits=logits, labels=labels)
    elif score_type == 'bsds_f1':
        return f1_score(logits=logits, labels=labels, threshold_labels=0.3)
    elif score_type == 'snemi_f1':
        logits = tf.reduce_mean(logits[..., :2], axis=-1)
        labels = tf.cast(tf.reduce_mean(labels[..., :2], axis=-1), tf.float32)
        labels = tf.expand_dims(labels, axis=-1)
        return f1_score(logits=logits, labels=labels, threshold_labels=0.3)
    elif score_type == 'timestep_f1':
        return f1_score(logits=logits[-1], labels=labels)
    elif score_type == 'ae_l2_loss':
        images = images + [103.94, 116.78, 123.68]  # [123.68, 116.78, 103.94]
        # images = tf.stack(
        #     [images[..., 2], images[..., 1], images[..., 0]], axis=-1)
        images = images / 255
        # images = tf.clip_by_value(images, 0, 1)
        # images = (images * 2) - 1
        # logits = tf.tanh(logits)
        return tf.nn.l2_loss(logits - images)
        # return tf.reduce_mean(tf.pow((logits - images), 2))
    elif score_type == 'ae_cce_loss':
        images = images + [103.94, 116.78, 123.68]  # [123.68, 116.78, 103.94]
        images = tf.clip_by_value(images, 0, 255)
        # images = tf.stack(
        #     [images[..., 2], images[..., 1], images[..., 0]], axis=-1)
        images = tf.cast(images, tf.int64)
        # images = tf.clip_by_value(images, 0, 1)
        # images = (images * 2) - 1
        # logits = tf.tanh(logits)
        return tf.losses.sparse_softmax_cross_entropy(labels=images, logits=logits)
        # return tf.reduce_mean(tf.pow((logits - images), 2))
    elif score_type == 'ae_bce_loss':
        images = images + [103.94, 116.78, 123.68]  # [123.68, 116.78, 103.94]
        images = tf.stack(
            [images[..., 2], images[..., 1], images[..., 0]], axis=-1)
        images = images / 255
        images = tf.clip_by_value(images, 0, 1)
        images = tf.squeeze(images)
        images = rgb_to_lab(images)
        images = images[None]
        # images = tf.stop_gradient(images)
        # Convert [-1, 1] to [0, 1]
        images = (1 + images) / 2
        images = tf.clip_by_value(images, 0, 1)
        return tf.losses.sigmoid_cross_entropy(images, logits)
        # return tf.nn.l2_loss(logits - images)
    elif score_type == 'contrastive':
        exp_logits = tf.math.exp(logits)
        pos = exp_logits[:, 0]
        total = tf.reduce_sum(exp_logits, -1)
        return -tf.reduce_sum(tf.log(pos / (total)))  #   * 100000
        # return -tf.reduce_sum((pos / (total + 1e-8)))
    elif score_type == 'top_5':
        logit_shape = logits.get_shape().as_list()
        label_shape = labels.get_shape().as_list()
        if np.any(logit_shape == 1):
            logits = np.squeeze(logits)
        if np.any(label_shape == 1):
            labels = np.squeeze(labels)
        labs = tf.cast(labels, tf.int64)
        return tf.reduce_mean(
            tf.cast(
                tf.nn.in_top_k(
                    predictions=tf.cast(
                        tf.argsort(
                            logits,
                            axis=-1,
                            direction='DESCENDING'),
                        tf.float32),
                    targets=labels,
                    k=5),
                tf.float32))
    elif score_type == 'accuracy':
        logit_shape = logits.get_shape().as_list()
        label_shape = labels.get_shape().as_list()
        if np.any(logit_shape == 1):
            logits = np.squeeze(logits)
        if np.any(label_shape == 1):
            labels = np.squeeze(labels)
        # logits = tf.contrib.layers.flatten(tf.cast(logits, tf.float32))
        logits = tf.cast(logits, tf.float32)
        # labels = tf.squeeze(labels, -1)
        # labels = tf.contrib.layers.flatten(tf.cast(labels, tf.float32))
        labs = tf.cast(labels, tf.int64)
        if loss_type == 'cce':
            preds = tf.argmax(logits, axis=-1)
            # preds = tf.cast(preds, tf.int32)
            # labs = tf.squeeze(labs, axis=-1)
        elif loss_type == 'cce_image':
            preds = tf.argmax(logits, axis=-1)
            # preds = tf.cast(preds, tf.int32)
            labs = tf.argmax(labs, axis=-1)
        elif (
                loss_type == 'sparse_cce_image' or
                loss_type == 'sparse_ce_image' or
                loss_type == 'coco_ce'):
            preds = tf.argmax(logits, axis=-1)
            if (
                len(
                    labs.get_shape().as_list()) == 4 and
                    labs.get_shape().as_list()[-1] == 1):
                labs = tf.squeeze(labs, -1)
        elif loss_type == 'bce':
            # if 0:  # len(labs.get_shape().as_list()) > 1:
            #     labs = tf.squeeze(labs)
            # else:
            labs = tf.squeeze(labs)
            # if 0:  # len(logits.get_shape().as_list()) > 1:
            #     #preds = tf.squeeze(logits)
            # else:
            preds = tf.greater(tf.squeeze(logits), 0)
            # preds = tf.greater(tf.squeeze(logits), 0.5)
        else:
            raise NotImplementedError(
                'Cannot understand requested metric w/ loss.')
        labs = tf.cast(labs, tf.float32)
        preds = tf.cast(preds, tf.float32)
        return tf.reduce_mean(tf.cast(tf.equal(preds, labs), tf.float32))
    else:
        raise NotImplementedError(score_type)


def dice_loss(logits, labels, smooth=1e-8, gamma=0.3, mask=None):
    """Dice loss on truncated labels."""
    dice_logits = tf.nn.sigmoid(logits)
    # dice_logits = tf.contrib.layers.flatten(tf.cast(dice_logits, tf.float32))
    # labels = tf.contrib.layers.flatten(tf.cast(labels, tf.float32))
    labels = tf.cast(labels, tf.float32)
    if gamma:
        pos_labels = tf.cast(tf.greater(labels, gamma), tf.float32)
        neg_labels = tf.cast(tf.equal(labels, 0.), tf.float32)
        labels = pos_labels + neg_labels
    logit_shape = np.array(dice_logits.get_shape().as_list())
    label_shape = np.array(labels.get_shape().as_list())
    assert np.all(logit_shape == label_shape), 'Logit/label shape mismatch'
    if mask is not None:
        dice_logits *= mask
        labels *= mask
    intersection = tf.reduce_sum(dice_logits * labels)
    logit_sum = tf.reduce_sum(dice_logits)
    label_sum = tf.reduce_sum(labels)
    return 1. - ((
        2. * intersection + smooth) / (logit_sum + label_sum + smooth))


# def hed_loss(
def hed_loss_bak(
        logits,
        labels,
        gamma,
        pool=None):
    with tf.name_scope('class_balanced_sigmoid_cross_entropy'):
        if gamma:
            pos_labels = tf.cast(tf.greater(labels, gamma), tf.float32)
            neg_labels = tf.cast(tf.less_equal(labels, 0.), tf.float32)
            mask = pos_labels + neg_labels
            y = pos_labels  # + neg_labels
        else:
            y = tf.cast(labels, tf.float32)
            mask = None
        count_neg = tf.reduce_sum(1. - y)
        count_pos = tf.reduce_sum(y)
        beta = tf.cast(count_neg / (count_neg + count_pos), tf.float32)
        pos_weight = beta / (1 - beta)
        cost = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=y, logits=logits) * pos_weight
        if mask is not None:
            cost *= mask
        if pool is not None:
            cost = tf.layers.max_pooling2d(
                inputs=cost,
                pool_size=pool,
                strides=(1, 1))
        return tf.reduce_mean(cost * (1 - beta))


# def hed_loss_bak(
def hed_loss(
        logits,
        labels,
        gamma,
        reduction=True):
    with tf.name_scope('class_balanced_sigmoid_cross_entropy'):
        y = tf.cast(
            tf.where(
                tf.greater_equal(labels, gamma),
                tf.ones_like(labels),
                labels), tf.float32)
        pos_loc = tf.cast(tf.equal(y, 1), tf.float32)
        neg_loc = tf.cast(tf.equal(y, 0), tf.float32)
        pos = tf.reduce_sum(pos_loc)
        neg = tf.reduce_sum(neg_loc)
        valid = neg + pos
        pos_loc *= (neg * 1. / valid)
        neg_loc *= (pos * 1.1 / valid)
        weights = pos_loc + neg_loc
        # cost = tf.nn.weighted_cross_entropy_with_logits(
        #     logits=logits, targets=y, pos_weight=weights)
        cost = tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=logits)
        if reduction:
            return tf.reduce_sum(cost * weights)
        else:
            return cost * weights


def occlusion_hed_loss(
        logits,
        labels,
        gamma,
        reduction=True):
    with tf.name_scope('class_balanced_sigmoid_cross_entropy'):
        y = tf.cast(
            tf.where(
                tf.greater_equal(labels, gamma),
                tf.ones_like(labels),
                labels), tf.float32)
        pos_loc = tf.cast(tf.equal(y, 1), tf.float32)
        neg_loc = tf.cast(tf.equal(y, 0), tf.float32)
        pos = tf.reduce_sum(pos_loc)
        neg = tf.reduce_sum(neg_loc)
        valid = neg + pos
        pos_loc *= (neg * 1. / valid)
        neg_loc *= (pos * 1.1 / valid)
        weights = pos_loc + neg_loc
        # cost = tf.nn.weighted_cross_entropy_with_logits(
        #     logits=logits, targets=y, pos_weight=weights)
        cost = tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=logits)
        mask = tf.cast(tf.less(labels, 0), tf.float32)
        cost *= (1 - mask)
        if reduction:
            return tf.reduce_sum(cost * weights)
        else:
            return cost * weights


def bsds_bce_loss(logits, labels, gamma=0.3):
    """Weighted CE from HED models for training on BSDS."""
    logit_shape = np.array(logits.get_shape().as_list())
    label_shape = np.array(labels.get_shape().as_list())
    assert np.all(logit_shape == label_shape), 'Logit/label shape mismatch'

    # Select dispositive 0s and values > gamma
    pos_mask = tf.greater_equal(labels, gamma)
    neg_mask = tf.equal(labels, 0)
    float_pos_mask = tf.cast(pos_mask, tf.float32)
    float_neg_mask = tf.cast(neg_mask, tf.float32)
    pos_weight = tf.cast(tf.reduce_sum(float_pos_mask), tf.float32)
    neg_weight = tf.cast(tf.reduce_sum(float_neg_mask), tf.float32)
    total = pos_weight + neg_weight
    float_pos_mask *= neg_weight * 1. / total
    float_neg_mask *= pos_weight * 1.1 / total
    float_mask = float_pos_mask + float_neg_mask
    return tf.reduce_sum(
        tf.nn.weighted_cross_entropy_with_logits(
            targets=labels,
            logits=logits,
            pos_weight=float_mask))  # / tf.cast(total, tf.float32)


def snemi_bce_loss(logits, labels):
    """Weighted CE from HED models for training on BSDS."""
    logit_shape = np.array(logits.get_shape().as_list())
    label_shape = np.array(labels.get_shape().as_list())
    assert np.all(logit_shape == label_shape), 'Logit/label shape mismatch'

    # Select dispositive 0s and values > gamma
    pos_mask = tf.greater(labels, 0.)
    neg_mask = tf.equal(labels, 0.)
    float_pos_mask = tf.cast(pos_mask, tf.float32)
    float_neg_mask = tf.cast(neg_mask, tf.float32)
    pos_weight = tf.cast(tf.reduce_sum(float_pos_mask), tf.float32)
    neg_weight = tf.cast(tf.reduce_sum(float_neg_mask), tf.float32)
    total = pos_weight + neg_weight
    float_pos_mask *= 1. * neg_weight / total
    float_neg_mask *= 1.1 * pos_weight / total
    float_mask = float_pos_mask + float_neg_mask
    return tf.reduce_sum(
        tf.nn.sigmoid_cross_entropy_with_logits(
            labels=labels,
            logits=logits) * float_mask)  # / tf.cast(total, tf.float32)


def f1_score(logits, labels, threshold_labels=False, eps=1e-4):
    """Calculate f1 score."""
    eps = 1e-4
    # logits = tf.squeeze(logits, -1)
    logits = tf.cast(logits, tf.float32)
    if len(logits.get_shape()) == 4 and int(logits.get_shape()[-1]) > 1:
        predicted = tf.cast(tf.argmax(logits, axis=-1), tf.float32)
    else:
        predicted = tf.squeeze(
            tf.cast(tf.round(tf.sigmoid(logits)), tf.float32))
    labels = tf.squeeze(labels, -1)
    if threshold_labels:
        labels = tf.cast(tf.greater(labels, threshold_labels), tf.float32)
    actual = tf.cast(labels, tf.float32)
    TP = tf.cast(tf.count_nonzero(predicted * actual), tf.float32)
    # TN = tf.cast(
    #     tf.count_nonzero((predicted - 1) * (actual - 1)), tf.float32)
    FP = tf.cast(tf.count_nonzero(predicted * (actual - 1)), tf.float32)
    FN = tf.cast(tf.count_nonzero((predicted - 1) * actual), tf.float32)
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    return (2 * precision * recall) / (precision + recall + eps)


def pearson_dissimilarity(labels, logits, REDUCE, eps_1=1e-4, eps_2=1e-12):
    """Calculate pearson diss. loss."""
    pred = logits
    x_shape = pred.get_shape().as_list()
    y_shape = labels.get_shape().as_list()
    if x_shape[-1] == 1 and len(x_shape) == 2:
        # If calculating score across exemplars
        pred = tf.squeeze(pred, -1)
        x_shape = [x_shape[0]]
        labels = tf.squeeze(labels, -1)
        y_shape = [y_shape[0]]

    if len(x_shape) > 2:
        # Reshape tensors
        x1_flat = tf.contrib.layers.flatten(pred)
    else:
        # Squeeze off singletons to make x1/x2 consistent
        x1_flat = tf.squeeze(pred, -1)
    if len(y_shape) > 2:
        x2_flat = tf.contrib.layers.flatten(labels)
    else:
        x2_flat = tf.squeeze(labels, -1)
    x1_mean = tf.reduce_mean(x1_flat, keep_dims=True, axis=[-1]) + eps_1
    x2_mean = tf.reduce_mean(x2_flat, keep_dims=True, axis=[-1]) + eps_1

    x1_flat_normed = x1_flat - x1_mean
    x2_flat_normed = x2_flat - x2_mean

    count = int(x2_flat.get_shape()[-1])
    cov = tf.div(
        tf.reduce_sum(
            tf.multiply(
                x1_flat_normed, x2_flat_normed),
            -1),
        count)
    x1_std = tf.sqrt(
        tf.div(
            tf.reduce_sum(
                tf.square(x1_flat - x1_mean),
                -1),
            count))
    x2_std = tf.sqrt(
        tf.div(
            tf.reduce_sum(
                tf.square(x2_flat - x2_mean),
                -1),
            count))
    corr = cov / (tf.multiply(x1_std, x2_std) + eps_2)
    if REDUCE is not None:
        corr = REDUCE(corr)
    return 1 - corr
