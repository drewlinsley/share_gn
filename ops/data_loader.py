import math
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from ops import rgb_lab_formulation as Conv_img


def rgb_to_lab(img):
    raw_input = tf.image.convert_image_dtype(img, dtype=tf.float32)
    # raw_input.set_shape([None, None, 3])

    # convert to lab-space image {L, a, b}
    lab = Conv_img.rgb_to_lab(raw_input)
    L_chan, a_chan, b_chan = Conv_img.preprocess_lab(lab)
    # lab = Conv_img.deprocess_lab(L_chan, a_chan, b_chan)

    # Convert from [-1, 1] to [0, 1]
    lab = tf.stack([L_chan, a_chan, b_chan], -1)
    return lab


def lab_to_rgb(lab):
    true_image = Conv_img.lab_to_rgb(lab)
    true_image = tf.image.convert_image_dtype(true_image, dtype=tf.uint8, saturate=True)
    return true_image


def eraser(input_img, p=1., s_l=0.02, s_h=0.4, r_1=0.3, r_2=1/0.3, v_l=0, v_h=255, pixel_level=False):

    img_h, img_w, img_c = input_img.get_shape().as_list()

    p_1 = np.random.rand()

    if p_1 > p:
        return input_img

    while True:
        s = np.random.uniform(s_l, s_h) * img_h * img_w
        r = np.random.uniform(r_1, r_2)
        w = int(np.sqrt(s / r))
        h = int(np.sqrt(s * r))
        left = np.random.randint(0, img_w)
        top = np.random.randint(0, img_h)

        if left + w <= img_w and top + h <= img_h:
            break

    if pixel_level:
        if input_img.ndim == 3:
            c = np.random.uniform(v_l, v_h, (h, w, img_c))
        if input_img.ndim == 2:
            c = np.random.uniform(v_l, v_h, (h, w))
    else:
        c = np.random.uniform(v_l, v_h)

    # Create mask and 0 target area
    mask = np.ones((img_h, img_w, img_c), dtype=np.float32)
    mask[top:top + h, left:left + w] = 0.
    input_img *= mask
    
    # Then create a bias
    bias = (1 - mask) * c
    input_img += bias

    # input_img[top:top + h, left:left + w] = c

    return input_img, mask[..., 0][..., None]


def center_eraser(input_img, p=1., s_l=0.02, s_h=0.4, r_1=0.3, r_2=1/0.3, v_l=0, v_h=255, pixel_level=True, half_size=16):

    img_h, img_w, img_c = input_img.get_shape().as_list()

    mid_h, mid_w = img_h // 2, img_w // 2
    left = mid_w - half_size
    top = mid_h - half_size

    if pixel_level:

        r = np.random.uniform(r_1, r_2)
        if len(input_img.get_shape().as_list()) == 3:
            c = np.random.uniform(v_l, v_h, (img_h, img_w, img_c))
        if len(input_img.get_shape().as_list()) == 2:
            c = np.random.uniform(v_l, v_h, (img_h, img_w))
    else:
        c = np.random.uniform(v_l, v_h)

    # Create mask and 0 target area
    mask = np.ones((img_h, img_w, img_c), dtype=np.float32)
    mask[top:top + half_size * 2, left:left + half_size * 2] = 0.
    v0 = tf.random.shuffle(input_img[..., 0])
    v1 = tf.random.shuffle(input_img[..., 1])
    v2 = tf.random.shuffle(input_img[..., 2])
    input_patch = tf.stack((v0, v1, v2), -1)

    # Create mask and 0 target area
    input_img *= mask

    # Then create a bias
    bias = (1 - mask) * input_patch
    input_img += bias

    # input_img[top:top + h, left:left + w] = c

    return input_img, mask[..., 0][..., None]


def repeat_elements(x, rep, axis):
    '''Repeats the elements of a tensor along an axis, like np.repeat
    If x has shape (s1, s2, s3) and axis=1, the output
    will have shape (s1, s2 * rep, s3)
    This function is taken from keras backend
    '''
    x_shape = x.get_shape().as_list()
    splits = tf.split(axis=axis, num_or_size_splits=x_shape[axis], value=x)
    x_rep = [s for s in splits for i in range(rep)]
    return tf.concat(axis=axis, values=x_rep)


def resize_image_label(im, model_input_image_size, f='bilinear'):
    """Resize images filter."""
    if f == 'bilinear':
        res_fun = tf.image.resize_images
    elif f == 'nearest':
        res_fun = tf.image.resize_nearest_neighbor
    elif f == 'bicubic':
        res_fun = tf.image.resize_bicubic
    elif f == 'area':
        res_fun = tf.image.resize_area
    else:
        raise NotImplementedError
    if len(im.get_shape()) > 3:
        # Spatiotemporal image set.
        nt = int(im.get_shape()[0])
        sims = tf.split(im, nt)
        for idx in range(len(sims)):
            # im = tf.squeeze(sims[idx])
            im = sims[idx]
            sims[idx] = res_fun(
                im,
                model_input_image_size,
                align_corners=True)
        im = tf.squeeze(tf.stack(sims))
        if len(im.get_shape()) < 4:
            im = tf.expand_dims(im, axis=-1)
    else:
        im = res_fun(
            tf.expand_dims(im, axis=0),
            model_input_image_size,
            align_corners=True)
        im = tf.squeeze(im, axis=0)
    return im


def crop_image_label(image, label, size, crop='random'):
    """Apply a crop to both image and label."""
    image_shape = image.get_shape().as_list()
    if len(size) > 2:
        size[-1] = image_shape[-1] + int(label.get_shape()[-1])
    if crop == 'random':
        combined = tf.concat([image, label], axis=-1)
        combined_crop = tf.random_crop(combined, size)
        image = combined_crop[:, :, :image_shape[-1]]
        label = combined_crop[:, :, image_shape[-1]:]
        return image, label
    else:
        # Center crop
        image = tf.image.resize_image_with_crop_or_pad(
            image,
            size[0],
            size[1])
        label = tf.image.resize_image_with_crop_or_pad(
            label,
            size[0],
            size[1])
        return image, label


def lr_flip_image_label(image, label):
    """Apply a crop to both image and label."""
    image_shape = [int(x) for x in image.get_shape()]
    combined = tf.concat([image, label], axis=-1)
    combined_crop = tf.image.random_flip_left_right(combined)
    image = combined_crop[:, :, :image_shape[-1]]
    label = combined_crop[:, :, image_shape[-1]:]
    return image, label


def ud_flip_image_label(image, label):
    """Apply a crop to both image and label."""
    image_shape = [int(x) for x in image.get_shape()]
    combined = tf.concat([image, label], axis=-1)
    combined_crop = tf.image.random_flip_up_down(combined)
    image = combined_crop[:, :, :image_shape[-1]]
    label = combined_crop[:, :, image_shape[-1]:]
    return image, label


def random_crop(image, model_input_image_size):
    """Wrapper for random cropping."""
    im_size = image.get_shape().as_list()
    if len(im_size) == 3:
        return tf.random_crop(
            image, model_input_image_size[:-1] + [im_size[-1]])
    elif len(im_size) == 4:
        if im_size[-1] > 1:
            raise NotImplementedError
        crop_size = model_input_image_size[:2] + [im_size[0]]
        trans_image = tf.transpose(tf.squeeze(image), [1, 2, 0])
        crop_image = tf.expand_dims(
            tf.transpose(
                tf.random_crop(trans_image, crop_size),
                [2, 0, 1]), axis=-1)
        return crop_image
    else:
        raise NotImplementedError


def center_crop(image, model_input_image_size):
    """Wrapper for center crop."""
    im_size = image.get_shape().as_list()
    target_height = model_input_image_size[0]
    target_width = model_input_image_size[1]
    if len(im_size) == 3:
        return tf.image.resize_image_with_crop_or_pad(
            image,
            target_height=target_height,
            target_width=target_width)
    elif len(im_size) == 4:
        time_split_image = tf.split(image, im_size[0], axis=0)
        crops = []
        for idx in range(len(time_split_image)):
            it_crop = tf.image.resize_image_with_crop_or_pad(
                tf.squeeze(time_split_image[idx], axis=0),
                target_height=target_height,
                target_width=target_width)
            crops += [tf.expand_dims(it_crop, axis=0)]
        return tf.concat(crops, axis=0)
    else:
        raise NotImplementedError


def image_flip(image, direction):
    """Wrapper for image flips."""
    im_size = image.get_shape().as_list()
    if direction == 'left_right':
        flip_function = tf.image.random_flip_left_right
    elif direction == 'up_down':
        flip_function = tf.image.random_flip_up_down
    else:
        raise NotImplementedError

    if len(im_size) == 3:
        return flip_function(image)
    elif len(im_size) == 4:
        if im_size[-1] > 1:
            raise NotImplementedError
        trans_image = tf.transpose(tf.squeeze(image), [1, 2, 0])
        flip_image = tf.expand_dims(
            tf.transpose(flip_function(trans_image), [2, 0, 1]), axis=-1)
        return flip_image
    else:
        raise NotImplementedError


def tf_blur(image, kernel_size, name, mean=0., sigma=1., normalize=False):
    """Construct a gaussian kernel with sigma=sigma and blur image."""
    d = tf.contrib.distributions.Normal(mean, sigma)
    vals = d.prob(tf.range(start=-kernel_size, limit=kernel_size + 1, dtype=tf.float32))
    gauss_kernel = tf.einsum('i,j->ij', vals, vals)  # Outer product
    gauss_kernel = tf.expand_dims(tf.expand_dims(gauss_kernel, axis=2), axis=3)
    with tf.variable_scope('blur_kernel', reuse=tf.AUTO_REUSE):
        gauss_kernel = tf.get_variable(
            name=name,
            initializer=gauss_kernel / tf.reduce_sum(gauss_kernel),
            trainable=False)
    im = tf.nn.conv2d(
        input=tf.expand_dims(image, 0),
        filter=gauss_kernel,
        strides=[1, 1, 1, 1],
        padding='SAME',
        name='%s_conv' % name)
    im = tf.squeeze(im, 0)
    if normalize:
        im = tf.maximum(im, 0)
        im_scale = tf.reduce_max(image) / tf.reduce_max(im)
        im *= im_scale
    return im


def cube_plus_crop(image, model_input_image_size, seed=None):
    """Get vert and horiz random crop boxes then randomly select one."""
    vol_size = image.get_shape().as_list()
    # crop_locations = [1050, 2050]

    # Concat volume and label into a single volume for cropping
    comb_size = image.get_shape().as_list()
    crop_size = [comb_size[0]] + model_input_image_size + [comb_size[-1]]
    crop_size = [comb_size[0]] + model_input_image_size + [comb_size[-1]]
    with ops.name_scope(
            'color_crop', 'random_crop', [image, crop_size]) as name:
        combined_volume = ops.convert_to_tensor(image, name='value')
        crop_size = ops.convert_to_tensor(
            crop_size, dtype=dtypes.int32, name='size')
        vol_shape = array_ops.shape(combined_volume)
        control_flow_ops.Assert(
            math_ops.reduce_all(vol_shape >= crop_size),
            ['Need vol_shape >= vol_size, got ', vol_shape, crop_size],
            summarize=1000)
        limit = vol_shape - crop_size + 1
        offset = tf.random_uniform(
            array_ops.shape(vol_shape),
            dtype=crop_size.dtype,
            maxval=crop_size.dtype.max,
            seed=seed) % limit
        # offset_2 = tf.random_uniform(
        #     array_ops.shape(vol_shape),
        #     dtype=crop_size.dtype,
        #     maxval=crop_size.dtype.max,
        #     seed=seed) % limit

        cropped_combined = array_ops.slice(
            combined_volume, offset, crop_size, name=name)
    cropped_volume = cropped_combined[:, :, :, :vol_size[-1]]
    cropped_label = cropped_combined[:, :, :, vol_size[-1]:]
    return cropped_volume, cropped_label


def image_augmentations(
        image,
        data_augmentations,
        model_input_image_size,
        label=None):
    """Coordinating image augmentations for both image and heatmap."""
    if image.get_shape() == None:
        im_size = model_input_image_size
    else:
        im_size = image.get_shape().as_list()
    im_size_check = True  # np.any(
        # np.less_equal(
        #     model_input_image_size[:2],
        #     im_size[:2]))
    if data_augmentations is not None:
        for aug in data_augmentations:
            # Pixel/image-level augmentations
            if aug == 'image_float32':
                image = tf.cast(image, tf.float32)
            if aug == 'label_float32':
                label = tf.cast(label, tf.float32)
            if aug == 'bfloat16':
                image = tf.cast(image, tf.bfloat16)
            if aug == 'singleton':
                image = tf.expand_dims(image, axis=-1)
                print('Adding singleton dimension to image.')
            if aug == 'sgl_label' or aug == 'singleton_label':
                label = tf.expand_dims(label, axis=-1)
                print('Adding singleton dimension to label.')
            if aug == 'coco_labels':
                label = tf.nn.relu(label - 91)
            if aug == 'contrastive_loss':
                label = tf.stack(
                    [tf.ones_like(label), tf.zeros_like(label)], -1)
            if aug == 'bsds_normalize':
                data = np.load(
                    '/media/data_cifs/image_datasets/BSDS500/images/train/file_paths.npz')
                mean = data['mean'].squeeze(0)
                stds = data['stds'].squeeze(0)
                image = (image - mean) / stds
            if aug == 'bsds_crop' and im_size_check:
                assert len(image.get_shape()) == 3, '4D not implemented yet.'
                scale_choices = tf.convert_to_tensor(
                    # [1. / 2., 1.1 / 2., 1.2 / 2.])
                    [1., 1, 1.1, 1.2])
                samples = tf.multinomial(
                    tf.log([tf.ones_like(scale_choices)]), 1)
                image_shape = image.get_shape().as_list()
                scale = scale_choices[tf.cast(samples[0][0], tf.int32)]
                scale_tf = tf.cast(
                    tf.round(
                        np.asarray(
                            image_shape[:2]).astype(
                            np.float32) * scale),
                    tf.int32)
                combined = tf.concat([image, label], axis=-1)
                combo_shape = combined.get_shape().as_list()
                combined_resize = tf.squeeze(
                    tf.image.resize_nearest_neighbor(
                        tf.expand_dims(combined, axis=0),
                        scale_tf,
                        align_corners=True),
                    axis=0)
                combined_crop = tf.random_crop(
                    combined_resize,
                    tf.concat(
                        [model_input_image_size[:2], [combo_shape[-1]]], 0))
                image = combined_crop[:, :, :image_shape[-1]]
                label = combined_crop[:, :, image_shape[-1]:]
                image.set_shape(model_input_image_size)
                label.set_shape(
                    model_input_image_size[:2] + [
                        combo_shape[-1] - model_input_image_size[-1]])
                print('Applying BSDS crop.')
            if aug == 'hed_resize' and im_size_check:
                assert len(image.get_shape()) == 3, '4D not implemented yet.'
                scale_choices = tf.convert_to_tensor(
                    # [1. / 2., 1.1 / 2., 1.2 / 2.])
                    np.arange(1, 1.51, 0.1))  # 0.7, 1.5
                samples = tf.multinomial(
                    tf.log([tf.ones_like(scale_choices)]), 1)
                image_shape = image.get_shape().as_list()
                scale = scale_choices[tf.cast(samples[0][0], tf.int32)]
                scale_tf = tf.cast(
                    tf.round(
                        np.asarray(
                            image_shape[:2]).astype(
                            np.float32) * scale),
                    tf.int32)
                combined = tf.concat([image, label], axis=-1)
                combo_shape = combined.get_shape().as_list()
                combined_resize = tf.squeeze(
                    tf.image.resize_bilinear(
                        tf.expand_dims(combined, axis=0),
                        scale_tf,
                        align_corners=True),
                    axis=0)
                print('Applying HED resize.')
            if aug == 'uint8_rescale':
                image = tf.cast(image, tf.float32) / 255.
                print('Applying uint8 rescale to the image.')
            if aug == 'cube_plus_rescale':
                image = tf.cast(image, tf.float32) / 13273.
                print('Applying uint8 rescale to the image.')
            if aug == 'uint8_rescale_label':
                label = tf.cast(label, tf.float32) / 255.
                print('Applying uint8 rescale to the label.')
            if aug == 'uint8_rescale_-1_1':
                image = 2 * (tf.cast(image, tf.float32) / 255.) - 1
                print('Applying uint8 rescale.')
            if aug == 'image_to_bgr':
                image = tf.stack(
                    [image[..., 2], image[..., 1], image[..., 0]], axis=-1)
            if aug == 'pascal_normalize':
                image = image - [123.68, 116.78, 103.94]
            if aug == 'ilsvrc12_normalize':
                MEAN_RGB = [0.485 * 255, 0.456 * 255, 0.406 * 255]
                STDDEV_RGB = [0.229 * 255, 0.224 * 255, 0.225 * 255]
                image = (image - MEAN_RGB) / STDDEV_RGB
            if aug == 'fixed_mask':  # BSDS imputation
                mask_size = np.array(image.get_shape().as_list())
                mask_size /= 2
                mask_size = mask_size.astype(int).tolist()
                image_size = image.get_shape().as_list()
                h_offset, w_offset, _ = mask_size
                h_mask = tf.concat(
                    [
                        tf.ones([h_offset, image_size[1], 3]),
                        tf.zeros([mask_size[0], image_size[1], 3]),
                        tf.ones([image_size[0] - (h_offset + mask_size[0]), image_size[1], 3])], 0)
                w_mask = tf.concat(
                    [
                        tf.ones([image_size[0], w_offset, 3]),
                        tf.zeros([image_size[0], mask_size[1], 3]),
                        tf.ones([image_size[0], image_size[1] - (w_offset + mask_size[1]), 3])], 1)
                mask = tf.cast(tf.greater(h_mask + w_mask, 0), tf.float32)
                image *= mask
                mask = tf.expand_dims(mask[..., 0], -1)
                label = -1 * (tf.cast(tf.equal(mask, 0), tf.float32) * -1) * label
                label += (mask * -10. )  # (tf.cast(tf.equal(mask, 0), tf.float32) * -10.)
            if aug == 'random_mask':  # BSDS imputation
                mask_size = np.array(image.get_shape().as_list())
                mask_size /= 2
                mask_size = mask_size.astype(int)
                image_size = image.get_shape().as_list()
                h_offset = tf.random_uniform(
                    [],
                    dtype=tf.int32,
                    maxval=image_size[0] - mask_size[0])
                w_offset = tf.random_uniform(
                    [],
                    dtype=tf.int32,
                    maxval=image_size[1] - mask_size[1])

                # Mask a corner
                h_mask = tf.concat(
                    [
                        tf.ones([h_offset, image_size[1], 3]),
                        tf.zeros([mask_size[0], image_size[1], 3]),
                        tf.ones([image_size[0] - (h_offset + mask_size[0]), image_size[1], 3])], 0)
                w_mask = tf.concat(
                    [
                        tf.ones([image_size[0], w_offset, 3]),
                        tf.zeros([image_size[0], mask_size[1], 3]),
                        tf.ones([image_size[0], image_size[1] - (w_offset + mask_size[1]), 3])], 1)
                mask = tf.cast(tf.greater(h_mask + w_mask, 0), tf.float32)
                image *= mask
                mask = tf.expand_dims(mask[..., 0], -1)
                label = -1 * (tf.cast(tf.equal(mask, 0), tf.float32) * -1) * label
                label += (mask * -10. )  # (tf.cast(tf.equal(mask, 0), tf.float32) * -10.)
            if aug == 'random_contrast':
                assert len(image.get_shape()) == 3, '4D not implemented yet.'
                image = tf.image.random_contrast(image, lower=0.2, upper=1.8)
                print('Applying random contrast.')
            if aug == 'random_brightness':
                assert len(image.get_shape()) == 3, '4D not implemented yet.'
                image = tf.image.random_brightness(image, max_delta=63.)
                print('Applying random brightness.')
            if aug == 'grayscale' and im_size_check:
                # image = tf.image.rgb_to_grayscale(image)
                if len(image.get_shape().as_list()) == 2:
                    image = tf.expand_dims(image, axis=-1)
                else:
                    image = tf.expand_dims(image[..., 0], axis=-1)
                print('Converting to grayscale.')
            if aug == 'rgb2gray' and im_size_check:
                image = tf.image.rgb_to_grayscale(image)
                print('Converting rgb2gray.')
            if aug == 'clip_uint8' and im_size_check:
                image = tf.minimum(image, 255.)
                image = tf.maximum(image, 0.)
            if aug == 'cube_plus_crop':
                image = cube_plus_crop(image, model_input_image_size)
            # Affine augmentations
            if aug == 'rotate' and im_size_check:
                max_theta = 22.
                angle_rad = (max_theta / 180.) * math.pi
                angles = tf.random_uniform([], -angle_rad, angle_rad)
                transform = tf.contrib.image.angles_to_projective_transforms(
                    angles,
                    im_size[0],
                    im_size[1])
                image = tf.contrib.image.transform(
                    image,
                    tf.contrib.image.compose_transforms(transform),
                    interpolation='BILINEAR')  # or 'NEAREST'
                print('Applying random rotate.')
            if aug == 'rotate90' and im_size_check:
                image = tf.image.rot90(
                    image,
                    tf.random_uniform(
                        shape=[],
                        minval=0,
                        maxval=4,
                        dtype=tf.int32))
                print('Applying random 90 degree rotate.')
            if aug == 'rotate90_image_label' and im_size_check:
                concat = tf.image.rot90(
                    tf.concat([image, label], -1),
                    tf.random_uniform(
                        shape=[],
                        minval=0,
                        maxval=4,
                        dtype=tf.int32))
                image = concat[..., :im_size[-1]]
                label = concat[..., im_size[-1]:]
                print('Applying random 90 degree rotate to images and labels.')
            if aug == 'stack3d':
                image = tf.concat([image, image, image], axis=-1)
            if aug == 'rot_image_label' and im_size_check:
                max_theta = 30.
                angle_rad = (max_theta / 180.) * math.pi
                angles = tf.random_uniform([], -angle_rad, angle_rad)
                transform = tf.contrib.image.angles_to_projective_transforms(
                    angles,
                    im_size[0],
                    im_size[1])
                image = tf.contrib.image.transform(
                    image,
                    tf.contrib.image.compose_transforms(transform),
                    interpolation='BILINEAR')  # or 'NEAREST'
                label = tf.contrib.image.transform(
                    label,
                    tf.contrib.image.compose_transforms(transform),
                    interpolation='BILINEAR')  # or 'NEAREST'
                print('Applying random rotate.')
            if aug == 'random_scale_crop_image_label'\
                    and im_size_check:
                scale_choices = tf.convert_to_tensor(
                    [1., 1.04, 1.08, 1.12, 1.16])
                samples = tf.multinomial(
                    tf.log([tf.ones_like(scale_choices)]), 1)
                image_shape = image.get_shape().as_list()
                scale = scale_choices[tf.cast(samples[0][0], tf.int32)]
                scale_tf = tf.cast(
                    tf.round(
                        np.asarray(
                            model_input_image_size[:2]).astype(
                            np.float32) * scale),
                    tf.int32)
                combined = tf.concat([image, label], axis=-1)
                combo_shape = combined.get_shape().as_list()
                combined_resize = tf.squeeze(
                    tf.image.resize_bicubic(
                        tf.expand_dims(combined, axis=0),
                        scale_tf,
                        align_corners=True),
                    axis=0)
                combined_crop = tf.random_crop(
                    combined_resize, tf.concat(
                        [model_input_image_size[:2], [combo_shape[-1]]], 0))
                image = combined_crop[:, :, :image_shape[-1]]
                label = combined_crop[:, :, image_shape[-1]:]
                image.set_shape(model_input_image_size)
                label.set_shape(
                    model_input_image_size[:2] + [
                        combo_shape[-1] - model_input_image_size[-1]])
            if aug == 'rc_res' and im_size_check:
                image = random_crop(image, model_input_image_size)
                if len(model_input_image_size) > 2:
                    model_input_image_size = model_input_image_size[:2]
                ms = [x // 2 for x in model_input_image_size]
                image = resize_image_label(
                    im=image,
                    model_input_image_size=ms,
                    f='bicubic')
                print('Applying random crop and resize.')
            if aug == 'cc_res' and im_size_check:
                image = center_crop(image, model_input_image_size)
                if len(model_input_image_size) > 2:
                    model_input_image_size = model_input_image_size[:2]
                ms = [x // 2 for x in model_input_image_size]
                image = resize_image_label(
                    im=image,
                    model_input_image_size=ms,
                    f='bicubic')
                print('Applying center crop and resize.')
            if aug == 'random_crop' and im_size_check:
                image = random_crop(image, model_input_image_size)
                print('Applying random crop.')
            if aug == 'center_crop' and im_size_check:
                image = center_crop(image, model_input_image_size)
                print('Applying center crop.')
            if aug == 'rc_image_label' and im_size_check:
                assert len(image.get_shape()) == 3, '4D not implemented yet.'
                image, label = crop_image_label(
                    image=image,
                    label=label,
                    size=model_input_image_size,
                    crop='random')
            if aug == 'cc_image_label' and im_size_check:
                assert len(image.get_shape()) == 3, '4D not implemented yet.'
                image, label = crop_image_label(
                    image=image,
                    label=label,
                    size=model_input_image_size,
                    crop='center')
            if aug == 'resize' and im_size_check:
                if len(model_input_image_size) > 2:
                    model_input_image_size = model_input_image_size[:2]
                image = resize_image_label(
                    im=image,
                    model_input_image_size=model_input_image_size,
                    f='bicubic')
                print('Applying area resize.')
            if aug == 'jk_resize' and im_size_check:
                if len(model_input_image_size) > 2:
                    model_input_image_size = model_input_image_size[:2]
                image = tf.image.resize_image_with_crop_or_pad(
                    image,
                    model_input_image_size[0],
                    model_input_image_size[1])
                print('Applying area resize.')
            if aug == 'random_crop_and_res_cube_plus' and im_size_check:
                im_shape = image.get_shape().as_list()
                im_shape[0] /= 4
                im_shape[1] /= 4
                image = resize_image_label(
                    im=image,
                    model_input_image_size=im_shape[:2],
                    f='bicubic')
                image = random_crop(image, model_input_image_size)
            if aug == 'center_crop_and_res_cube_plus' and im_size_check:
                im_shape = image.get_shape().as_list()
                im_shape[0] /= 4
                im_shape[1] /= 4
                image = resize_image_label(
                    im=image,
                    model_input_image_size=im_shape[:2],
                    f='bicubic')
                image = center_crop(image, model_input_image_size)
            if aug == 'res_and_crop' and im_size_check:
                model_input_image_size_1 = np.asarray(
                    model_input_image_size[:2]) + 28
                image = resize_image_label(
                    im=image,
                    model_input_image_size=model_input_image_size_1,
                    f='area')
                image = center_crop(image, model_input_image_size)
                print('Applying area resize.')
            if aug == 'res_nn' and im_size_check:
                assert len(image.get_shape()) == 3, '4D not implemented yet.'
                if len(model_input_image_size) > 2:
                    model_input_image_size = model_input_image_size[:2]
                image = resize_image_label(
                    im=image,
                    model_input_image_size=model_input_image_size,
                    f='nearest')
                print('Applying nearest resize.')
            if aug == "flip_polarity":
                image = 255 - image
            if aug == 'res_image_label' and im_size_check:
                assert len(image.get_shape()) == 3, '4D not implemented yet.'
                if len(model_input_image_size) > 2:
                    model_input_image_size = model_input_image_size[:2]
                image = resize_image_label(
                    im=image,
                    model_input_image_size=model_input_image_size,
                    f='bicubic')
                label = resize_image_label(
                    im=label,
                    model_input_image_size=model_input_image_size,
                    f='bicubic')
                print('Applying bilinear resize.')
            if aug == 'res_nn_image_label' and im_size_check:
                assert len(image.get_shape()) == 3, '4D not implemented yet.'
                if len(model_input_image_size) > 2:
                    model_input_image_size = model_input_image_size[:2]
                image = resize_image_label(
                    im=image,
                    model_input_image_size=model_input_image_size,
                    f='nearest')
                label = resize_image_label(
                    im=label,
                    model_input_image_size=model_input_image_size,
                    f='nearest')
                print('Applying nearest resize.')
            if aug == 'left_right':
                image = image_flip(image, direction='left_right')
                print('Applying random flip left-right.')
            if aug == 'up_down':
                image = image_flip(image, direction='up_down')
                print('Applying random flip up-down.')
            if aug == 'lr_flip_image_label':
                assert len(image.get_shape()) == 3, '4D not implemented yet.'
                image, label = lr_flip_image_label(image, label)
            if aug == 'ud_flip_image_label':
                assert len(image.get_shape()) == 3, '4D not implemented yet.'
                image, label = ud_flip_image_label(image, label)
            if aug == 'gratings_modulate':
                # modulate = 2  # 10
                print("Warning: Gratings modulate is disabled.")
                # image //= modulate
                # offset = (255 / 2) - ((255 / modulate) / 2)
                # image += offset
            if aug == 'gratings_modulate_half':
                modulate = 2  # 10
                print("Warning: Gratings modulate is disabled.")
                image //= modulate
                # offset = (255 / 2) - ((255 / modulate) / 2)
                # image += offset
                image = image + (255 / 4)
            if aug == 'rotate_60':
                max_theta = 60.
                angle_rad = (max_theta / 180.) * math.pi
                angles = angle_rad  # tf.random_uniform([], -angle_rad, angle_rad)
                transform = tf.contrib.image.angles_to_projective_transforms(
                    angles,
                    im_size[0],
                    im_size[1])
                image = tf.contrib.image.transform(
                    image,
                    tf.contrib.image.compose_transforms(transform),
                    interpolation='BILINEAR')  # or 'NEAREST'
            if aug == 'gratings_modulate_06':
                modulate = 10 / (0.06 / 2 + 0.6)
                # modulate = 10
                print("Warning: Gratings modulate is disabled.")
                image //= modulate
                offset = (255 / 2) - ((255 / modulate) / 2)
                image += offset
            if aug == 'gratings_modulate_12':
                modulate = 10 / (0.12 / 2 + 0.6)
                print("Warning: Gratings modulate is disabled.")
                image //= modulate
                offset = (255 / 2) - ((255 / modulate) / 2)
                image += offset
            if aug == 'gratings_modulate_25':
                modulate = 10 / (0.25 / 2 + 0.6)
                print("Warning: Gratings modulate is disabled.")
                image //= modulate
                offset = (255 / 2) - ((255 / modulate) / 2)
                image += offset
            if aug == 'gratings_modulate_50':
                modulate = 10 / (0.50 / 2 + 0.6)
                print("Warning: Gratings modulate is disabled.")
                image //= modulate
                offset = (255 / 2) - ((255 / modulate) / 2)
                image += offset
            if aug == 'gratings_modulate_75':
                modulate = 10 / (0.75 / 2 + 0.6)
                print("Warning: Gratings modulate is disabled.")
                image //= modulate
                offset = (255 / 2) - ((255 / modulate) / 2)
                image += offset

            if aug == 'gratings_modulate_2':
                modulate = 2
                image //= modulate
                offset = (255 / 2) - ((255 / modulate) / 2)
                image += offset

            if aug == 'gaussian_noise':
                im_shape = image.get_shape().as_list()
                assert len(im_shape) == 3, '4D not implemented yet.'
                sigma = 1. / 10.
                mu = 0.
                image = image + tf.random_normal(
                    im_shape,
                    mean=mu,
                    stddev=sigma)
                print('Applying gaussian noise.')
            if aug == 'gaussian_noise_small':
                im_shape = image.get_shape().as_list()
                assert len(im_shape) == 3, '4D not implemented yet.'
                sigma = 1. / 20.
                mu = 0.
                image = image + tf.random_normal(
                    im_shape,
                    mean=mu,
                    stddev=sigma)
                print('Applying gaussian noise.')
            if aug == 'mixup':
                raise RuntimeError('Mixup not properly implemented yet.')
                alpha = 0.4
                dist = tf.distributions.Beta(alpha, alpha)
                image = image * dist + (1 - dist) * tf.roll(image, 0, 1)
                label = label * dist + (1 - dist) * tf.roll(label, 0, 1)
            if aug == 'jigsaw_ecrf':
                hh, ww, ic = image.get_shape().as_list()[:3]
                hh, ww, lc = label.get_shape().as_list()[:3]
                box_size = 32  # 320 // 14 -- closest 16 or 20
                pad = [[0,0],[0,0]]
                p = box_size
                patches = tf.space_to_batch_nd(tf.concat([image, label], -1)[None], [box_size, box_size], pad)
                patches = tf.split(patches,p * p,0)
                patches = tf.stack(patches, 3)
                patches = tf.reshape(patches, [(hh // p) ** 2, p, p, ic + lc])

                # Process patches here
                patches = tf.random.shuffle(patches, 0)

                # Using patches here to reconstruct
                patches_proc = tf.reshape(patches, [1, hh // p, hh // p, p * p, ic + lc])
                patches_proc = tf.split(patches_proc, p * p, 3)
                patches_proc = tf.stack(patches_proc, axis=0)
                patches_proc = tf.reshape(patches_proc, [p * p, hh // p, hh // p, ic + lc])
                image = tf.batch_to_space_nd(patches_proc, [p, p], pad)
                image = tf.reshape(image, [hh, ww, ic + lc])
                label = image[..., -1][..., None]
                image = image[..., :3]
            if aug == 'jigsaw_ecrf_big':
                hh, ww, ic = image.get_shape().as_list()[:3]
                hh, ww, lc = label.get_shape().as_list()[:3]
                box_size = 80  # 320 // 14 -- closest 16 or 20
                pad = [[0,0],[0,0]]
                p = box_size
                patches = tf.space_to_batch_nd(tf.concat([image, label], -1)[None], [box_size, box_size], pad)
                patches = tf.split(patches,p * p,0)
                patches = tf.stack(patches, 3)
                patches = tf.reshape(patches, [(hh // p) ** 2, p, p, ic + lc])

                # Process patches here
                patches = tf.random.shuffle(patches, 0)

                # Using patches here to reconstruct
                patches_proc = tf.reshape(patches, [1, hh // p, hh // p, p * p, ic + lc])
                patches_proc = tf.split(patches_proc, p * p, 3)
                patches_proc = tf.stack(patches_proc, axis=0)
                patches_proc = tf.reshape(patches_proc, [p * p, hh // p, hh // p, ic + lc])
                image = tf.batch_to_space_nd(patches_proc, [p, p], pad)
                image = tf.reshape(image, [hh, ww, ic + lc])
                label = image[..., -1][..., None]
                image = image[..., :3]
            if aug == 'jigsaw_crf':
                hh, ww, ic = image.get_shape().as_list()[:3]
                hh, ww, lc = label.get_shape().as_list()[:3]
                box_size = 16  # 320 // 14 -- closest 16 or 20
                pad = [[0,0],[0,0]]
                p = box_size
                patches = tf.space_to_batch_nd(tf.concat([image, label], -1)[None], [box_size, box_size], pad)
                patches = tf.split(patches,p * p,0)
                patches = tf.stack(patches, 3)
                patches = tf.reshape(patches, [(hh // p) ** 2, p, p, ic + lc])

                # Process patches here
                patches = tf.random.shuffle(patches, 0)

                # Using patches here to reconstruct
                patches_proc = tf.reshape(patches, [1, hh // p, hh // p, p * p, ic + lc])
                patches_proc = tf.split(patches_proc, p * p, 3)
                patches_proc = tf.stack(patches_proc, axis=0)
                patches_proc = tf.reshape(patches_proc, [p * p, hh // p, hh // p, ic + lc])
                image = tf.batch_to_space_nd(patches_proc, [p, p], pad)
                image = tf.reshape(image, [hh, ww, ic + lc])
                label = image[..., -1][..., None]
                image = image[..., :3]
            if aug == 'occlusion':
                image, mask = eraser(image)
                label *= (1 - mask)  # tf.cast(tf.greater(mask, 0), tf.float32)
                label -= mask  # tf.cast(tf.equal(mask, 0), tf.float32)  # Set background to -1 for excluding from loss
            if aug == 'center_occlusion':
                image, mask = center_eraser(image)
                label *= (1 - mask)  # tf.cast(tf.greater(mask, 0), tf.float32)
                label -= mask  # tf.cast(tf.equal(mask, 0), tf.float32)  # Set background to -1 for excluding from loss
            if aug == 'hed_brightness':
                image = tf.image.random_brightness(image, 63)
            if aug == 'hed_contrast':
                image = tf.image.random_contrast(image, lower=0.6, upper=1.3)  # 0.4 1.5
            if aug == 'blur_labels':
                label = tf_blur(
                    image=label,
                    kernel_size=3,  # extent
                    name='label_blur',
                    normalize=True,
                    sigma=1.)
            if aug == 'calculate_rate_time_crop':
                im_shape = image.get_shape().as_list()
                minval = im_shape[0] // 3
                time_crop = tf.random_uniform(
                    [],
                    minval=minval,
                    maxval=im_shape[0],
                    dtype=tf.int32)

                # For now always pull from the beginning
                indices = tf.range(0, time_crop, dtype=tf.int32)
                selected_image = tf.gather(image, indices)
                padded_image = tf.zeros(
                    [im_shape[0] - time_crop] + im_shape[1:],
                    dtype=selected_image.dtype)

                # Randomly concatenate pad to front or back
                image = tf.cond(
                    pred=tf.greater(
                        tf.random_uniform(
                            [],
                            minval=0,
                            maxval=1,
                            dtype=tf.float32),
                        0.5),
                    true_fn=lambda: tf.concat(
                        [selected_image, padded_image], axis=0),
                    false_fn=lambda: tf.concat(
                        [padded_image, selected_image], axis=0)
                )
                image.set_shape(im_shape)

                # Convert label to rate
                label = label / im_shape[0]
            if aug == 'calculate_rate':
                label = label / image.get_shape().as_list()[0]
                print('Applying rate transformation.')
            if aug == 'threshold':
                image = tf.cast(tf.greater(image, 0.1), tf.float32)
                print('Applying threshold.')
            if aug == 'nonzero_label':
                label = tf.cast(tf.greater(label, 0.2), tf.float32)
                print('Applying threshold.')
            if aug == 'zero_one':
                image = tf.minimum(tf.maximum(image, 0.), 1.)
                print('Applying threshold.')
            if aug == 'timestep_duplication':
                image = tf.stack([image for iid in range(7)])
                print('Applying timestep duplication.')
            if aug == 'per_image_standardization':
                image = tf.image.per_image_standardization(image)
                print('Applying per-image zscore.')
            if aug == 'flip_image_polarity':
                image = tf.abs(image - 1.)
            if aug == 'flip_label_polarity':
                label = tf.abs(label - 1.)
            if aug == 'NCHW':
                image = tf.transpose(image, (2, 0, 1))
            if aug == 'bfloat16_image':
                image = tf.cast(image, tf.bfloat16)
            if aug == 'bfloat16_label':
                label = tf.cast(label, tf.bfloat16)
            if aug == 'hfloat16_image':
                image = tf.cast(image, tf.float16)
            if aug == 'hfloat16_label':
                label = tf.cast(label, tf.float16)
            if aug == 'threshold_label':
                label = tf.cast(tf.greater(label, 0.999), tf.float32)
                print('Applying threshold of 0.999 to the label.')
    # else:
    #     assert len(image.get_shape()) == 3, '4D not implemented yet.'
    #     image = tf.image.resize_image_with_crop_or_pad(
    #         image, model_input_image_size[0], model_input_image_size[1])
    return image, label


def decode_data(features, reader_settings):
    """Decode data from TFrecords."""
    if features.dtype == tf.string:
        return tf.decode_raw(
            features,
            reader_settings)
    else:
        return tf.cast(
            features,
            reader_settings)


def read_and_decode(
        filename_queue,
        model_input_image_size,
        tf_dict,
        tf_reader_settings,
        data_augmentations,
        number_of_files,
        aux=None,
        resize_output=None):
    """Read and decode tensors from tf_records and apply augmentations."""
    reader = tf.TFRecordReader()

    # Switch between single/multi-file reading
    if number_of_files == 1:
        _, serialized_example = reader.read(filename_queue)
        features = tf.parse_single_example(
            serialized_example,
            features=tf_dict)
    else:
        _, serialized_examples = reader.read_up_to(
            filename_queue,
            num_records=number_of_files)
        features = tf.parse_example(
            serialized_examples,
            features=tf_dict)

    # Handle decoding of each element
    image = decode_data(
        features=features['image'],
        reader_settings=tf_reader_settings['image']['dtype'])
    label = decode_data(
        features=features['label'],
        reader_settings=tf_reader_settings['label']['dtype'])

    # Reshape each element
    if 'height' in tf_dict.keys():
        # Assume variable height width. Make a crop. Only for per-pixel atm.
        height = decode_data(
            features=features['height'],
            reader_settings=tf_reader_settings['height']['dtype'])
        width = decode_data(
            features=features['width'],
            reader_settings=tf_reader_settings['width']['dtype'])
        if 'coco_preproc' in data_augmentations:
            print('Warning: Interchanging height/width for COCO.')
            oheight = height
            height = width
            width = oheight
        image = tf.reshape(
            image, [height, width, tf_reader_settings['image']['reshape'][-1]])
        label = tf.cast(label, image.dtype)
        label = tf.reshape(
            label, [height, width, tf_reader_settings['label']['reshape'][-1]])
        if 'coco_preproc' in data_augmentations:
            image, label = crop_image_label(
                image=image,
                label=label,
                size=np.copy(tf_reader_settings['image']['reshape']).tolist(),
                crop='center')
    else:
        image = tf.reshape(image, tf_reader_settings['image']['reshape'])
        if tf_reader_settings['label']['reshape'] is not None:
            label = tf.reshape(label, tf_reader_settings['label']['reshape'])

    if image.dtype == tf.float64:
        print('Forcing float64 image to float32.')
        image = tf.cast(image, tf.float32)
    if label.dtype == tf.float64:
        print('Forcing float64 label to float32.')
        label = tf.cast(label, tf.float32)

    if aux is not None:
        aux_data = decode_data(
            features=features[aux.keys()[0]],
            reader_settings=tf_reader_settings[aux.keys()[0]]['dtype'])
        aux = tf.reshape(
            aux_data, tf_reader_settings[aux.keys()[0]]['reshape'])
    else:
        aux = tf.constant(0)

    # Preprocess images and heatmaps
    if len(model_input_image_size) == 3:
        # 2D image augmentations
        image, label = image_augmentations(
            image=image,
            label=label,
            model_input_image_size=model_input_image_size,
            data_augmentations=data_augmentations)
        if resize_output is not None:
            # Resize labels after augmentations
            if isinstance(resize_output, dict):
                if resize_output.keys()[0] == 'resize':
                    label = resize_image_label(
                        im=label,
                        model_input_image_size=resize_output,
                        f='nearest')
                elif resize_output.keys()[0] == 'pool':
                    label = tf.expand_dims(label, axis=0)
                    label = tf.nn.max_pool(
                        value=label,
                        ksize=resize_output['pool']['kernel'],
                        strides=resize_output['pool']['stride'],
                        padding='SAME')
                    label = tf.squeeze(label, axis=0)
                else:
                    raise NotImplementedError(resize_output.keys()[0])
            else:
                label = resize_image_label(
                    im=label,
                    model_input_image_size=resize_output,
                    f='nearest')
    elif len(model_input_image_size) == 4:
        # 3D image augmentations.
        # TODO: optimize 3D augmentations with c++. This is slow.
        split_images = tf.split(
            image,
            model_input_image_size[0],
            axis=0)
        split_images = [tf.squeeze(im, axis=0) for im in split_images]
        images, labels = [], []
        if np.any(['label' in x for x in data_augmentations if x is not None]):
            split_labels = tf.split(
                label,
                model_input_image_size[0],
                axis=0)
            split_labels = [tf.squeeze(lab, axis=0) for lab in split_labels]
            for im, lab in zip(split_images, split_labels):
                it_im, it_lab = image_augmentations(
                    image=im,
                    label=lab,
                    model_input_image_size=model_input_image_size[1:],
                    data_augmentations=data_augmentations)
                if resize_output is not None:
                    # Resize labels after augmentations
                    it_lab = resize_image_label(
                        im=it_lab,
                        model_input_image_size=resize_output,
                        f='area')
                images += [it_im]
                labels += [it_lab]
            label = tf.stack(
                labels,
                axis=0)
            image = tf.stack(
                images,
                axis=0)
        else:
            if None not in data_augmentations:
                for im in split_images:
                    it_im = image_augmentations(
                        image=im,
                        model_input_image_size=model_input_image_size[1:],
                        data_augmentations=data_augmentations)
                    images += [it_im]
                image = tf.stack(
                    images,
                    axis=0)
    # if image.dtype != tf.float32:
    #     image = tf.cast(image, tf.float32)
    return image, label, aux


def placeholder_image_augmentations(
        images,
        model_input_image_size,
        data_augmentations,
        batch_size,
        labels=None,
        aug_lab=False):
    """Apply augmentations to placeholder data."""
    split_images = tf.split(images, batch_size, axis=0)
    if labels is not None:
        split_labels = tf.split(labels, batch_size, axis=0)
    else:
        split_labels = [None] * batch_size
    aug_images, aug_labels = [], []
    for idx in range(batch_size):
        if aug_lab:
            aug_image, aug_label = image_augmentations(
                image=tf.squeeze(split_images[idx], axis=0),
                data_augmentations=data_augmentations,
                model_input_image_size=model_input_image_size,
                label=tf.squeeze(split_labels[idx], axis=0))
        else:
            aug_image, _ = image_augmentations(
                image=tf.squeeze(split_images[idx], axis=0),
                data_augmentations=data_augmentations,
                model_input_image_size=model_input_image_size)
            aug_label = split_labels[idx]
        aug_images += [aug_image]
        aug_labels += [aug_label]
    return tf.stack(aug_images), tf.stack(aug_labels)


def inputs_mt(
        dataset,
        batch_size,
        model_input_image_size,
        tf_dict,
        data_augmentations,
        num_epochs,
        tf_reader_settings,
        shuffle,
        number_of_files=1,
        resize_output=None):
    """Read tfrecords and prepare them for queueing. Multithread."""
    min_after_dequeue = 1000
    capacity = 10000 * batch_size  # min_after_dequeue + 5 * batch_size
    num_threads = 2

    # Check if we need timecourses.
    if len(model_input_image_size) == 4:
        number_of_files = model_input_image_size[0]

    # Start data loader loop
    with tf.name_scope('input'):
        filename_queue = tf.train.string_input_producer(
            [dataset], num_epochs=num_epochs)
        # Even when reading in multiple threads, share the filename
        # queue.
        image_data, label_data = [], []
        for idx in range(batch_size):
            images, labels = read_and_decode(
                filename_queue=filename_queue,
                model_input_image_size=model_input_image_size,
                tf_dict=tf_dict,
                tf_reader_settings=tf_reader_settings,
                data_augmentations=data_augmentations,
                number_of_files=number_of_files,
                resize_output=resize_output)
            image_data += [tf.expand_dims(images, axis=0)]
            label_data += [tf.expand_dims(labels, axis=0)]

        # Shuffle the examples and collect them into batch_size batches.
        # (Internally uses a RandomShuffleQueue.)
        # We run this in two threads to avoid being a bottleneck.
        batch_data = [
            tf.concat(image_data, axis=0),  # CHANGE TO STACK
            tf.concat(label_data, axis=0)
        ]
        if shuffle:
            images, labels = tf.train.shuffle_batch(
                batch_data,
                batch_size=batch_size,
                num_threads=num_threads,
                capacity=capacity,
                enqueue_many=True,
                # Ensures a minimum amount of shuffling of examples.
                min_after_dequeue=min_after_dequeue)
        else:
            images, labels = tf.train.batch(
                batch_data,
                batch_size=batch_size,
                num_threads=num_threads,
                enqueue_many=True,
                capacity=capacity)
        return images, labels


def inputs(
        dataset,
        batch_size,
        model_input_image_size,
        tf_dict,
        data_augmentations,
        num_epochs,
        tf_reader_settings,
        shuffle,
        aux=None,
        number_of_files=1,
        resize_output=None):
    """Read tfrecords and prepare them for queueing."""
    min_after_dequeue = 1000
    capacity = min_after_dequeue + 1000 * batch_size  # min_after_dequeue + 5 * batch_size
    num_threads = 1

    # Check if we need timecourses.
    if len(model_input_image_size) == 4:
        number_of_files = model_input_image_size[0]
    # Start data loader loop
    if not isinstance(dataset, list):
        dataset = [dataset]
    with tf.name_scope('input'):
        filename_queue = tf.train.string_input_producer(
            dataset, num_epochs=num_epochs)
        # Even when reading in multiple threads, share the filename
        # queue.
        batch_data = read_and_decode(
            filename_queue=filename_queue,
            model_input_image_size=model_input_image_size,
            tf_dict=tf_dict,
            tf_reader_settings=tf_reader_settings,
            data_augmentations=data_augmentations,
            number_of_files=number_of_files,
            aux=aux,
            resize_output=resize_output)

        # Shuffle the examples and collect them into batch_size batches.
        # (Internally uses a RandomShuffleQueue.)
        # We run this in two threads to avoid being a bottleneck.
        if shuffle:
            images, labels, aux = tf.train.shuffle_batch(
                batch_data,
                batch_size=batch_size,
                num_threads=num_threads,
                capacity=capacity,
                # Ensures a minimum amount of shuffling of examples.
                min_after_dequeue=min_after_dequeue)
        else:
            images, labels, aux = tf.train.batch(
                batch_data,
                batch_size=batch_size,
                num_threads=num_threads,
                capacity=capacity)
        return images, labels, aux

