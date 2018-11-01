import tensorflow as tf
import random
from random import shuffle
from glob import glob
import os
import numpy as np

import cv2


def _corrupt_brightness(image, mask):
    """Radnomly applies a random brightness change."""
    cond_brightness = tf.cast(tf.random_uniform(
        [], maxval=2, dtype=tf.int32), tf.bool)
    image = tf.cond(cond_brightness, lambda: tf.image.random_hue(
        image, 0.1), lambda: tf.identity(image))
    return image, mask


def _corrupt_contrast(image, mask):
    """Randomly applies a random contrast change."""
    cond_contrast = tf.cast(tf.random_uniform(
        [], maxval=2, dtype=tf.int32), tf.bool)
    image = tf.cond(cond_contrast, lambda: tf.image.random_contrast(
        image, 0.2, 1.8), lambda: tf.identity(image))
    return image, mask


def _corrupt_saturation(image, mask):
    """Randomly applies a random saturation change."""
    cond_saturation = tf.cast(tf.random_uniform(
        [], maxval=2, dtype=tf.int32), tf.bool)
    image = tf.cond(cond_saturation, lambda: tf.image.random_saturation(
        image, 0.2, 1.8), lambda: tf.identity(image))
    return image, mask


def _crop_random_segmentation(image, mask, target_size):
    """Randomly crops image and mask in accord."""
    img_rows, img_cols = target_size
    seed = random.random()
    cond_crop_image = tf.cast(tf.random_uniform(
        [], maxval=2, dtype=tf.int32, seed=seed), tf.bool)
    cond_crop_mask = tf.cast(tf.random_uniform(
        [], maxval=2, dtype=tf.int32, seed=seed), tf.bool)

    image = tf.cond(cond_crop_image, lambda: tf.random_crop(
        image, [int(img_rows * 0.85), int(img_cols * 0.85), 3], seed=seed), lambda: tf.identity(image))
    mask = tf.cond(cond_crop_mask, lambda: tf.random_crop(
        mask, [int(img_rows * 0.85), int(img_cols * 0.85), 1], seed=seed), lambda: tf.identity(mask))
    image = tf.expand_dims(image, axis=0)
    mask = tf.expand_dims(mask, axis=0)

    image = tf.image.resize_images(image, [img_rows, img_cols])
    mask = tf.image.resize_images(mask,   [img_rows, img_cols])

    image = tf.squeeze(image, axis=0)
    mask = tf.squeeze(mask, axis=0)

    return image, mask

def _crop_random_classification(image, label, target_size):
    """Randomly crops image and mask in accord."""
    seed = random.random()
    img_rows, img_cols = target_size
    cond_crop_image = tf.cast(tf.random_uniform(
        [], maxval=2, dtype=tf.int32, seed=seed), tf.bool)
    
    image = tf.cond(cond_crop_image, lambda: tf.random_crop(
        image, [int(img_rows * 0.85), int(img_cols * 0.85), 3], seed=seed), lambda: tf.identity(image))
    image = tf.expand_dims(image, axis=0)
    
    image = tf.image.resize_images(image, [img_rows, img_cols])
    
    image = tf.squeeze(image, axis=0)
    
    return image, label


def _flip_left_right_segmentation(image, mask):
    """Randomly flips image and mask left or right in accord."""
    seed = random.random()
    image = tf.image.random_flip_left_right(image, seed=seed)
    mask = tf.image.random_flip_left_right(mask, seed=seed)

    return image, mask

def _flip_left_right_classification(image, label):
    """Randomly flips image and mask left or right in accord."""
    seed = random.random()
    image = tf.image.random_flip_left_right(image, seed=seed)

    return image, label


def _normalize_data(image, mask):
    """Normalize image and mask within range 0-1."""
    image = tf.cast(image, tf.float32)
    image = image / 255.0

    mask = tf.cast(mask, tf.float32)
    mask = mask / 255.0
    mask = tf.cast(mask, tf.int32)

    return image, mask


def _resize_data(image, mask):
    """Resizes images to smaller dimensions."""
    image = tf.image.resize_images(image, [480, 640])
    mask = tf.image.resize_images(mask, [480, 640])

    return image, mask


def _parse_data_segmentation(image_paths, mask_paths):
    """Reads image and mask files"""
    image_content = tf.read_file(image_paths)
    label_content = tf.read_file(mask_paths)

    images = tf.image.decode_jpeg(image_content, channels=3)
    mask = tf.image.decode_jpeg(label_content, channels=1)

    return images, mask

def _parse_data_classification(image_paths, labels, target_size, class_len):
    """Reads image and mask files"""
    image_content = tf.read_file(image_paths)

    images = tf.image.decode_jpeg(image_content, channels=3)
    images = tf.image.resize_images(images, target_size)
    labels = tf.one_hot(labels, class_len)
    # print (images)

    return images, labels

def _resize_data_classification(image, label, target_size):

    image = tf.image.resize_images(image, target_size)

    return image, label

def segmentation_data(image_paths, mask_paths, augment=False, shuffle_data = True, 
                    seed=None,  num_parallel_calls=2, prefetch=64, batch_size=32):
    """Reads data, normalizes it, shuffles it, then batches it, returns a
       the next element in dataset op and the dataset initializer op.
       Inputs:
        image_paths: A list of paths to individual images
        mask_paths: A list of paths to individual mask images
        augment: Boolean, whether to augment data or not
        batch_size: Number of images/masks in each batch returned
        num_threads: Number of parallel calls to make
       Returns:
        next_element: A tensor with shape [2], where next_element[0]
                      is image batch, next_element[1] is the corresponding
                      mask batch
        init_op: Data initializer op, needs to be executed in a session
                 for the data queue to be filled up and the next_element op
                 to yield batches"""

    # Convert lists of paths to tensors for tensorflow
    images_name_tensor = tf.constant(image_paths)
    mask_name_tensor = tf.constant(mask_paths)

    # Create dataset out of the 2 files:
    data = tf.data.Dataset.from_tensor_slices(
        (images_name_tensor, mask_name_tensor))

    # Parse images and labels
    data = data.map(
        _parse_data_segmentation, num_parallel_calls=num_parallel_calls).prefetch(prefetch)

    # If augmentation is to be applied
    
    if augment:
        data = data.map(_corrupt_brightness,
                        num_parallel_calls=num_parallel_calls).prefetch(prefetch)

        data = data.map(_corrupt_contrast,
                        num_parallel_calls=num_parallel_calls).prefetch(prefetch)

        data = data.map(_corrupt_saturation,
                        num_parallel_calls=num_parallel_calls).prefetch(prefetch)

        data = data.map(
            _crop_random_segmentation, num_parallel_calls=num_parallel_calls).prefetch(prefetch)

        data = data.map(_flip_left_right_segmentation,
                        num_parallel_calls=num_parallel_calls).prefetch(prefetch)

    # Batch the data
    data = data.batch(batch_size)

    # Resize to smaller dims for speed
    data = data.map(_resize_data, num_parallel_calls=num_parallel_calls).prefetch(prefetch)

    # Normalize
    # data = data.map(_normalize_data,
    #                 num_parallel_calls=num_threads).prefetch(30)

    data = data.shuffle(prefetch)

    # Create iterator
    iterator = tf.data.Iterator.from_structure(
        data.output_types, data.output_shapes)

    # Next element Op
    next_element = iterator.get_next()

    # Data set init. op
    init_op = iterator.make_initializer(data)

    return next_element, init_op

def flow_from_directory(directory, target_size=(256, 256), batch_size=32, shuffle_data=True, 
                seed=None,  num_parallel_calls=2, 
                prefetch=64):
    augment = True
    class_names = [os.path.split(x)[1] for x in glob(os.path.join(directory, "*")) ]
    class_map = dict(zip(class_names, range(len(class_names))))
    dataset = glob(os.path.join(directory, "**", "*"))
    label = []
    for i in dataset:
        label.append(os.path.split(os.path.split(i)[0])[1])
    print (class_map)
    labels = (list(map(class_map.get, label)))
    if shuffle_data:
        unshuffle = list(zip(dataset, labels))
        shuffle(unshuffle)
        dataset, labels = zip(*unshuffle)
    
    image_data = tf.constant(dataset)
    label_data = tf.constant(labels)
    data = tf.data.Dataset.from_tensor_slices((image_data, label_data))
    
    data = data.map(lambda x, y: _parse_data_classification(x, y, target_size, len(class_names)), num_parallel_calls=num_parallel_calls).prefetch(prefetch)
    
    if augment:
        data = data.map(_corrupt_brightness,
                        num_parallel_calls=num_parallel_calls).prefetch(30)

        data = data.map(_corrupt_contrast,
                        num_parallel_calls=num_parallel_calls).prefetch(30)

        data = data.map(_corrupt_saturation,
                        num_parallel_calls=num_parallel_calls).prefetch(30)

        data = data.map(
            lambda x, y: _crop_random_classification(x, y, target_size), num_parallel_calls=num_parallel_calls).prefetch(30)

        data = data.map(_flip_left_right_classification,
                        num_parallel_calls=num_parallel_calls).prefetch(30)
    
    
    # To ensure that batch of different images are passed
    # data = data.apply(tf.contrib.data.unbatch())
    data = data.shuffle(prefetch).repeat()

    # Batch the data
    data = data.batch(batch_size)

    # Resize to smaller dims for speed
    data = data.map(lambda x, y: _resize_data_classification(x, y, target_size), num_parallel_calls=num_parallel_calls).prefetch(prefetch)


    # Normalize
    # data = data.map(_normalize_data,
    #                 num_parallel_calls=num_parallel_calls).prefetch(30)


    # Create iterator
    iterator = tf.data.Iterator.from_structure(
        data.output_types, data.output_shapes)

    # Next element Op
    next_element = iterator.get_next()

    # Data set init. op
    init_op = iterator.make_initializer(data)

    return next_element, init_op

def _parse_single_image(image_paths, s):
    """Reads image and s is the scale factor"""
    image_content = tf.read_file(image_paths)
    
    images = tf.image.decode_jpeg(image_content, channels=3)
    
    # Define some transformation for the label (if you are using unsupervised learning)
    # Since the task is super-resolution, we need to downscale the images.
    print (images.get_shape())
    init_w = tf.shape(images)[0]
    init_h = tf.shape(images)[1]
    
    new_w = tf.to_int32(tf.to_float(init_w) / s)
    new_h = tf.to_int32(tf.to_float(init_h) / s)

    mask = tf.image.resize_images(images, (new_w, new_h))

    return images, mask

def read_no_labels(directory, s = 2, batch_size=1, shuffle_data=True, 
                seed=None,  num_parallel_calls=2, 
                prefetch=64):
    
    dataset = glob(os.path.join(directory, "*"))
    
    image_data = tf.constant(dataset)
    data = tf.data.Dataset.from_tensor_slices(image_data)
    
    data = data.map(lambda x: _parse_single_image(x, s), num_parallel_calls=num_parallel_calls).prefetch(prefetch)
    
    # Dont perform data augmentation
    '''
    if augment:
        data = data.map(_corrupt_brightness,
                        num_parallel_calls=num_parallel_calls).prefetch(30)

        data = data.map(_corrupt_contrast,
                        num_parallel_calls=num_parallel_calls).prefetch(30)

        data = data.map(_corrupt_saturation,
                        num_parallel_calls=num_parallel_calls).prefetch(30)

        data = data.map(
            lambda x, y: _crop_random_classification(x, y, target_size), num_parallel_calls=num_parallel_calls).prefetch(30)

        data = data.map(_flip_left_right_classification,
                        num_parallel_calls=num_parallel_calls).prefetch(30)
    '''
    
    # To ensure that batch of different images are passed
    # data = data.apply(tf.contrib.data.unbatch())
    data = data.shuffle(prefetch).repeat()

    # Batch the data
    data = data.batch(batch_size)

    # Resize to smaller dims for speed
    # data = data.map(lambda x, y: _resize_data_classification(x, y, target_size), num_parallel_calls=num_parallel_calls).prefetch(prefetch)


    # Normalize
    # data = data.map(_normalize_data,
    #                 num_parallel_calls=num_parallel_calls).prefetch(30)


    # Create iterator
    iterator = tf.data.Iterator.from_structure(
        data.output_types, data.output_shapes)

    # Next element Op
    next_element = iterator.get_next()

    # Data set init. op
    init_op = iterator.make_initializer(data)

    return next_element, init_op


if __name__ == '__main__':
    dire = '/home/tlokeshkumar/Downloads/GTOS_256/h_sample002_01'
    n, ini = read_no_labels(dire)

    print ("passed the functions successfully!")

    init = tf.global_variables_initializer()
    
    sess = tf.Session()
    sess.run(init)
    sess.run(ini)

    for i in range(5):
        a, b = sess.run(n)
        print (b)
        print (a.shape)
        print (np.squeeze(b, axis=0).shape)
        cv2.imshow("image", np.squeeze(a, axis=0))
        cv2.imshow("downsampled", np.squeeze(b.astype('uint8'), axis=0))
        cv2.waitKey()
        cv2.destroyAllWindows()
