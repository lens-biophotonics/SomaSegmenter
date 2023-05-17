# =============================================================================
# DATA GENERATORS
#
# last major rev. 2020/04
#
# Filippo Maria Castelli
# LENS Biophotonics Group
# castelli@lens.unifi.it
# =============================================================================

from pathlib import Path
from functools import partial
import logging

import numpy as np
from skimage.io import imread
import tensorflow as tf

import tifffile


def build_dataset(
        frame_path,
        labels_path,
        crop_shape=(64, 64),
        batch_size=5,
        data_augmentation=True,
        augment_settings=None,
        augmentation_threads=4,
):
    """
    Dataset building routine

    Parameters
    ----------
    frame_path : pathlib.Path
        path to training data .tif
        has to be in [0, 255] range
    labels_path : pathlib.Path
        path to training labels .tif
        has to be in [0, 255] range
    crop_shape : tuple
        shape of random crops
    batch_size : int
        number of examples in a single batch
    data_augmentation : bool
        If True enables data augmentation
        Defaults to True
    augment_settings : dict
        Dictionary with independent probabilities for every augmentation step
        Keys must be "p_rotate", "p_flip", "p_gamma_transform", "p_brightness_scale", "p_gaussian_noise"
    augmentation_threads : int
        number of independent data augmentation threads
        
    Returns
    -------
    dataset : tf.data.Dataset
        Tensorflow dataset
    """

    if augment_settings is None:
        augment_settings = {}
        
    frame_npy, labels_npy = load_volumes(frame_path, labels_path)

    img_shape = frame_npy.shape

    logging.debug("Conversion to tf.Tensor")
    frame = tf.convert_to_tensor(frame_npy, name="frame", dtype=tf.float32)
    labels = tf.convert_to_tensor(labels_npy, name="labels", dtype=tf.float32)
    
    # Removing original tensors from memory as we don't use them anymore
    del frame_npy, labels_npy
    
    assert tf.math.reduce_max(labels) <= 1, "Labels max is > 1, please normalize labels"
    assert tf.math.reduce_min(labels) >= 0, "Labels min is < 1, please normalize labels"
    
    # creating dataset
    dataset = tf.data.Dataset.from_tensor_slices((frame, labels))
    
    # NOTE ON DATASET SHUFFLING:
    # shuffle before repeat provides ordering guarantees
    # repeat before shuffle blurs the boundaries between epochs but no reset
    # is needed between epochs and provides better performance
    
    dataset = dataset.repeat()
    dataset = dataset.shuffle(buffer_size=img_shape[0] + 100)

    # Performing random crops
    partial_random_crop = partial(random_crop, crop_shape=crop_shape)

    dataset = dataset.map(
        map_func=lambda img_stack, label_stack: tf.py_function(
            func=partial_random_crop, inp=[img_stack, label_stack], Tout=[tf.float32, tf.float32]
        ),
        num_parallel_calls=int(augmentation_threads),
    )
    dataset = dataset.unbatch()
    
    # Optional data agumentation sequence
    if data_augmentation:
        partial_augment = partial(augment, **augment_settings)
        
        dataset = dataset.map(
            map_func=lambda img, label: tf.py_function(
                func=partial_augment, inp=[img, label], Tout=[tf.float32, tf.float32]
            ),
            
            num_parallel_calls=int(augmentation_threads)
        )
    # Restoring dimension information
    img_crop_shape = (crop_shape[0], crop_shape[1], img_shape[-1])
    label_crop_shape = (crop_shape[0], crop_shape[1], 1)

    dataset = dataset.map(
        map_func=lambda img, label: set_shape(img, label, img_crop_shape, label_crop_shape))
    
    # batching dataset
    dataset = dataset.batch(batch_size=batch_size)
    # prefetching
    dataset = dataset.prefetch(buffer_size=100)

    # Defining a custom steps_per_epoch attribute
    dataset.steps_per_epoch = (np.prod(img_shape[:-1]) // np.prod(crop_shape)) // batch_size

    return dataset


def load_volumes(frame_path, labels_path):
    """load, normalize and fix shapes of stacks stacks

    Parameters:
    -----------
    frame_path : Pathlib path
        Path of input data stack
    labels_path : Pathlib path
        Path of label data stack
        
    Returns:
    --------
    frame_npy : numpy array
        normalized input image in [z, y, x, ch] format
    labels_npy : numpy array
        normalized label image in [z, y, x, 1] format    
    """
    
    logging.debug("reading from disk")
    frame_npy = (imread(str(frame_path), plugin="pil") / 255).astype(np.float32)
    labels_npy = (imread(str(labels_path), plugin="pil") / 255).astype(np.float32)
    # print("FRAME SHAPE {}".format(frame_npy.shape))
    # print("MASK SHAPE {}".format(labels_npy.shape))

    if len(frame_npy.shape) == 4 and len(labels_npy.shape) == 4:
        
        # case: multichannel labels
        if labels_npy.shape[-1] != 1:
            raise Exception("Labels must be monochrome for single-class semantic segmentation")

    elif len(frame_npy.shape) == 4 and len(labels_npy.shape) == 3:
        # case: multichannel frames and monochrome labels
        # expand label dimension
        labels_npy = np.expand_dims(labels_npy, axis=-1)
    elif len(frame_npy.shape) == 3 and len(labels_npy.shape) == 3:
        # case: both images are monochrome
        # expand both images dimensions
        labels_npy = np.expand_dims(labels_npy, axis=-1)
        frame_npy = np.expand_dims(frame_npy, axis=-1)
    else:
        raise Exception("Could not recognize format for frame and labels")

    return frame_npy, labels_npy


def set_shape(img, label, img_shape, label_shape):
    """simple patch function for reshaping in dataset generation"""
    img.set_shape(img_shape)
    label.set_shape(label_shape)
    return img, label


def random_crop(img, label, crop_shape):
    """
    Randomly crop the same crop_shape pathc from both img and label
    The total number of crops is total_image_pixels // total_crop_pixels

    Parameters
    ----------
    img : tf.Tensor
        [width, height] tf.Tensor of training data
    label : tf.Tensor
        [width, height] tf.Tensor of labels
    crop_shape : tuple
        shape of area to crop
        
    Returns
    -------
    img_crop_stack : tf.Tensor
        [n_crops, width, height] img crop tensor
    label_crop_stack : tf.Tensor
        [n_crops, width, height] label crop tensor
    """
    # Counting how many crops are needed
    img_shape = img.shape.as_list()
    n_pix_img = np.prod(img_shape)
    n_pix_crop = np.prod(crop_shape)
    n_crops = n_pix_img // n_pix_crop

    # defining crop boxes
    lower_x = np.random.uniform(
        low=0, high=1 - (crop_shape[0] / img_shape[0]), size=(n_crops)
    )
    lower_y = np.random.uniform(
        low=0, high=1 - (crop_shape[1] / img_shape[1]), size=(n_crops)
    )
    
    upper_x = lower_x + crop_shape[0] / img_shape[0]
    upper_y = lower_y + crop_shape[1] / img_shape[1]

    crop_boxes = np.column_stack((lower_x, lower_y, upper_x, upper_y))

    # concatenate img and mask along channel
    concat = tf.concat([img, label], axis=-1)

    # adding a batch dimension
    concat = tf.expand_dims(concat, axis=0)

    # image cropping
    # cropped shape should be [n_crops, crop_height, crop_width, channels]
    crops = tf.image.crop_and_resize(
        image=concat,
        boxes=crop_boxes,
        box_indices=np.zeros(n_crops),
        crop_size=crop_shape,
        method="nearest"
    )
    
    img_crop_stack = crops[..., :-1]
    label_crop_stack = tf.expand_dims(crops[..., -1], axis=-1)

    return img_crop_stack, label_crop_stack


def random_rotate(img, label):
    """Perform random 90 degree rotation"""
    rot = tf.random.uniform(shape=[], minval=1, maxval=3, dtype=tf.int32)
    # tf.image.rot90 supports [width, height, channels] or [batch, width, height, channels]
    img = tf.image.rot90(image=img, k=rot)
    label = tf.image.rot90(image=label, k=rot)

    return img, label


def random_flip(img, label):
    """Perform random up/down left/right flip"""
    flips = np.random.choice(a=[True, False], size=(2))
    if flips[0]:
        img = tf.image.flip_left_right(img)
        label = tf.image.flip_left_right(label)

    if flips[1]:
        img = tf.image.flip_up_down(img)
        label = tf.image.flip_up_down(label)

    return img, label


def random_brightness_scale(img, label, scale_range=0.2):
    """Randomly scale image values"""
    min_scale = 1.0 - scale_range / 2
    max_scale = 1.0 + scale_range / 2
    scale = tf.random.uniform(
        shape=[], minval=min_scale, maxval=max_scale, dtype=tf.float32
    )
    img = img * scale
    return tf.clip_by_value(img, clip_value_min=0.0, clip_value_max=1.0), label


def random_gaussian_noise(img, label, std_min=1e-5, std_max=0.06):
    """Add gaussian noise"""
    noise_std = tf.random.uniform(
        shape=[], minval=std_min, maxval=std_max, dtype=tf.float32
    )
    noise = tf.random.normal(
        shape=tf.shape(img), mean=0.0, stddev=noise_std, dtype=tf.float32
    )
    img = img + noise
    return tf.clip_by_value(img, clip_value_min=0.0, clip_value_max=1.0), label


def random_gamma_transform(img, label, gamma_range=0.2):
    """Perform a gamma transform with a random gamma factor"""
    min_gamma = 1.0 - gamma_range / 2
    max_gamma = 1.0 + gamma_range / 2
    gamma = tf.random.uniform(
        shape=[], minval=min_gamma, maxval=max_gamma, dtype=tf.float32
    )
    img = tf.pow(img, gamma)
    return (
        tf.clip_by_value(img, clip_value_min=0.0, clip_value_max=1.0),
        label,
    )  # shouldn't exceed 1 actually


def augment(
        img,
        label,
        p_rotate=0.5,
        p_flip=0.5,
        p_brightness_scale=0.5,
        p_gaussian_noise=0.5,
        p_gamma_transform=0.5,
        brightness_scale_range=0.2,
        gaussian_noise_std_max=0.06,
        gamma_range=0.1
):
    """
    Apply data augmentation steps
    each step step is performed with a given probability.

    Parameters
    ----------
    img : tf.Tensor
        [width, height, channels] image tensor
    label : tf.Tensor
        [width, height, channels] label tensor
    p_rotate : float
        independent probability of random rotate step
    p_flip : float
        independent probability of random flip step
    p_brightness_scale : float
        independent probability of random brightness scale step
    p_gaussian_noise : float
        independent probability of gaussian noise addition step
    p_gamma_transform : float
        independent probability of gamma transform step
    brightness_scale_range : float
        range around 1.0 for brightness scaling
    gaussian_noise_std_max : float
        max standard deviation for gaussian noise
    gamma_range : float
        range around 1.0 for gamma transform

    Returns
    -------
    augmented_img : tf.Tensor
        data agumented img
    augmented_label: tf.Tensor
        data augmented label

    """
    # Executing each step with a random probability.
    x = np.random.uniform(low=0, high=1, size=(5))

    if x[0] < p_rotate:
        img, label = random_rotate(img, label)
    if x[1] < p_flip:
        img, label = random_flip(img, label)
    if x[2] < p_brightness_scale:
        img, label = random_brightness_scale(img, label, scale_range=brightness_scale_range)
    if x[3] < p_gaussian_noise:
        img, label = random_gaussian_noise(img, label, std_max=gaussian_noise_std_max)
    if x[4] < p_gamma_transform:
        img, label = random_gamma_transform(img, label, gamma_range=gamma_range)
        
    return img, label


if __name__ == "__main__":
    # Define paths
    # frame_pathlib_path = Path("X:/castelli/em_dataset/test_frames.tif")
    # labels_pathlib_path = Path("X:/castelli/em_dataset/test_masks.tif")

    # frame_pathlib_path = Path("/mnt/NASone3/castelli/2pe_dataset/test_frames.tif")
    # labels_pathlib_path = Path("/mnt/NASone3/castelli/2pe_dataset/test_masks.tif")
    frame_pathlib_path = Path("/mnt/NASone3/castelli/em_dataset/test_frames.tif")
    labels_pathlib_path = Path("/mnt/NASone3/castelli/em_dataset/test_masks.tif")

    # Define independent proabilities for single data augmentations
    augment_settings = {
        "p_rotate": 0.5,
        "p_flip": 0.5,
        "p_brightness_scale": 0.,
        "p_gaussian_noise": 0.,
        "p_gamma_transform": 0.,
    }

    # Load data from disk in two numpy arrays
    try:
        numpy_frames
    except NameError:
        logging.debug("loading from disk MAIN")
        numpy_frames, numpy_labels = load_volumes(frame_pathlib_path, labels_pathlib_path)

    # Build a tf.data.Dataset
    crop_img_shape = (256, 256)
    logging.debug("conversion to tf.Tensor")
    frame = tf.convert_to_tensor(numpy_frames[0], name="frame", dtype=tf.float32)
    labels = tf.convert_to_tensor(numpy_labels[0], name="labels", dtype=tf.float32)

    random_crop(frame, labels, crop_img_shape)
    ds = build_dataset(
        frame_path=frame_pathlib_path,
        labels_path=labels_pathlib_path,
        batch_size=1,
        data_augmentation=False,
        crop_shape=crop_img_shape,
        # augment_settings=augment_probs,
        frame_npy=numpy_frames,
        labels_npy=numpy_labels
    )
    # Create an iterator over the dataset
    iterator = ds.__iter__()

    # Extract 1000 iterations over the dataset
    fs = np.ndarray(shape=(1000, crop_img_shape[0], crop_img_shape[1], 1))
    msks = np.copy(fs)

    logging.debug("iterating")
    for i in range(1000):
        a, b = next(iterator)
        fs[i, ...] = a
        msks[i, ...] = b

    logging.debug("plotting")
    from multi_slice_viewer import multi_slice_viewer

    fs = np.squeeze(fs)
    msks = np.squeeze(msks)
    multi_slice_viewer(fs, msks, ignore_channel=False)
    combined = (np.concatenate([fs, msks], axis=2)).astype(np.float32)
    tifffile.imwrite("combined.tif", combined)
