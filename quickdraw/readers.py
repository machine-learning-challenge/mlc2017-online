# Copyright 2016 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS-IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Provides readers configured for different datasets."""

import tensorflow as tf
import utils
import tensorflow.contrib.slim as slim

from tensorflow import logging
import numpy as np

def resize_axis(tensor, axis, new_size, fill_value=0):
  """Truncates or pads a tensor to new_size on on a given axis.

  Truncate or extend tensor such that tensor.shape[axis] == new_size. If the
  size increases, the padding will be performed at the end, using fill_value.

  Args:
    tensor: The tensor to be resized.
    axis: An integer representing the dimension to be sliced.
    new_size: An integer or 0d tensor representing the new value for
      tensor.shape[axis].
    fill_value: Value to use to fill any new entries in the tensor. Will be
      cast to the type of tensor.

  Returns:
    The resized tensor.
  """
  tensor = tf.convert_to_tensor(tensor)
  shape = tf.unstack(tf.shape(tensor))

  pad_shape = shape[:]
  pad_shape[axis] = tf.maximum(0, new_size - shape[axis])

  shape[axis] = tf.minimum(shape[axis], new_size)
  shape = tf.stack(shape)

  resized = tf.concat([
      tf.slice(tensor, tf.zeros_like(shape), shape),
      tf.fill(tf.stack(pad_shape), tf.cast(fill_value, tensor.dtype))
  ], axis)

  # Update shape.
  new_shape = tensor.get_shape().as_list()  # A copy is being made.
  new_shape[axis] = new_size
  resized.set_shape(new_shape)
  return resized

class BaseReader(object):
  """Inherit from this class when implementing new readers."""

  def prepare_reader(self, unused_filename_queue):
    """Create a thread for generating prediction and label tensors."""
    raise NotImplementedError()



class QuickDrawFeatureReader(BaseReader):
  """Reads TFRecords of pre-aggregated Examples.
  """

  def __init__(self,
               num_classes=10):
    self.num_classes = num_classes

  def prepare_reader(self, filename_queue, batch_size=32):
    reader = tf.TFRecordReader()
    _, serialized_examples = reader.read_up_to(filename_queue, batch_size)

    tf.add_to_collection("serialized_examples", serialized_examples)
    return self.prepare_serialized_examples(serialized_examples)

  def prepare_serialized_examples(self, serialized_examples, width=50, height=50):
    # set the mapping from the fields to data types in the proto
    feature_map = {
           'image': tf.FixedLenFeature((), tf.string, default_value=''),
           'label': tf.FixedLenFeature([], tf.int64)
    }
    features = tf.parse_example(serialized_examples, features=feature_map)

    def decode_and_resize(image_str_tensor):
      """Decodes png string, resizes it and returns a uint8 tensor."""
  
      # Output a grayscale (channels=1) image
      image = tf.image.decode_png(image_str_tensor, channels=1)
  
      # Note resize expects a batch_size, but tf_map supresses that index,
      # thus we have to expand then squeeze.  Resize returns float32 in the
      # range [0, uint8_max]
      image = tf.expand_dims(image, 0)
      image = tf.image.resize_bilinear(
          image, [height, width], align_corners=False)
      image = tf.squeeze(image, squeeze_dims=[0])
      image = tf.cast(image, dtype=tf.uint8)
      return image

    images_str_tensor = features["image"]
    images = tf.map_fn(
        decode_and_resize, images_str_tensor, back_prop=False, dtype=tf.uint8)
    images = tf.image.convert_image_dtype(images, dtype=tf.float32)
    images = tf.subtract(images, 0.5)
    images = tf.multiply(images, 2.0)

    def dense_to_one_hot(label_batch, num_classes):
      one_hot = tf.map_fn(lambda x : tf.cast(slim.one_hot_encoding(x, num_classes), tf.int32), label_batch)
      one_hot = tf.reshape(one_hot, [-1, num_classes])
      return one_hot

    labels = tf.cast(features['label'], tf.int32)
    labels = dense_to_one_hot(labels, 10)

    return images, labels

class QuickDrawTestFeatureReader(BaseReader):
  """Reads TFRecords of pre-aggregated Examples.
  """

  def __init__(self,
               num_classes=10):
    self.num_classes = num_classes

  def prepare_reader(self, filename_queue, batch_size=32):
    reader = tf.TFRecordReader()
    _, serialized_examples = reader.read_up_to(filename_queue, batch_size)

    tf.add_to_collection("serialized_examples", serialized_examples)
    return self.prepare_serialized_examples(serialized_examples)

  def prepare_serialized_examples(self, serialized_examples, width=50, height=50):
    # set the mapping from the fields to data types in the proto
    feature_map = {
           'image': tf.FixedLenFeature((), tf.string, default_value=''),
           'image_id': tf.FixedLenFeature((), tf.string, default_value=''),
    }
    features = tf.parse_example(serialized_examples, features=feature_map)

    def decode_and_resize(image_str_tensor):
      """Decodes png string, resizes it and returns a uint8 tensor."""
  
      # Output a grayscale (channels=1) image
      image = tf.image.decode_png(image_str_tensor, channels=1)
  
      # Note resize expects a batch_size, but tf_map supresses that index,
      # thus we have to expand then squeeze.  Resize returns float32 in the
      # range [0, uint8_max]
      image = tf.expand_dims(image, 0)
      image = tf.image.resize_bilinear(
          image, [height, width], align_corners=False)
      image = tf.squeeze(image, squeeze_dims=[0])
      image = tf.cast(image, dtype=tf.uint8)
      return image

    images_str_tensor = features["image"]
    images = tf.map_fn(
        decode_and_resize, images_str_tensor, back_prop=False, dtype=tf.uint8)
    images = tf.image.convert_image_dtype(images, dtype=tf.float32)
    images = tf.subtract(images, 0.5)
    images = tf.multiply(images, 2.0)

    image_id = features["image_id"]
    return image_id, images
  
