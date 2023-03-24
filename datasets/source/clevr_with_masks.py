# Copyright 2019 DeepMind Technologies Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""CLEVR (with masks) dataset reader."""

import tensorflow.compat.v1 as tf
import numpy as np
import os
tf.enable_eager_execution()



COMPRESSION_TYPE = tf.io.TFRecordOptions.get_compression_type_string('GZIP')
IMAGE_SIZE = [240, 320]
# The maximum number of foreground and background entities in the provided
# dataset. This corresponds to the number of segmentation masks returned per
# scene.
MAX_NUM_ENTITIES = 11
BYTE_FEATURES = ['mask', 'image', 'color', 'material', 'shape', 'size']

# Create a dictionary mapping feature names to `tf.Example`-compatible
# shape and data type descriptors.
features = {
    'image': tf.FixedLenFeature(IMAGE_SIZE+[3], tf.string),
    'mask': tf.FixedLenFeature([MAX_NUM_ENTITIES]+IMAGE_SIZE+[1], tf.string),
    'x': tf.FixedLenFeature([MAX_NUM_ENTITIES], tf.float32),
    'y': tf.FixedLenFeature([MAX_NUM_ENTITIES], tf.float32),
    'z': tf.FixedLenFeature([MAX_NUM_ENTITIES], tf.float32),
    'pixel_coords': tf.FixedLenFeature([MAX_NUM_ENTITIES, 3], tf.float32),
    'rotation': tf.FixedLenFeature([MAX_NUM_ENTITIES], tf.float32),
    'size': tf.FixedLenFeature([MAX_NUM_ENTITIES], tf.string),
    'material': tf.FixedLenFeature([MAX_NUM_ENTITIES], tf.string),
    'shape': tf.FixedLenFeature([MAX_NUM_ENTITIES], tf.string),
    'color': tf.FixedLenFeature([MAX_NUM_ENTITIES], tf.string),
    'visibility': tf.FixedLenFeature([MAX_NUM_ENTITIES], tf.float32),
}


def _decode(example_proto):
  # Parse the input `tf.Example` proto using the feature description dict above.
  single_example = tf.parse_single_example(example_proto, features)
  for k in BYTE_FEATURES:
    single_example[k] = tf.squeeze(tf.decode_raw(single_example[k], tf.uint8),
                                   axis=-1)
  return single_example


def dataset(tfrecords_path, read_buffer_size=None, map_parallel_calls=None):
  """Read, decompress, and parse the TFRecords file.

  Args:
    tfrecords_path: str. Path to the dataset file.
    read_buffer_size: int. Number of bytes in the read buffer. See documentation
      for `tf.data.TFRecordDataset.__init__`.
    map_parallel_calls: int. Number of elements decoded asynchronously in
      parallel. See documentation for `tf.data.Dataset.map`.

  Returns:
    An unbatched `tf.data.TFRecordDataset`.
  """
  raw_dataset = tf.data.TFRecordDataset(
      tfrecords_path, compression_type=COMPRESSION_TYPE,
      buffer_size=read_buffer_size)
  return raw_dataset.map(_decode, num_parallel_calls=map_parallel_calls)

i = 0

data_dict = {'train': 70_000,
             'val': 15_000,
             'test': 14_998}

dataset_name = 'clevr_with_masks_6'
current_dir = '/mnt/data/users_data/smirnov/quantised_sa/datasets/sa_clevr'
dataset_path = '/mnt/data/users_data/smirnov/quantised_sa/datasets/clevr_with_masks_train.tfrecords'
ds = iter(dataset(dataset_path))


# for name, l in data_dict.items():
#   print(f"{name} started")
#   images = np.empty((0, 3, 240, 320), dtype=np.uint8)
#   masks = np.empty((0, 11, 1, 240, 320), dtype=np.uint8)
#   visibility = np.empty((0, 11), dtype=float)
#   for i in range(l):
#     if (i + 1) % 1_000 == 0:
#       print(i + 1)
#     try:
#       d = dict(next(ds))
#     except StopIteration:
#       print(i)
#       break
#
#     if sum(d['visibility'].numpy()) > 7:
#         continue
#     images = np.vstack((images, np.expand_dims(d['image'].numpy().transpose(2, 0, 1), axis=0)))
#     masks = np.vstack((masks, np.expand_dims(d['mask'].numpy().transpose(0, 3, 1, 2), axis=0)))
#     visibility = np.vstack((visibility, np.expand_dims(d['visibility'].numpy(), axis=0)))


for name, l in data_dict.items():
  print(f"{name} started")
  images = np.zeros((l, 3, 240, 320), dtype=np.uint8)
  masks = np.zeros((l, 11, 1, 240, 320), dtype=np.uint8)
  visibility = np.zeros((l, 11), dtype=float)

  count = 0
  for i in range(l):
    if (i + 1) % 1_000 == 0:
      print(i + 1)
      print(count)
    try:
      d = dict(next(ds))
    except StopIteration:
      print(i)
      break

    if sum(d['visibility'].numpy()) > 7:
        continue
    images[count] = d['image'].numpy().transpose(2, 0, 1)
    masks[count] = d['mask'].numpy().transpose(0, 3, 1, 2)
    visibility[count] = d['visibility'].numpy()
    count+=1

  # check the directory does not exist
  save_path = os.path.join(current_dir, dataset_name, dataset_name + '_' + name)
  if not (os.path.exists(save_path)):
      # create the directory you want to save to
      os.makedirs(save_path)

  # write the file in the new directory
  if name == 'train':
      np.savez(save_path, images=images[:count], visibility=visibility[:count])
  else:
      np.savez(save_path, images=images[:count], masks=masks[:count], visibility=visibility[:count])
  item = next(iter(ds))

print("Done")