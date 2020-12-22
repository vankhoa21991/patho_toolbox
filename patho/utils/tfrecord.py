import tensorflow as tf
import numpy as np


image_feature_description = {
    'height': tf.io.FixedLenFeature([], tf.int64),
    'width': tf.io.FixedLenFeature([], tf.int64),
    'num_channels': tf.io.FixedLenFeature([], tf.int64),
    'img_bytes': tf.io.FixedLenFeature([], tf.string),
    'mask': tf.io.FixedLenFeature([], tf.string),
}

image_feature_description = {
    'height': tf.io.FixedLenFeature([], tf.int64),
    'width': tf.io.FixedLenFeature([], tf.int64),
    'num_channels': tf.io.FixedLenFeature([], tf.int64),
    'img_bytes': tf.io.FixedLenFeature([], tf.string),
    'mask': tf.io.FixedLenFeature([], tf.string),
}


def _parse_image_and_masks_function(example_proto):
    single_example = tf.io.parse_single_example(example_proto, image_feature_description)
    img_bytes = tf.io.decode_raw(single_example['img_bytes'], out_type='uint8')
    img_array = tf.reshape(img_bytes, (512, 512, 3))
    mask_bytes = tf.io.decode_raw(single_example['mask'], out_type='bool')
    mask = tf.reshape(mask_bytes, (512, 512, 1))

    ## normalize images array and cast image and mask to float32
    #     img_array = tf.cast(img_array, tf.float32) / 255.0
    #     mask = tf.cast(mask, tf.float32)
    return img_array, mask


def read_dataset(storage_file_path):
    encoded_image_dataset = tf.data.TFRecordDataset(storage_file_path, compression_type="GZIP")
    parsed_image_dataset = encoded_image_dataset.map(_parse_image_and_masks_function)
    return parsed_image_dataset

def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
  """Returns a float_list from a float / double."""
  return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature(value):
  """Returns an int64_list from a bool / enum / int / uint."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def image_example(image, mask):
    image_shape = image.shape
    img_bytes = image.tostring()
    mask_bytes = mask.tostring()
    feature = {
        'height': _int64_feature(image_shape[0]),
        'width': _int64_feature(image_shape[1]),
        'num_channels': _int64_feature(image_shape[2]),
        'img_bytes': _bytes_feature(img_bytes),
        'mask' : _bytes_feature(mask_bytes),
    }
    return tf.train.Example(features=tf.train.Features(feature=feature))

def create_tfrecord(image, mask, output_path):
    opts = tf.io.TFRecordOptions(compression_type="GZIP")
    with tf.io.TFRecordWriter(output_path, opts) as writer:
        tf_example = image_example(image, mask)
        writer.write(tf_example.SerializeToString())
    writer.close()

def _parse_image_function(example_proto):
    single_example = tf.io.parse_single_example(example_proto, image_feature_description)
    image = tf.reshape( tf.io.decode_raw(single_example['img_bytes'],out_type='uint8'), (512, 512, 3))
    mask =  tf.reshape(tf.io.decode_raw(single_example['mask'],out_type='bool'),(512, 512, 1))
    ## normalize images array and cast image and mask to float32
    image = tf.cast(image, tf.float32) / 255.0
    mask = tf.cast(mask, tf.float32)
    return image, mask

def load_dataset(filenames, ordered=False):
    AUTO = tf.data.experimental.AUTOTUNE
    ignore_order = tf.data.Options()
    if not ordered:
        ignore_order.experimental_deterministic = False
    dataset = tf.data.TFRecordDataset(filenames, num_parallel_reads=AUTO, compression_type="GZIP")
    dataset = dataset.with_options(ignore_order)
    dataset = dataset.map(_parse_image_function, num_parallel_calls=AUTO)
    return dataset

def get_dataset(tf_files, shuf=20000, bs=32):
    AUTO = tf.data.experimental.AUTOTUNE
    dataset = load_dataset(tf_files)
    #dataset = dataset.repeat()
    dataset = dataset.shuffle(shuf)
    dataset = dataset.batch(bs, drop_remainder=True)
    dataset = dataset.prefetch(AUTO)
    return dataset
