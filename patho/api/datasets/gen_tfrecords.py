# https://www.kaggle.com/sreevishnudamodaran/tpu-hubmap-double-u-net-model-augmentation

import os
import glob
import tensorflow as tf
from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from patho.cfg.config import path_cfg,img_proc_cfg
from patho.utils.utils import read_single, read_data
from patho.utils.plot import plot_tfrecord
from patho.utils.tfrecord import image_example,read_dataset



def write_dataset(img_files, msk_files, records_per_part, prefix):
    opts = tf.io.TFRecordOptions(compression_type="GZIP")
    part_num = 0
    num_records = 0
    output_path = prefix + '_part{}.tfrecords'.format(part_num)
    writer = tf.io.TFRecordWriter(output_path, opts)

    for img_file, msk_file in tqdm(zip(img_files, msk_files), total=len(img_files), position=0, leave=True):
        image, mask = read_single(img_file, msk_file)
        assert image.shape == (512, 512, 3), print("Wrong image.shape", image.shape)
        assert mask.shape == (512, 512), print("mask.shape", mask.shape)
        # print("image.shape", image.shape)
        mask = np.expand_dims(mask, axis=-1)
        tf_example = image_example(image, mask)
        writer.write(tf_example.SerializeToString())
        num_records += 1
        if (num_records == records_per_part - 1):
            # close current file and open new one
            print("wrote part #{}".format(part_num))
            writer.close()
            part_num += 1
            output_path = prefix + '_part{}.tfrecords'.format(part_num)
            writer = tf.io.TFRecordWriter(output_path, opts)
            num_records = 0
    writer.close()

if __name__ == '__main__':
    tileProcCfg = img_proc_cfg()
    path_cfg = path_cfg()

    if not os.path.exists(path_cfg.DATA_TFRECORD):
        os.makedirs(path_cfg.DATA_TFRECORD)

    aug_img_paths = glob.glob(path_cfg.DATA_512_AUG_DIR + "/images_aug/*.png")
    aug_msk_paths = glob.glob(path_cfg.DATA_512_AUG_DIR + "/masks_aug/*.png")
    aug_img_paths2 = glob.glob(path_cfg.DATA_512_AUG_DIR + "/images_aug2/*.png")
    aug_msk_paths2 = glob.glob(path_cfg.DATA_512_AUG_DIR + "/masks_aug2/*.png")

    aug_img_paths.extend(aug_img_paths2)
    aug_msk_paths.extend(aug_msk_paths2)

    print("Number of Augmented Images", len(aug_img_paths))
    print("Number of Augmented Masks", len(aug_msk_paths))

    val_files = pd.read_csv(path_cfg.DATA_512_AUG_DIR + '/validation_files.csv')
    print(val_files.sample(3))
    val_files['image_val_files'] = val_files['image_val_files'].apply(lambda x: path_cfg.DATA_512_IMG_DIR + '/' + x.split('/')[-1])
    val_files['mask_val_files'] = val_files['mask_val_files'].apply(
        lambda x: path_cfg.DATA_512_MASK_DIR + '/' + x.split('/')[-1])

    if input('Write new train tfrecord? y/n \n').upper() == 'Y':
        print("Writing Train Dataset")
        write_dataset(aug_img_paths, aug_msk_paths, 256, path_cfg.DATA_TFRECORD + '/train/')

    if input('Write new val tfrecord? y/n \n').upper() == 'Y':
        print("Writing Validation Dataset")

        image_val_files = val_files['image_val_files'].tolist()
        mask_val_files = val_files['mask_val_files'].tolist()
        print("Total Val Image Count:", len(image_val_files))
        print("Total Val Mask Count:", len(mask_val_files))
        print(image_val_files[:2])
        write_dataset(image_val_files, mask_val_files, 256, path_cfg.DATA_TFRECORD + '/val/')


    train_tf_gcs = path_cfg.DATA_TFRECORD + '/train/*.tfrecords'
    val_tf_gcs = path_cfg.DATA_TFRECORD + '/val/*.tfrecords'
    train_tf_files = tf.io.gfile.glob(train_tf_gcs)
    val_tf_files = tf.io.gfile.glob(val_tf_gcs)
    print(val_tf_files[:3])
    print("Train TFrecord Files:", len(train_tf_files))
    print("Val TFrecord Files:", len(val_tf_files))

    train_dataset = read_dataset(train_tf_files[15])
    validation_dataset = read_dataset(val_tf_files[15])

    plot_tfrecord(train_dataset, validation_dataset)