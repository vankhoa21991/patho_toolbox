# This code is from https://www.kaggle.com/vankhoa21991/tpu-hubmap-double-u-net-model-augmentation/edit

import json
import os
import glob
import re
import datetime
import os.path as osp
from path import Path
import collections
import sys
import uuid
import random
import warnings
from itertools import chain
from tqdm import tqdm

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rcParams
sns.set(rc={"font.size":9,"axes.titlesize":15,"axes.labelsize":9,
            "axes.titlepad":2, "axes.labelpad":9, "legend.fontsize":7,
            "legend.title_fontsize":7, 'axes.grid' : False,
           'figure.titlesize':35})

# from skimage import measure
from PIL import Image
import cv2
# from skimage.io import imread, imshow, imread_collection, concatenate_images
# from skimage.transform import resize
# from skimage.morphology import label

import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv2D, Conv2DTranspose
from tensorflow.keras.layers import MaxPooling2D, UpSampling2D, Conv2DTranspose
from tensorflow.keras.layers import concatenate
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras import backend as K
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam

from keras.engine.topology import Layer
from keras.utils.generic_utils import get_custom_objects
# from kaggle_datasets import KaggleDatasets
# from kaggle_secrets import UserSecretsClient

from sklearn.model_selection import train_test_split
from tensorflow.keras.losses import binary_crossentropy
import plotly
import plotly.graph_objs as go
import numpy as np   # So we can use random numbers in examples
from patho.cfg.config import path_cfg,img_proc_cfg
from patho.utils.utils import read_single, read_data

from albumentations import (
CLAHE,
ElasticTransform,
GridDistortion,
OpticalDistortion,
HorizontalFlip,
RandomBrightnessContrast,
RandomGamma,
HueSaturationValue,
RGBShift,
MedianBlur,
GaussianBlur,
GaussNoise,
ChannelShuffle,
CoarseDropout
)

def create_df_img(image_paths, mask_paths, output_path):
    '''
    Create image dataframe, with filter images

    :param image_paths:
    :param mask_paths:
    :return:
    '''
    lowband_density_values = []
    mask_density_values = []

    for img_path, msk_path in tqdm(zip(image_paths, mask_paths), total=len(image_paths)):
        image, mask = read_single(img_path, msk_path)
        img_hist = np.histogram(image)
        # print("img_hist", img_hist)
        lowband_density = np.sum(img_hist[0][0:4])
        mask_density = np.count_nonzero(mask)
        # print("lowband_density", lowband_density)
        # print("highband_density", highband_density)
        # print("mask_density", mask_density)
        lowband_density_values.append(lowband_density)
        mask_density_values.append(mask_density)
    train_helper_df = pd.DataFrame(data=list(zip(image_paths, mask_paths, lowband_density_values,
                                                 mask_density_values)),
                                   columns=['image_path', 'mask_path', 'lowband_density', 'mask_density'])
    train_helper_df.astype(dtype={'image_path': 'object', 'mask_path': 'object',
                                  'lowband_density': 'int64', 'mask_density': 'int64'})
    train_helper_df.info()
    train_helper_df.to_csv(output_path,index=False)
    return train_helper_df

def filter_tile(df,th=100):
    df_out = df[df.lowband_density > th]

    images_tissue = df[df.lowband_density > 100].image_path
    masks_tissue = df[df.lowband_density > 100].mask_path
    return images_tissue, masks_tissue

def augment_data(image_paths, mask_paths, output_dir):

    if not os.path.exists(output_dir + '/images_aug2'):
        os.makedirs(output_dir + '/images_aug2')
    if not os.path.exists(output_dir + '/masks_aug2'):
        os.makedirs(output_dir + '/masks_aug2')

    for image, mask in tqdm(zip(image_paths, mask_paths), total=len(image_paths)):
        images_aug = []
        masks_aug = []
        image_name = Path(image).stem
        mask_name = Path(mask).stem

        x, y = read_single(image, mask)
        mask_density = np.count_nonzero(y)

        ## Augmenting only images with Gloms
        if(mask_density>0):

            try:
                h, w, c = x.shape
            except Exception as e:
                image = image[:-1]
                x, y = read_single(image, mask)
                h, w, c = x.shape

            aug = CLAHE(clip_limit=1.0, tile_grid_size=(8, 8), always_apply=False, p=1)
            augmented = aug(image=x, mask=y)
            x0 = augmented['image']
            y0 = augmented['mask']

            ## ElasticTransform
            aug = ElasticTransform(p=1, alpha=120, sigma=512*0.05, alpha_affine=512*0.03)
            augmented = aug(image=x, mask=y)
            x1 = augmented['image']
            y1 = augmented['mask']

            ## Grid Distortion
            aug = GridDistortion(p=1)
            augmented = aug(image=x, mask=y)
            x2 = augmented['image']
            y2 = augmented['mask']

            ## Optical Distortion
            aug = OpticalDistortion(p=1, distort_limit=2, shift_limit=0.5)
            augmented = aug(image=x, mask=y)
            x3 = augmented['image']
            y3 = augmented['mask']

            ## Horizontal Flip
            aug = HorizontalFlip(p=1)
            augmented = aug(image=x, mask=y)
            x4 = augmented['image']
            y4 = augmented['mask']

            ## Random Brightness and Contrast
            aug = RandomBrightnessContrast(p=1)
            augmented = aug(image=x, mask=y)
            x5 = augmented['image']
            y5 = augmented['mask']

            aug = RandomGamma(p=1)
            augmented = aug(image=x, mask=y)
            x6 = augmented['image']
            y6 = augmented['mask']

            aug = HueSaturationValue(p=1)
            augmented = aug(image=x, mask=y)
            x7 = augmented['image']
            y7 = augmented['mask']

            aug = RGBShift(p=1)
            augmented = aug(image=x, mask=y)
            x8 = augmented['image']
            y8 = augmented['mask']

            aug = MedianBlur(p=1, blur_limit=5)
            augmented = aug(image=x, mask=y)
            x9 = augmented['image']
            y9 = augmented['mask']

            aug = GaussianBlur(p=1, blur_limit=3)
            augmented = aug(image=x, mask=y)
            x10 = augmented['image']
            y10 = augmented['mask']

            aug = GaussNoise(p=1)
            augmented = aug(image=x, mask=y)
            x11 = augmented['image']
            y11 = augmented['mask']

            aug = ChannelShuffle(p=1)
            augmented = aug(image=x, mask=y)
            x12 = augmented['image']
            y12 = augmented['mask']

            aug = CoarseDropout(p=1, max_holes=8, max_height=32, max_width=32)
            augmented = aug(image=x, mask=y)
            x13 = augmented['image']
            y13 = augmented['mask']

            images_aug.extend([
                    x0, x1, x2, x3, x4, x5, x6,
                    x7, x8, x9, x10, x11, x12,
                    x13])

            masks_aug.extend([
                    y0, y1, y2, y3, y4, y5, y6,
                    y7, y8, y9, y10, y11, y12,
                    y13])

            idx = 0
            for i, m in zip(images_aug, masks_aug):
                tmp_image_name = f"{image_name}_{idx}.png"
                tmp_mask_name  = f"{mask_name}_{idx}.png"

                image_path = os.path.join(output_dir + "/images_aug2/", tmp_image_name)
                mask_path  = os.path.join(output_dir + "/masks_aug2/", tmp_mask_name)

                cv2.imwrite(image_path, i)
                cv2.imwrite(mask_path, m)

                idx += 1

    return images_aug, masks_aug

def plot_img_and_mask(images, masks):
    max_rows = 6
    max_cols = 6
    fig, ax = plt.subplots(max_rows, max_cols, figsize=(20, 18))
    fig.suptitle('Sample Images', y=0.93)
    plot_count = (max_rows * max_cols) // 2
    for idx, (img, mas) in enumerate(zip(images[:plot_count], masks[:plot_count])):
        row = (idx // max_cols) * 2
        row_masks = row + 1
        col = idx % max_cols
        ax[row, col].imshow(img)
        # sns.distplot(img_array.flatten(), ax=ax[1]);
        ax[row_masks, col].imshow(mas)
    plt.show()

if __name__ == '__main__':
    tileProcCfg = img_proc_cfg()
    path_cfg = path_cfg()

    image_paths = glob.glob(f"{path_cfg.DATA_512_IMG_DIR}/*.png")
    mask_paths = glob.glob(f"{path_cfg.DATA_512_MASK_DIR}/*.png")
    len(image_paths)

    if input('Create new df img? y/n \n').upper() == 'Y':
        train_helper_df = create_df_img(image_paths, mask_paths, path_cfg.DF_IMG_DATA_512)
    else:
        train_helper_df = pd.read_csv(path_cfg.DF_IMG_DATA_512)

    images_tissue, masks_tissue = filter_tile(train_helper_df)

    images, masks = read_data(images_tissue[1200:1218], masks_tissue[1200:1218])

    plot_img_and_mask(images, masks)

    image_90_per_tissues, image_val_files, mask_90_per_tissues, mask_val_files = train_test_split(images_tissue,
                                                                                                  masks_tissue,
                                                                                                  test_size=0.30,
                                                                                                  random_state=17)
    print(
        "Split Counts\n\tImage_90_per_files:\t{0}\n\tMask_90_per_files:\t{2}\n\tVal Images:\t\t{1}\n\tVal Masks:\t\t{3}\n"
        .format(len(image_90_per_tissues), len(image_val_files), len(mask_90_per_tissues), len(mask_val_files)))

    images_aug, masks_aug = augment_data(image_90_per_tissues, mask_90_per_tissues, path_cfg.DATA_512_AUG_DIR)
