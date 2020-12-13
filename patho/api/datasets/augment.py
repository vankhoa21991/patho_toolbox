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
from patho2.patho.utils.path import path_cfg,img_proc_cfg
from patho2.patho.utils.utils import read_single, read_data

def create_df_img(image_paths, mask_paths):
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
    return train_helper_df

if __name__ == '__main__':
    tileProcCfg = img_proc_cfg()
    path_cfg = path_cfg()

    image_paths = glob.glob(f"{path_cfg.DATA_512_IMG_DIR}/*.png")[:1000]
    mask_paths = glob.glob(f"{path_cfg.DATA_512_MASK_DIR}/*.png")[:1000]
    len(image_paths)

    create_df_img(image_paths, mask_paths)
