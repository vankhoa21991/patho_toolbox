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
from kaggle_datasets import KaggleDatasets
from kaggle_secrets import UserSecretsClient

from sklearn.model_selection import train_test_split
from tensorflow.keras.losses import binary_crossentropy
import plotly
import plotly.graph_objs as go
import numpy as np   # So we can use random numbers in examples