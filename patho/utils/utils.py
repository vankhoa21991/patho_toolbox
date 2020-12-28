import numpy as np
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt
import torch
import random
import os
from tqdm import tqdm

def set_all_seeds(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def enc2mask(encs, shape):
    '''

    :param encs:
    :param shape:
    :return:
    '''
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for m,enc in enumerate(encs):
        if isinstance(enc,np.float) and np.isnan(enc): continue
        s = enc.split()
        for i in range(len(s)//2):
            start = int(s[2*i]) - 1
            length = int(s[2*i+1])
            img[start:start+length] = 1 + m
    return img.reshape(shape).T

def mask2enc(mask, n=1):
    '''

    :param mask:
    :param n:
    :return:
    '''
    pixels = mask.T.flatten()
    encs = []
    for i in range(1,n+1):
        p = (pixels == i).astype(np.int8)
        if p.sum() == 0: encs.append(np.nan)
        else:
            p = np.concatenate([[0], p, [0]])
            runs = np.where(p[1:] != p[:-1])[0] + 1
            runs[1::2] -= runs[::2]
            encs.append(' '.join(str(x) for x in runs))
    return encs

def read_single(img_path, msk_path):
    """ Read the image and mask from the given path. """
    image = cv2.imread(img_path, cv2.IMREAD_COLOR)
    mask = cv2.imread(msk_path, cv2.IMREAD_GRAYSCALE)
    return image, mask

def read_data(image_paths, mask_paths, gloms_only=False):
    '''

    :param image_paths:
    :param mask_paths:
    :param gloms_only:
    :return:
    '''
    images = []
    masks = []

    for img_path, msk_path in tqdm(zip(image_paths, mask_paths), total=len(image_paths)):

        image, mask = read_single(img_path, msk_path)
        mask_density = np.count_nonzero(mask)
        if gloms_only:
            if(mask_density>0):
                images.append(image)
                masks.append(mask)
        else:
            images.append(image)
            masks.append(mask)

    images = np.array(images)
    masks = np.array(masks)
    print('images shape:', images.shape)
    print('masks shape:', masks.shape)
    return images, masks


def rle_encode_less_memory(img):
    pixels = img.T.flatten()
    pixels[0] = 0
    pixels[-1] = 0
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 2
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)


def make_grid(shape, window=256, min_overlap=32):
    """
        Return Array of size (N,4), where N - number of tiles,
        2nd axis represente slices: x1,x2,y1,y2
    """
    x, y = shape
    nx = x // (window - min_overlap) + 1
    x1 = np.linspace(0, x, num=nx, endpoint=False, dtype=np.int64)
    x1[-1] = x - window
    x2 = (x1 + window).clip(0, x)
    ny = y // (window - min_overlap) + 1
    y1 = np.linspace(0, y, num=ny, endpoint=False, dtype=np.int64)
    y1[-1] = y - window
    y2 = (y1 + window).clip(0, y)
    slices = np.zeros((nx, ny, 4), dtype=np.int64)

    for i in range(nx):
        for j in range(ny):
            slices[i, j] = x1[i], x2[i], y1[j], y2[j]
    return slices.reshape(nx * ny, 4)

def check_lr(DEMO_EPOCHS, demo_optim, demo_sched):
    lrs = []

    for i in range(DEMO_EPOCHS):
        demo_optim.step()
        lrs.append(demo_optim.param_groups[0]["lr"])
        demo_sched.step()

    plt.plot(lrs)
    plt.show()