import numpy as np
import cv2
from tqdm import tqdm

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