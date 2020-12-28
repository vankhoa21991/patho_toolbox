from tqdm import tqdm
import tensorflow as tf
import numpy as np
import cv2
import glob
import gc
import rasterio
import pathlib
import pandas as pd
from rasterio.windows import Window
from tensorflow.keras.layers import *
from tensorflow.keras.applications import *
from tensorflow.keras.callbacks import *
from tensorflow.keras.metrics import *
from tensorflow.keras.optimizers import Adam

from patho.model.model_keras import build_model,dice_coeff,bce_dice_loss,tversky_loss,focal_tversky_loss
from patho.utils.tfrecord import get_dataset, read_dataset
from patho.utils.utils import rle_encode_less_memory, make_grid
from patho.cfg.config import path_cfg,img_proc_cfg

def inference(model_path,data_path, WINDOW=512, MIN_OVERLAP=0, NEW_SIZE=512, THRESHOLD=0.4):
    identity = rasterio.Affine(1, 0, 0, 0, 1, 0)
    p = pathlib.Path(data_path)

    model = tf.keras.models.load_model(model_path, compile=False)
    model.summary(line_length=150)

    subm = {}
    strategy = tf.distribute.MirroredStrategy()

    with strategy.scope():
        for i, filename in tqdm(enumerate(p.glob('test/*.tiff')),
                                total=len(list(p.glob('test/*.tiff')))):

            print(f'{i + 1} Predicting {filename.stem}')

            if i < 3:
                continue

            dataset = rasterio.open(filename.as_posix(), transform=identity)
            slices = make_grid(dataset.shape, window=WINDOW, min_overlap=MIN_OVERLAP)
            preds = np.zeros(dataset.shape, dtype=np.uint8)

            for j,(x1, x2, y1, y2) in tqdm(enumerate(slices)):
                image = dataset.read([1, 2, 3],
                                     window=Window.from_slices((x1, x2), (y1, y2)))
                image = np.moveaxis(image, 0, -1)
                image = cv2.resize(image, (NEW_SIZE, NEW_SIZE), interpolation=cv2.INTER_AREA)
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                image = np.expand_dims(image, 0)

                pred = np.squeeze(model.predict(image))
                pred = cv2.resize(pred, (y2-y1, x2-x1))

                preds[x1:x2, y1:y2] += (pred > THRESHOLD).astype(np.uint8)

                if j > 10:
                    break

            preds = (preds > 0.5).astype(np.uint8)
            print(np.max(preds))
            subm[i] = {'id': filename.stem, 'predicted': rle_encode_less_memory(preds)}
            # print(np.sum(preds))
            del preds
            gc.collect();

    return subm

if __name__ == '__main__':
    tileProcCfg = img_proc_cfg()
    path_cfg = path_cfg()

    subm = inference(path_cfg.RESULTS_DIR + '/hubmap-model-1.h5', path_cfg.DATA_DIR)

    submission = pd.DataFrame.from_dict(subm, orient='index')
    submission.to_csv(path_cfg.RESULTS_DIR + '/submission.csv', index=False)
    print(submission.head())