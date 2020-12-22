from patho.cfg.config import path_cfg,img_proc_cfg
from patho.utils.utils import read_single, read_data
import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import *
from tensorflow.keras.applications import *
from tensorflow.keras.callbacks import *
from tensorflow.keras.optimizers import Adam, Nadam
from tensorflow.keras.metrics import *
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras import backend as K
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam

from patho.model.model_keras import build_model,dice_coeff,bce_dice_loss,tversky_loss,focal_tversky_loss
from patho.utils.tfrecord import get_dataset
from patho.utils.tfrecord import read_dataset
from patho.cfg.config import path_cfg,img_proc_cfg

def train(train_dataset, validation_dataset,output_dir):
    strategy = tf.distribute.MirroredStrategy()

    with strategy.scope():
        metrics = [
            dice_coeff,
            #        iou,
            bce_dice_loss,
            #        focal_loss,
            Recall(),
            Precision(),
            tversky_loss,
            focal_tversky_loss
        ]

        callbacks = [
            ModelCheckpoint(output_dir + '/hubmap-model-1.h5', verbose=1),
            ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10),
            CSVLogger(output_dir + "/data.csv"),
            #    TensorBoard(),
            EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=False)
        ]

        model = build_model()
        model.summary(line_length=150)
        model.compile(optimizer=Adam(lr=1e-3), loss='dice', metrics=metrics)


        train_steps = round((47703 // 32) * 0.70)
        validation_steps = round((5829 // 32) * 0.70)

        model.fit(train_dataset, epochs=20, steps_per_epoch=train_steps,
                  validation_data=validation_dataset, validation_steps=validation_steps,
                  callbacks=callbacks)

        model.save_weights(output_dir + "/hubmap_model_1.h5")

if __name__ == '__main__':
    tileProcCfg = img_proc_cfg()
    path_cfg = path_cfg()

    train_tf_gcs = path_cfg.DATA_TFRECORD + '/train/*.tfrecords'
    val_tf_gcs = path_cfg.DATA_TFRECORD + '/val/*.tfrecords'
    train_tf_files = tf.io.gfile.glob(train_tf_gcs)
    val_tf_files = tf.io.gfile.glob(val_tf_gcs)
    print(val_tf_files[:3])
    print("Train TFrecord Files:", len(train_tf_files))
    print("Val TFrecord Files:", len(val_tf_files))

    train_dataset = read_dataset(train_tf_files[15])
    validation_dataset = read_dataset(val_tf_files[15])

    train_dataset = get_dataset(train_tf_files, shuf=20000)
    validation_dataset = get_dataset(val_tf_files,shuf=5000)

    train(train_dataset, validation_dataset,output_dir=path_cfg.RESULTS_DIR)