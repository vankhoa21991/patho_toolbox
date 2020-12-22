import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import *
from tensorflow.keras.applications import *
from tensorflow.keras.callbacks import *
from tensorflow.keras.optimizers import Adam, Nadam
from tensorflow.keras.metrics import *
from tensorflow.keras.losses import binary_crossentropy
from keras.utils.generic_utils import get_custom_objects
from tensorflow.keras import backend as K


np.random.seed(13)
tf.random.set_seed(13)


def squeeze_excite_block(inputs, ratio=8):
    init = inputs
    channel_axis = -1
    filters = init.shape[channel_axis]
    se_shape = (1, 1, filters)

    se = GlobalAveragePooling2D()(init)
    se = Reshape(se_shape)(se)
    se = Dense(filters // ratio, activation='relu', kernel_initializer='he_normal', use_bias=False)(se)
    se = Dense(filters, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(se)

    x = Multiply()([init, se])
    return x


def conv_block(inputs, filters):
    x = inputs

    x = Conv2D(filters, (3, 3), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(filters, (3, 3), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = squeeze_excite_block(x)

    return x


def encoder1(inputs):
    skip_connections = []

    model = VGG19(include_top=False, weights='imagenet', input_tensor=inputs)
    names = ["block1_conv2", "block2_conv2", "block3_conv4", "block4_conv4"]
    for name in names:
        skip_connections.append(model.get_layer(name).output)

    output = model.get_layer("block5_conv4").output
    return output, skip_connections


def decoder1(inputs, skip_connections):
    num_filters = [256, 128, 64, 32]
    skip_connections.reverse()
    x = inputs
    shape = x.shape

    for i, f in enumerate(num_filters):
        x = Conv2DTranspose(shape[3], (2, 2), activation="relu", strides=(2, 2))(x)
        x = Concatenate()([x, skip_connections[i]])
        x = conv_block(x, f)

    return x


def encoder2(inputs):
    num_filters = [32, 64, 128, 256]
    skip_connections = []
    x = inputs

    for i, f in enumerate(num_filters):
        x = conv_block(x, f)
        skip_connections.append(x)
        x = MaxPool2D((2, 2))(x)

    return x, skip_connections


def decoder2(inputs, skip_1, skip_2):
    num_filters = [256, 128, 64, 32]
    skip_2.reverse()
    x = inputs
    shape = x.shape

    for i, f in enumerate(num_filters):
        x = Conv2DTranspose(shape[3], (2, 2), activation="relu", strides=(2, 2))(x)
        x = Concatenate()([x, skip_1[i], skip_2[i]])
        x = conv_block(x, f)

    return x


def output_block(inputs):
    x = Conv2D(1, (1, 1), padding="same")(inputs)
    x = Activation('sigmoid')(x)
    return x


def ASPP(x, filter):
    shape = x.shape

    y1 = AveragePooling2D(pool_size=(shape[1], shape[2]))(x)
    y1 = Conv2D(filter, 1, padding="same")(y1)
    y1 = BatchNormalization()(y1)
    y1 = Activation("relu")(y1)
    shape2 = y1.shape

    y1 = Conv2DTranspose(shape2[3], (8, 8), activation="relu", strides=(shape[1], shape[2]))(y1)

    y2 = Conv2D(filter, 1, dilation_rate=1, padding="same", use_bias=False)(x)
    y2 = BatchNormalization()(y2)
    y2 = Activation("relu")(y2)

    y3 = Conv2D(filter, 3, dilation_rate=6, padding="same", use_bias=False)(x)
    y3 = BatchNormalization()(y3)
    y3 = Activation("relu")(y3)

    y4 = Conv2D(filter, 3, dilation_rate=12, padding="same", use_bias=False)(x)
    y4 = BatchNormalization()(y4)
    y4 = Activation("relu")(y4)

    y5 = Conv2D(filter, 3, dilation_rate=18, padding="same", use_bias=False)(x)
    y5 = BatchNormalization()(y5)
    y5 = Activation("relu")(y5)

    y = Concatenate()([y1, y2, y3, y4, y5])

    y = Conv2D(filter, 1, dilation_rate=1, padding="same", use_bias=False)(y)
    y = BatchNormalization()(y)
    y = Activation("relu")(y)

    return y

strategy = tf.distribute.MirroredStrategy()
with strategy.scope():
    def dice_coeff(y_true, y_pred):
        # add epsilon to avoid a divide by 0 error in case a slice has no pixels set
        # we only care about relative value, not absolute so this alteration doesn't matter
        _epsilon = 10 ** -7
        intersections = tf.reduce_sum(y_true * y_pred)
        unions = tf.reduce_sum(y_true + y_pred)
        dice_scores = (2.0 * intersections + _epsilon) / (unions + _epsilon)
        return dice_scores


    def dice_loss(y_true, y_pred):
        loss = 1 - dice_coeff(y_true, y_pred)
        return loss


    def iou(y_true, y_pred):
        def f(y_true, y_pred):
            intersection = (y_true * y_pred).sum()
            union = y_true.sum() + y_pred.sum() - intersection
            x = (intersection + smooth) / (union + smooth)
            x = x.astype(np.float32)
            return x

        return tf.numpy_function(f, [y_true, y_pred], tf.float32)


    def bce_dice_loss(y_true, y_pred):
        return binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)


    #     def focal_loss(y_true, y_pred):
    #         alpha=0.25
    #         gamma=2
    #         def focal_loss_with_logits(logits, targets, alpha, gamma, y_pred):
    #             weight_a = alpha * (1 - y_pred) ** gamma * targets
    #             weight_b = (1 - alpha) * y_pred ** gamma * (1 - targets)
    #             return (tf.math.log1p(tf.exp(-tf.abs(logits))) + tf.nn.relu(-logits)) * (weight_a + weight_b) + logits * weight_b

    #         y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(), 1 - tf.keras.backend.epsilon())
    #         logits = tf.math.log(y_pred / (1 - y_pred))
    #         loss = focal_loss_with_logits(logits=logits, targets=y_true, alpha=alpha, gamma=gamma, y_pred=y_pred)
    #         # or reduce_sum and/or axis=-1
    #         return tf.reduce_mean(loss)

    def tversky(y_true, y_pred, smooth=1, alpha=0.7):
        y_true_pos = tf.reshape(y_true, [-1])
        y_pred_pos = tf.reshape(y_pred, [-1])
        true_pos = tf.reduce_sum(y_true_pos * y_pred_pos)
        false_neg = tf.reduce_sum(y_true_pos * (1 - y_pred_pos))
        false_pos = tf.reduce_sum((1 - y_true_pos) * y_pred_pos)
        return (true_pos + smooth) / (true_pos + alpha * false_neg + (1 - alpha) * false_pos + smooth)


    def tversky_loss(y_true, y_pred):
        return 1 - tversky(y_true, y_pred)


    def focal_tversky_loss(y_true, y_pred, gamma=0.75):
        tv = tversky(y_true, y_pred)
        return K.pow((1 - tv), gamma)


    get_custom_objects().update({"dice": dice_loss})

def build_model():
    inputs = Input((512, 512, 3))
    x, skip_1 = encoder1(inputs)
    x = ASPP(x, 64)
    x = decoder1(x, skip_1)
    outputs1 = output_block(x)

    x = inputs * outputs1

    x, skip_2 = encoder2(x)
    x = ASPP(x, 64)
    x = decoder2(x, skip_1, skip_2)
    outputs2 = output_block(x)
    outputs = Concatenate()([outputs1, outputs2])

    combine_output = Conv2D(1, (64, 64), activation="sigmoid", padding="same")(outputs)

    model = Model(inputs, combine_output)
    return model