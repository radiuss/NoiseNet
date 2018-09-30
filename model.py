__author__ = "Mykhail L. Uss"
__copyright__ = "Copyright 2018, Mykhail L. Uss"
__credits__ = ["Mykhail L. Uss"]
__license__ = "GPLv3"
__version__ = "1.0.1"
__maintainer__ = "Mykhail L. Uss"
__email__ = "mykhail.uss@gmail.com", "uss@xai.edu.ua"
__status__ = "Prototype"

import numpy as np
from keras.models import Model
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.core import Dense, Flatten, Lambda
from keras.layers import Input, concatenate, Add
from keras import backend as K
from keras import regularizers

K.set_image_data_format('channels_first')
np.set_printoptions(suppress=True)


# Defaults
IMG_SIZE = 32  # input image patch size
NUM_CHANNELS = 1  # number of channels. Grayscale patch
SIGMA_EPSILON = 0.001  # positive bias to prevent division by zero uncertainty quantifier output
LAMBDA_REG = 0.025  # L2 regularization


def u_l2(y_true, y_pred):
    """ Custom loss function for regression with uncertainty estimation

    :param y_true: ground truth noise SD values
    :param y_pred: predicted noise SD values and
                   relative SD values estimation error SD prediction
    :return: loss function value
    """
    sigma_rel = y_pred[:, 1, None]  # predicted value of relative SD of SD estimation error
    sigma = sigma_rel * y_true  # predicted value of absolute SD of SD estimation error
    squared_rel_error = K.square((y_pred[:, 0, None] - y_true) / sigma)
    return K.mean(squared_rel_error + 2. * K.log(sigma), axis=-1)


def m_uncert(y_true, y_pred):
    """Custom metric:  mean predicted value of relative SD of SD estimation error

    :param y_true: ground truth noise SD values
    :param y_pred: predicted noise SD values and
                   relative SD values estimation error SD prediction
    :return: mean predicted value of relative SD of SD estimation error
    """
    sigma_rel = y_pred[:, 1, None]  # predicted value of relative SD of SD estimation error
    return K.mean(sigma_rel, axis=-1)


def err_rel(y_true, y_pred):
    """ Custom metric: mean value of relative noise SD estimation error

    :param y_true: ground truth noise SD values
    :param y_pred: predicted noise SD values and
                   relative SD values estimation error SD prediction
    :return: mean value of relative noise SD estimation error (in percents)
    """
    sigma_rel = y_pred[:, 1, None]  # predicted value of relative SD of SD estimation error
    squared_error = K.square(y_pred[:, 0, None] / y_true - 1.0)  # relative noise SD estimation error
    return 100.0 * K.sum(squared_error / sigma_rel, axis=-1) / K.sum(1.0 / sigma_rel, axis=-1)


def err_norm_sd(y_true, y_pred):
    """ Custom metric: mean value of normalized noise SD estimation error
                       For a perfect uncertainty quantifier, err_norm_sd should be equal to unity

    :param y_true: ground truth noise SD values
    :param y_pred: predicted noise SD values and
                   relative SD values estimation error SD prediction
    :return: mean value of normalized noise SD estimation error
    """
    sigma_rel = y_pred[:, 1, None]  # predicted value of relative SD of SD estimation error
    sigma = sigma_rel * y_true  # predicted value of absolute SD of SD estimation error
    squared_rel_error = K.square((y_pred[:, 0, None] - y_true) / sigma)  # normalized relative noise SD estimation error
    return K.sum(squared_rel_error / sigma_rel, axis=-1) / K.sum(1.0 / sigma_rel, axis=-1)


def bias(y_true, y_pred):
    """ Custom metric: mean value of noise SD estimation error bias

    :param y_true: ground truth noise SD values
    :param y_pred: predicted noise SD values and
                   relative SD values estimation error SD prediction
    :return: mean value of noise SD estimation error bias (in percents)
    """
    sigma_rel = y_pred[:, 1, None]  # predicted value of relative SD of SD estimation error
    sigma = sigma_rel * y_true
    mu = y_pred[:, 0, None]
    k = K.mean(mu * y_true / sigma ** 2.) / K.mean(K.square(y_true / sigma))
    return 100.*(k - 1.)


def noise_net_model():
    """ NoiseNet model

    :return: model
    """
    inp = Input(shape=(NUM_CHANNELS, IMG_SIZE, IMG_SIZE))

    # feature extractor
    x = Conv2D(32, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(LAMBDA_REG))(inp)
    x = Conv2D(32, (3, 3), activation='relu', kernel_regularizer=regularizers.l2(LAMBDA_REG))(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(LAMBDA_REG))(x)
    x = Conv2D(64, (3, 3), activation='relu', kernel_regularizer=regularizers.l2(LAMBDA_REG))(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(LAMBDA_REG))(x)
    x = Conv2D(128, (3, 3), activation='relu', kernel_regularizer=regularizers.l2(LAMBDA_REG))(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Flatten()(x)

    # regressor
    x_regressor = Dense(512, activation='relu', kernel_regularizer=regularizers.l2(LAMBDA_REG))(x)
    sd_hat = Dense(1, activation='relu')(x_regressor)

    # uncertainty quantifier
    sd_hat_extern = Input(shape=(1,))
    x_quantifier = Dense(512, activation='relu', kernel_regularizer=regularizers.l2(LAMBDA_REG))(x)
    y_extern = Dense(512, activation='relu', kernel_regularizer=regularizers.l2(LAMBDA_REG))(sd_hat_extern)
    x_quantifier = Add()([y_extern, x_quantifier])
    sigma_rel = Dense(1, activation='relu')(x_quantifier)
    sigma_rel = Lambda(lambda y: y + SIGMA_EPSILON)(sigma_rel)

    predictions = concatenate([sd_hat, sigma_rel])
    model = Model(inputs=[inp, sd_hat_extern], outputs=predictions)

    return model
