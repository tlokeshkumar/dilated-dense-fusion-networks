# Implementation of Deep Boosting for Image Denoising

import tensorflow as tf
from tensorflow.python.keras import layers
# from keras.layers import *
# from keras.layers import Conv2DTranspose

from tensorflow.python.keras.layers import (Activation, AveragePooling2D,
                                            BatchNormalization, Conv2D, Conv3D,
                                            Dense, Flatten,
                                            GlobalAveragePooling2D,
                                            GlobalMaxPooling2D, Input,
                                            MaxPooling2D, MaxPooling3D,
                                            Reshape, Dropout, concatenate,
                                            Conv2DTranspose, ZeroPadding2D,
                                            Subtract, Add, PReLU)

from tensorflow.python.keras import applications
from tensorflow.python.keras.models import Model
from tensorflow.python.keras import backend as K_B


def feature_extraction(x, ch='ddfn', name='fe'):
    
    if ch == 'ddfn':
        # Upper conv + dil_conv layers
        x_cu = Conv2D(8, 3, padding='same', name='fe_up_c', activation='relu')(x)
        x_du = Conv2D(8, 3, padding='same', dilation_rate=2, name='fe_up_d', activation='relu')(x_cu)

        # Lower dil_conv + conv layers
        x_dd = Conv2D(8, 3, padding='same', dilation_rate=2, name='fe_dn_d', activation='relu')(x)
        x_cd = Conv2D(8, 3, padding='same', name='fe_dn_c', activation='relu')(x_dd)
        
        return concatenate([x_cu, x_du, x_dd, x_cd])

def boost_block(x, block,  ch = 'ddfn'):
    if ch == 'ddfn':
        x_in = Conv2D(24, 1, activation='relu')(x)
        
        # Upper Conv ,Dil_conv layers
        x_up_c = Conv2D(6, 3, padding = 'same', activation='relu', name='boost_up_c_'+block)(x_in)
        x_up_d = Conv2D(6, 3, dilation_rate=2, padding='same', activation='relu', name='boost_up_d_'+block)(x_up_c)
        
        # Lower Deconv, Conv layers
        x_dn_d = Conv2D(6, 3, dilation_rate=2, padding='same', activation='relu', name='boost_dn_d_'+block)(x_in)
        x_dn_c = Conv2D(6, 3, padding='same', activation='relu', name='boost_dn_c_'+block)(x_dn_d)

        con_x = concatenate([x_up_c, x_up_d, x_dn_d, x_dn_c])

        x_out = Conv2D(8,1, padding='same', activation='relu')(con_x)

        return concatenate([x_out, x_in])

def reconstruction(x, ch='ddfn'):
    if ch == 'ddfn':
        return Conv2D(1,1,name='recons')(x)

def build_model(input_tensor=None, input_shape=None, num_boost=8,ch='ddfn'):
    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        img_input = input_tensor

    # Feature Extraction Layers
    x = feature_extraction(img_input)

    # Feature Integration Layers
    for i in range(num_boost):
        x = boost_block(x, str(i))
    
    # Reconstruction Layers
    x = reconstruction(x)

    m = Model(inputs = img_input, outputs = x)
    return m

def loss_funcs(model, labels):
    out = model.output
    mse = tf.losses.mean_squared_error(out, labels)

    with tf.name_scope('loss'):
        tf.summary.scalar('Mean', tf.reduce_mean(mse))
        tf.summary.scalar('Max', tf.reduce_max(mse))
        tf.summary.scalar('Min', tf.reduce_min(mse))
    
    with tf.name_scope('Predictions'):
        tf.summary.scalar('Mean', tf.reduce_mean(out))
        tf.summary.scalar('Max', tf.reduce_max(out))
        tf.summary.scalar('Min', tf.reduce_min(out))
    return mse

if __name__ == '__main__':
    m = build_model(input_shape=[256,256,3])
    print(m.summary())