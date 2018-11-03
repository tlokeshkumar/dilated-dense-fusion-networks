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
from tensorflow.python.keras.regularizers import  l2
#  ,kernel_regularizer=l2(l2_coeff)
def feature_extraction(x, ch='ddfn', name='fe', l2_coeff=0):
    
    if ch == 'ddfn':
        # Upper conv + dil_conv layers
        x_cu = Conv2D(8, 3, padding='same', name='fe_up_c', activation='relu',kernel_regularizer=l2(l2_coeff))(x)
        x_du = Conv2D(8, 3, padding='same', dilation_rate=2, name='fe_up_d', activation='relu',kernel_regularizer=l2(l2_coeff))(x_cu)

        # Lower dil_conv + conv layers
        x_dd = Conv2D(8, 3, padding='same', dilation_rate=2, name='fe_dn_d', activation='relu',kernel_regularizer=l2(l2_coeff))(x)
        x_cd = Conv2D(8, 3, padding='same', name='fe_dn_c', activation='relu',kernel_regularizer=l2(l2_coeff))(x_dd)
        
        return concatenate([x_cu, x_du, x_dd, x_cd])

def boost_block(x, block,  ch = 'ddfn', l2_coeff=0):
    if ch == 'ddfn':
        x_in = Conv2D(24, 1, activation='relu',kernel_regularizer=l2(l2_coeff))(x)
        
        # Upper Conv ,Dil_conv layers
        x_up_c = Conv2D(6, 3, padding = 'same', activation='relu', name='boost_up_c_'+block,kernel_regularizer=l2(l2_coeff))(x_in)
        x_up_d = Conv2D(6, 3, dilation_rate=2, padding='same', activation='relu', name='boost_up_d_'+block,kernel_regularizer=l2(l2_coeff))(x_up_c)
        
        # Lower Deconv, Conv layers
        x_dn_d = Conv2D(6, 3, dilation_rate=2, padding='same', activation='relu', name='boost_dn_d_'+block,kernel_regularizer=l2(l2_coeff))(x_in)
        x_dn_c = Conv2D(6, 3, padding='same', activation='relu', name='boost_dn_c_'+block,kernel_regularizer=l2(l2_coeff))(x_dn_d)

        con_x = concatenate([x_up_c, x_up_d, x_dn_d, x_dn_c])

        x_out = Conv2D(8,1, padding='same', activation='relu',kernel_regularizer=l2(l2_coeff))(con_x)

        return concatenate([x_out, x_in])

def reconstruction(x, ch='ddfn', l2_coeff=0):
    if ch == 'ddfn':
        return Conv2D(3,1,name='recons',kernel_regularizer=l2(l2_coeff))(x)

def build_model(input_tensor=None, input_shape=None, num_boost=8,l2_coeff=0.001, ch='ddfn'):
    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        img_input = input_tensor

    # Feature Extraction Layers
    x = feature_extraction(img_input, l2_coeff=l2_coeff)

    # Feature Integration Layers
    for i in range(num_boost):
        x = boost_block(x, str(i), l2_coeff=l2_coeff)
    
    # Reconstruction Layers
    x = reconstruction(x, l2_coeff=l2_coeff)

    m = Model(inputs = img_input, outputs = x)
    return m,m.output

def loss_funcs(model,out, labels)
    # tf.summary.image('Pred', out)

    mse = tf.losses.mean_squared_error(model,out, labels)
    # l2_loss = tf.losses.get_regularization_loss()
    # mse += l2_loss
    total_loss = mse + tf.add_n(model.losses)
    with tf.name_scope('Images'):
        tf.summary.image('Pred', model.output)
        tf.summary.image('Noise', model.input)
        tf.summary.image('ground_truth', labels)
    with tf.name_scope('loss'):
        tf.summary.scalar('Total_Loss', tf.reduce_mean(total_loss))
        tf.summary.scalar('Label_Loss', tf.reduce_mean(mse))
        tf.summary.scalar('Regularization_Loss', tf.reduce_mean(total_loss-mse))

    with tf.name_scope('Predictions'):
        mean, var = tf.nn.moments(out,axes=[0])
        tf.summary.scalar('Mean', tf.reduce_mean(mean))
        tf.summary.scalar('Max', tf.reduce_max(out))
        tf.summary.scalar('Min', tf.reduce_min(out))
        tf.summary.scalar('Var', tf.reduce_mean(var))
    return total_loss

if __name__ == '__main__':
    m = build_model(input_shape=[256,256,3])
    print(m.summary())
