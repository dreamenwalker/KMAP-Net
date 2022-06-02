# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 15:35:42 2020
@author: Dreamen
"""
from __future__ import print_function
import numpy as np
import warnings
import keras
from keras import layers
from keras.layers import Input,Dense,Activation,Flatten,Conv2D,MaxPooling2D,GlobalMaxPooling2D,ZeroPadding2D
from keras.layers import GlobalAveragePooling2D,AveragePooling2D,BatchNormalization,Lambda,Multiply
from keras.models import Model
from keras.preprocessing import image
from keras.regularizers import l1,l2,l1_l2
import keras.backend as K
import keras.layers as KL
import keras.engine as KE
import keras.models as KM
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import decode_predictions
from keras.applications.imagenet_utils import preprocess_input
from keras_applications.imagenet_utils import _obtain_input_shape
from keras.engine.topology import get_source_inputs
clinical_features = ['age','gender','cTstage','cNstage','cTNMstage']
clinical_features = []
import tensorflow as tf
# tf.test.gpu_device_name()
def L12_reg(weight_matrix):
    return None
    # return 0.01 * K.sum(K.abs(weight_matrix)) + 0.01 * K.sum(K.pow(weight_matrix,2))
def se_block(input_tensor, c=4):#c is reduction ratio
    num_channels = int(input_tensor._keras_shape[-1]) # Tensorflow backend
    bottleneck = int(num_channels // c)

    se_branch = GlobalAveragePooling2D()(input_tensor)
    se_branch = Dense(bottleneck, use_bias=False, activation='relu')(se_branch)
    se_branch = Dense(num_channels, use_bias=False, activation='sigmoid')(se_branch)

    out = Multiply()([input_tensor, se_branch])
    return out

######## build attention module
# 判断输入数据格式，是channels_first还是channels_last
channel_axis = 1 if K.image_data_format() == "channels_first" else 3
# self-define attention by liwen
from keras.layers import Lambda

def reduce_landmarks(x):
    return tf.subtract(x,tf.reduce_mean(x))
# CBAM keras https://blog.csdn.net/a2824256/article/details/108752660
def channel_attention(input_xs, reduction_ratio=0.125):
    # get channel
    channel = int(input_xs.shape[channel_axis])
    maxpool_channel = KL.GlobalMaxPooling2D()(input_xs)
    maxpool_channel = KL.Reshape((1, 1, channel))(maxpool_channel)
    avgpool_channel = KL.GlobalAvgPool2D()(input_xs)
    avgpool_channel = KL.Reshape((1, 1, channel))(avgpool_channel)
    Dense_One = KL.Dense(units=int(channel * reduction_ratio), activation='relu', kernel_initializer='he_normal', use_bias=True, bias_initializer='zeros')
    Dense_Two = KL.Dense(units=int(channel), activation='relu', kernel_initializer='he_normal', use_bias=True, bias_initializer='zeros')
    # max path
    mlp_1_max = Dense_One(maxpool_channel)
    mlp_2_max = Dense_Two(mlp_1_max)
    mlp_2_max = KL.Reshape(target_shape=(1, 1, int(channel)))(mlp_2_max)
    # avg path
    mlp_1_avg = Dense_One(avgpool_channel)
    mlp_2_avg = Dense_Two(mlp_1_avg)
    mlp_2_avg = KL.Reshape(target_shape=(1, 1, int(channel)))(mlp_2_avg)
    channel_attention_feature = KL.Add()([mlp_2_max, mlp_2_avg])
    channel_attention_feature = KL.Activation('sigmoid')(channel_attention_feature)
    return KL.Multiply()([channel_attention_feature, input_xs])

# SAM
def spatial_attention(channel_refined_feature):
    maxpool_spatial = KL.Lambda(lambda x: K.max(x, axis=3, keepdims=True))(channel_refined_feature)
    avgpool_spatial = KL.Lambda(lambda x: K.mean(x, axis=3, keepdims=True))(channel_refined_feature)
    max_avg_pool_spatial = KL.Concatenate(axis=3)([maxpool_spatial, avgpool_spatial])
    return KL.Conv2D(filters=1, kernel_size=(3, 3), padding="same", activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(max_avg_pool_spatial)

def cbam_module(input_xs, reduction_ratio=0.5):
    channel_refined_feature = channel_attention(input_xs, reduction_ratio=reduction_ratio)
    spatial_attention_feature = spatial_attention(channel_refined_feature)
    refined_feature = KL.Multiply()([channel_refined_feature, spatial_attention_feature])
    return KL.Add()([refined_feature, input_xs])
#selfdefined module 
def selfchannel_attention(input_xs, reduction_ratio=0.125): #reduction_ratio is the same as the C in the SEnet
    # get channel
    channel = int(input_xs.shape[channel_axis])
    avgpool_channel = KL.GlobalAvgPool2D()(input_xs)
    avgvalue = K.mean(avgpool_channel)
    # avgvalue = KL.Reshape((1, 1, channel))(avgvalue)
    avgpool_channel = KL.Reshape((1, 1, channel))(avgpool_channel)
    channel_attention_weighttemp =  Lambda(reduce_landmarks)(avgpool_channel)
    channel_attention_weight = KL.Activation('relu')(channel_attention_weighttemp)
    channel_attention_feature = KL.Multiply()([channel_attention_weight, input_xs])
    # 对 avg之后的feature map 进行 max 加强
    maxpool_channeltemp = KL.GlobalMaxPooling2D()(channel_attention_feature)
    maxpool_channel = KL.Reshape((1, 1, channel))(maxpool_channeltemp)
    channel_attention_feature = KL.Multiply()([maxpool_channel, input_xs])
    return KL.Activation('relu')(channel_attention_feature)
def selfspatial_attention(channel_refined_feature):
    maxpool_spatial = KL.Lambda(lambda x: K.max(x, axis=3, keepdims=True))(channel_refined_feature)
    avgpool_spatial = KL.Lambda(lambda x: K.mean(x, axis=3, keepdims=True))(channel_refined_feature)
    spaAtt7 = KL.Conv2D(filters=1, kernel_size=(7, 7), padding="same", activation='relu',
                     kernel_initializer='he_normal', use_bias=False)(channel_refined_feature)
    spaAtt3 = KL.Conv2D(filters=1, kernel_size=(3, 3), padding="same", activation='relu',
                    kernel_initializer='he_normal', use_bias=False)(channel_refined_feature)
    max_avg_77_33_pool_spatial = KL.Concatenate(axis=3)([maxpool_spatial, avgpool_spatial,spaAtt7,spaAtt3])
    # select 7*7 to focus local spatial information
    return KL.Conv2D(filters=1, kernel_size=(7, 7), padding="same", activation='sigmoid', 
                    kernel_initializer='he_normal', use_bias=False)(max_avg_77_33_pool_spatial)

def selfcbam_module(input_xs, reduction_ratio=0.5):
    channel_refined_feature = selfchannel_attention(input_xs, reduction_ratio=reduction_ratio)
    spatial_attention_feature = selfspatial_attention(channel_refined_feature)
    refined_feature = KL.Multiply()([channel_refined_feature, spatial_attention_feature])
    return KL.Add()([refined_feature, input_xs])
# CAM

def identity_block(input_tensor, kernel_size, filters, stage, block,use_bias=True, train_bn=True):
    """The identity block is the block that has no conv layer at shortcut.
    # Arguments
        input_tensor: input tensor
        kernel_size: defualt 3, the kernel size of middle conv layer at main path
        filters: list of integers, the filterss of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
    stage is phase for different
    # Returns
        Output tensor for the block.
    """
    filters1, filters2, filters3 = filters
    bn_axis = 3
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    x = Conv2D(filters1, (3, 3), strides= (1,1),padding="SAME", name=conv_name_base + '2a', kernel_initializer="he_normal",
                      kernel_regularizer=L12_reg)(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters2, kernel_size, kernel_initializer="he_normal",
               padding='same', name=conv_name_base + '2b',kernel_regularizer=L12_reg)(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    # x = Activation('relu')(x)

    # x = Conv2D(filters3, kernel_size,(1, 1), name=conv_name_base + '2c', kernel_initializer="he_normal",
    #                   kernel_regularizer=L12_reg)(x)
    # x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    x = layers.add([x, input_tensor])
    x = Activation('relu')(x)
    return x

def conv_block(input_tensor, kernel_size, filters, stage, block, strides=(2, 2), use_bias=True, train_bn=True):
    """conv_block is the block that has a conv layer at shortcut
    # Arguments
        input_tensor: input tensor
        kernel_size: defualt 3, the kernel size of middle conv layer at main path
        filters: list of integers, the filterss of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names

    # Returns
        Output tensor for the block.

    Note that from stage 3, the first conv layer at main path is with strides=(2,2)
    And the shortcut should have strides=(2,2) as well
    """
    filters1, filters2, filters3 = filters

    bn_axis = 3

    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Conv2D(filters1,  kernel_size=(3, 3),
                           padding="same",strides=strides,kernel_regularizer=L12_reg, kernel_initializer="he_normal",
               name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters2, kernel_size, padding='same', kernel_initializer="he_normal",
               name=conv_name_base + '2b',kernel_regularizer=L12_reg)(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    # x = Activation('relu')(x)# the relu function used for the subsection and input is the sum of

    # x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c', kernel_initializer="he_normal",
    #                   kernel_regularizer=L12_reg)(x)
    # x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)
    # x = se_block(x)
    # x =cbam_module(x)
    x = selfcbam_module(x)
    shortcut = Conv2D(filters3, (1, 1), strides=strides, kernel_regularizer=L12_reg, kernel_initializer="he_normal",
                      name=conv_name_base + '1')(input_tensor)
    shortcut = BatchNormalization(axis=bn_axis, name=bn_name_base + '1')(shortcut)

    x = layers.add([x, shortcut])
    x = Activation('relu')(x)
    return x

#%%
def resnet_self(input_tensor = None, include_top=True,num_outputs=1,clincal_features_size=5,
                 input_shape=(224,224,3),architecture = 'resnet50', stage5=True, train_bn=True):
    """Build a ResNet graph.
        architecture: Can be resnet50 or resnet101
        stage5: Boolean. If False, stage5 of the network is not created
        train_bn: Boolean. Train or freeze Batch Norm layers
    """
    print('******now, multitaskNet718 is using******************')
    img_input = Input(shape=input_shape,name = 'input')
    clinical_data = Input(shape=(clincal_features_size,), name = 'clinical_features')
    # assert architecture in ["resnet50", "resnet101"]
    # Stage 1
    # CC = x = KL.ZeroPadding2D((3, 3))(img_input)
    C0 = x = KL.Conv2D(32, (3, 3), strides=(1, 1), name='conv1', padding="same",use_bias=True, kernel_initializer="he_normal",
                      kernel_regularizer=L12_reg)(img_input)
    x = BatchNormalization(axis=3, name='bn_conv1')(x)
    x = KL.Activation('relu')(x)
    C1 = x = KL.MaxPooling2D((3, 3), strides=(2, 2), padding="same")(x)
    # Stage 2
    C2 = x = conv_block(x, 3, [64, 64, 64], stage=2, block='a', strides=(2, 2), train_bn=train_bn)
    # x = identity_block(x, 3, [64, 64, 256], stage=2, block='b', train_bn=train_bn)
    # C2 = x = identity_block(x, 3, [64, 64, 64], stage=2, block='c', train_bn=train_bn)
    # Stage 3
    C3 = x = conv_block(x, 3, [128, 128, 128], stage=3, block='a', train_bn=train_bn)
    # x = identity_block(x, 3, [128, 128, 512], stage=3, block='b', train_bn=train_bn)# at shortcut no conv see notebook
    # x = identity_block(x, 3, [128, 128, 512], stage=3, block='c', train_bn=train_bn)
    # C3 = x = identity_block(x, 3, [128, 128, 128], stage=3, block='d', train_bn=train_bn)
    # Stage 4
    C4 = x = conv_block(x, 3, [256, 256, 256], stage=4, block='a', train_bn=train_bn)
    # block_count = {"resnet50": 5, "resnet101": 22}[architecture] # if architecture is resnet50, the block_count is 5
    # for i in range(block_count):
    # C4 = x = identity_block(x, 3, [256, 256, 256], stage=4, block=chr(98 + i), train_bn=train_bn)#chr(98) is b
    # C4 = x
    # Stage 5
    if stage5:
        C5 =x = conv_block(x, 3, [512, 512, 512], stage=5, block='a', train_bn=train_bn)
        # x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b', train_bn=train_bn)
        # C5 = x = identity_block(x, 3, [512, 512, 512], stage=5, block='c', train_bn=train_bn)
    else:
        C5 = None
    # _, C2, C3, C4, C5 = resnet_graph(input_image=None, architecture="resnet50",stage5=True, train_bn=True)
# Top-down Layers 构建自上而下的网络结构
# 从 C5开始处理，先卷积来转换特征图尺寸
# input2 for extend ROI
    img_input2 = Input(shape=input_shape,name = 'inputExtend')
    C02 = x = KL.Conv2D(32, (3, 3), strides=(1, 1), name='conv12', padding="same",use_bias=True, kernel_initializer="he_normal",  
                      kernel_regularizer=L12_reg)(img_input2)
    x = BatchNormalization(axis=3, name='bn_conv12')(x)
    x = KL.Activation('relu')(x)
    C12 = x = KL.MaxPooling2D((3, 3), strides=(2, 2), padding="same")(x)
    # Stage 2
    C22 = x = conv_block(x, 3, [64, 64, 64], stage=22, block='a', strides=(2, 2), train_bn=train_bn)
    # x = identity_block(x, 3, [64, 64, 256], stage=2, block='b', train_bn=train_bn)
    # C2 = x = identity_block(x, 3, [64, 64, 64], stage=2, block='c', train_bn=train_bn)
    # Stage 3
    C32 = x = conv_block(x, 3, [128, 128, 128], stage=32, block='a', train_bn=train_bn)
    # x = identity_block(x, 3, [128, 128, 512], stage=3, block='b', train_bn=train_bn)# at shortcut no conv see notebook
    # x = identity_block(x, 3, [128, 128, 512], stage=3, block='c', train_bn=train_bn)
    # C3 = x = identity_block(x, 3, [128, 128, 128], stage=3, block='d', train_bn=train_bn)
    # Stage 4
    C42 = x = conv_block(x, 3, [256, 256, 256], stage=42, block='a', train_bn=train_bn)
    # block_count = {"resnet50": 5, "resnet101": 22}[architecture] # if architecture is resnet50, the block_count is 5
    # for i in range(block_count):
    # C4 = x = identity_block(x, 3, [256, 256, 256], stage=4, block=chr(98 + i), train_bn=train_bn)#chr(98) is b
    # C4 = x
    # Stage 5
    if stage5:
        C52 =x = conv_block(x, 3, [512, 512, 512], stage=52, block='a', train_bn=train_bn)
        # x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b', train_bn=train_bn)
        # C5 = x = identity_block(x, 3, [512, 512, 512], stage=5, block='c', train_bn=train_bn)
    else:
        C52 = None
    C1concat = KL.Concatenate(axis=3)([C1, C12])
    C1concat = KL.Conv2D(32, (1, 1), name='C1ConcatConv',kernel_regularizer=L12_reg)(C1concat)  # 256
    C2concat = KL.Concatenate(axis=3)([C2, C22])
    C2concat = KL.Conv2D(64, (1, 1), name='C2concatConv',kernel_regularizer=L12_reg)(C2concat)  # 256
    C3concat = KL.Concatenate(axis=3)([C3, C32])
    C3concat = KL.Conv2D(128, (1, 1), name='C3concatConv',kernel_regularizer=L12_reg)(C3concat)  # 256
    C4concat = KL.Concatenate(axis=3)([C4, C42])
    C4concat = KL.Conv2D(256, (1, 1), name='C4concatConv',kernel_regularizer=L12_reg)(C4concat)  # 256
    C5concat = KL.Concatenate(axis=3)([C5, C52])
    C5concat = KL.Conv2D(512, (1, 1), name='C5concatConv',kernel_regularizer=L12_reg)(C5concat)  # 256
    #### OS branch
    P5 = KL.Conv2D(256, (1, 1), name='fpn_c5p5',kernel_regularizer=L12_reg)(C5concat)  # 256
    P5up = KL.UpSampling2D(size=(2, 2), name="p5upsampled")(P5)
    P4 = KL.Add(name="fpn_p5addc4")([
                KL.UpSampling2D(size=(2, 2), name="fpn_p5upsampled")(P5),
                KL.Conv2D(256, (1, 1),name='fpn_c4p4')(C4concat)])#503 256 >128
    x = KL.Activation('relu')(P4)
    x = BatchNormalization(axis=3, name='bnp4')(x)
    P4 = Activation('relu')(x)
    P4 = KL.Conv2D(128, (1, 1), padding="SAME", kernel_initializer="he_normal",name="fpn_p4")(P4)
    P3 = KL.Add(name="fpn_p4addc3")([
                KL.UpSampling2D(size=(2, 2), name="fpn_p4upsampled")(P4),
                KL.Conv2D(128, (1, 1), name='fpn_c3p3')(C3concat)])

    x = KL.Activation('relu')(P3)
    x = BatchNormalization(axis=3, name='bnp3')(x)
    P3 = Activation('relu')(x)
    P3 = KL.Conv2D(64, (1, 1), padding="SAME",name="fpn_p3", kernel_initializer="he_normal")(P3)
    P2 = KL.Add(name="fpn_p3addc2")([
                KL.UpSampling2D(size=(2, 2),name="fpn_p3upsampled")(P3),
                KL.Conv2D(64, (1, 1),  kernel_initializer="he_normal",name='fpn_c2p2')(C2concat)])
    x = KL.Activation('relu')(P2)
    x = BatchNormalization(axis=3, name='bnp2')(x)
    P2 = Activation('relu')(x)
    P2 = KL.Conv2D(32, (1, 1), padding="SAME", kernel_initializer="he_normal", name="fpn_p2")(P2)
    P1 = KL.Add(name = "fpn_p2addc1OS")([KL.UpSampling2D(size=(2, 2),name="fpn_p2upsampled")(P2),
    KL.Conv2D(32, (1, 1), name='fpn_c1p1')(C1concat)])# change channel 64 to 256 for C1
    # P2-P5最后又做了一次3*3的卷积，作用是消除上采样带来的混叠效应
    # Attach 3x3 conv to all P layers to get the final feature maps.
    P1 = KL.Conv2D(32, (3, 3), padding="SAME",  kernel_initializer="he_normal",name="fpn_Convedp1")(P1)
    P2 = KL.Conv2D(32, (3, 3), padding="SAME", name="fpn_Convedp2")(P2)
    P3 = KL.Conv2D(64, (3, 3), padding="SAME", name="fpn_Convedp3")(P3)
    P4 = KL.Conv2D(128, (3, 3), padding="SAME", name="fpn_Convedp4")(P4)
    P5 = KL.Conv2D(256, (3, 3), padding="SAME", name="fpn_Convedp5")(P5)
    gpP1 = GlobalAveragePooling2D(dim_ordering='default', name='global_poolP1')(P1)
    gpP2 = GlobalAveragePooling2D(dim_ordering='default', name='global_poolP2')(P2)
    gpP3 = GlobalAveragePooling2D(dim_ordering='default', name='global_poolP3')(P3)
    gpP4 = GlobalAveragePooling2D(dim_ordering='default', name='global_poolP4')(P4)
    gpP5 = GlobalAveragePooling2D(dim_ordering='default', name='global_poolP5')(P5up)

    # for T stage
    P5Tstage = KL.Conv2D(256, (1, 1), name='fpn_c5p5Tstage',kernel_regularizer=None)(C5concat)  # 256
    P5upTstage = KL.UpSampling2D(size=(2, 2), name="p5upsampledTstage")(P5Tstage)
    P4 = KL.Add(name="fpn_p5addc4Tstage")([P5upTstage,KL.Conv2D(256, (1, 1),name='fpn_c4p4Tstage')(C4concat)])#503 256 >128
    x = KL.Activation('relu')(P4)
    x = BatchNormalization(axis=3, name='bnp4Tstage')(x)
    P4 = Activation('relu')(x)
    P4 = KL.Conv2D(128, (1, 1), padding="SAME", kernel_initializer="he_normal",name="fpn_p4Tstage")(P4)
    P3 = KL.Add(name="fpn_p4addc3Tstage")([
                KL.UpSampling2D(size=(2, 2), name="fpn_p4upsampledTstage")(P4),
                KL.Conv2D(128, (1, 1), name='fpn_c3p3Tstage')(C3concat)])

    x = KL.Activation('relu')(P3)
    x = BatchNormalization(axis=3, name='bnp3Tstage')(x)
    P3 = Activation('relu')(x)
    P3 = KL.Conv2D(64, (1, 1), padding="SAME",name="fpn_p3Tstage", kernel_initializer="he_normal")(P3)
    P2 = KL.Add(name="fpn_p3addc2Tstage")([
                KL.UpSampling2D(size=(2, 2),name="fpn_p3upsampledTstage")(P3),
                KL.Conv2D(64, (1, 1),  kernel_initializer="he_normal",name='fpn_c2p2Tstage')(C2concat)])
    x = KL.Activation('relu',name = 'ReluTstage')(P2)
    x = BatchNormalization(axis=3, name='bnp2Tstage')(x)
    P2 = Activation('relu')(x)
    P2 = KL.Conv2D(32, (1, 1), padding="SAME", kernel_initializer="he_normal", name="fpn_p2Tstage")(P2)
    P1 = KL.Add(name = "fpn_p2addc1Tstage")([KL.UpSampling2D(size=(2, 2),name="fpn_p2upsampledTstage")(P2),
    KL.Conv2D(32, (1, 1), name='fpn_c1p1Tstage')(C1concat)])# change channel 64 to 256 for C1
    # P2-P5最后又做了一次3*3的卷积，作用是消除上采样带来的混叠效应
    # Attach 3x3 conv to all P layers to get the final feature maps.
    P1Tstage = KL.Conv2D(32, (3, 3), padding="SAME",  kernel_initializer="he_normal",name="fpn_Convedp1Tstage")(P1)
    P2 = KL.Conv2D(32, (3, 3), padding="SAME", name="fpn_Convedp2Tstage")(P2)
    P3 = KL.Conv2D(64, (3, 3), padding="SAME", name="fpn_Convedp3Tstage")(P3)
    P4 = KL.Conv2D(128, (3, 3), padding="SAME", name="fpn_Convedp4Tstage")(P4)
    P5 = KL.Conv2D(256, (3, 3), padding="SAME", name="fpn_Convedp5Tstage")(P5)
    gpP1Tstage = GlobalAveragePooling2D(dim_ordering='default', name='global_poolP1Tstage')(P1Tstage)
    gpP2Tstage = GlobalAveragePooling2D(dim_ordering='default', name='global_poolP2Tstage')(P2)
    gpP3Tstage = GlobalAveragePooling2D(dim_ordering='default', name='global_poolP3Tstage')(P3)
    gpP4Tstage = GlobalAveragePooling2D(dim_ordering='default', name='global_poolP4Tstage')(P4)
    gpP5Tstage = GlobalAveragePooling2D(dim_ordering='default', name='global_poolP5Tstage')(P5upTstage)

    #Nstage
    P5Nstage = KL.Conv2D(256, (1, 1), name='fpn_c5p5Nstage',kernel_regularizer=L12_reg)(C5concat)  # 256
    P5upNstage = KL.UpSampling2D(size=(2, 2), name="p5upsampledNstage")(P5Nstage)
    P4 = KL.Add(name="fpn_p5addc4Nstage")([P5upNstage,KL.Conv2D(256, (1, 1),name='fpn_c4p4Nstage')(C4concat)])#503 256 >128
    x = KL.Activation('relu')(P4)
    x = BatchNormalization(axis=3, name='bnp4Nstage')(x)
    P4 = Activation('relu')(x)
    P4 = KL.Conv2D(128, (1, 1), padding="SAME", kernel_initializer="he_normal",name="fpn_p4Nstage")(P4)
    P3 = KL.Add(name="fpn_p4addc3Nstage")([
                KL.UpSampling2D(size=(2, 2), name="fpn_p4upsampledNstage")(P4),
                KL.Conv2D(128, (1, 1), name='fpn_c3p3Nstage')(C3concat)])

    x = KL.Activation('relu')(P3)
    x = BatchNormalization(axis=3, name='bnp3Nstage')(x)
    P3 = Activation('relu')(x)
    P3 = KL.Conv2D(64, (1, 1), padding="SAME",name="fpn_p3Nstage", kernel_initializer="he_normal")(P3)
    P2 = KL.Add(name="fpn_p3addc2Nstage")([
                KL.UpSampling2D(size=(2, 2),name="fpn_p3upsampledNstage")(P3),
                KL.Conv2D(64, (1, 1),  kernel_initializer="he_normal",name='fpn_c2p2Nstage')(C2concat)])
    x = KL.Activation('relu')(P2)
    x = BatchNormalization(axis=3, name='bnp2Nstage')(x)
    P2 = Activation('relu')(x)
    P2 = KL.Conv2D(32, (1, 1), padding="SAME", kernel_initializer="he_normal", name="fpn_p2Nstage")(P2)
    P1 = KL.Add(name = "fpn_p2addc1Nstage")([KL.UpSampling2D(size=(2, 2),name="fpn_p2upsampledNstage")(P2),
    KL.Conv2D(32, (1, 1), name='fpn_c1p1Nstage')(C1concat)])# change channel 64 to 256 for C1
    # P2-P5最后又做了一次3*3的卷积，作用是消除上采样带来的混叠效应
    # Attach 3x3 conv to all P layers to get the final feature maps.
    P1Nstage = KL.Conv2D(32, (3, 3), padding="SAME",  kernel_initializer="he_normal",name="fpn_Convedp1Nstage")(P1)
    P2Nstage = KL.Conv2D(32, (3, 3), padding="SAME", name="fpn_Convedp2Nstage")(P2)
    P3Nstage = KL.Conv2D(64, (3, 3), padding="SAME", name="fpn_Convedp3Nstage")(P3)
    P4Nstage = KL.Conv2D(128, (3, 3), padding="SAME", name="fpn_Convedp4Nstage")(P4)
    P5Nstage = KL.Conv2D(256, (3, 3), padding="SAME", name="fpn_Convedp5Nstage")(P5)
    gpP1Nstage = GlobalAveragePooling2D(dim_ordering='default', name='global_poolP1Nstage')(P1Nstage)
    gpP2Nstage = GlobalAveragePooling2D(dim_ordering='default', name='global_poolP2Nstage')(P2Nstage)
    gpP3Nstage = GlobalAveragePooling2D(dim_ordering='default', name='global_poolP3Nstage')(P3Nstage)
    gpP4Nstage = GlobalAveragePooling2D(dim_ordering='default', name='global_poolP4Nstage')(P4Nstage)
    gpP5Nstage = GlobalAveragePooling2D(dim_ordering='default', name='global_poolP5Nstage')(P5upNstage)

    # rpn_feature_maps = [P2, P3, P4, P5, P6]
    #C1 out 55*55*64 the channel of C1 is the same as the C2, so no pooling
    # C12_input = KL.Conv2D(256, (1, 1), name='fpn_C1toC2')(C1)
    # C12_input = Activation('relu')(C12_input)
    #c2 out 55*55*256    C1 output 56*56*64
    '''
    这个是为了分别提取浅层特征，暂时不用
    C2_input = KL.Conv2D(64, (3, 3), strides=(2, 2), padding="same",name='fpn_C1toC2')(C1)
    C2_output = KL.Add(name="fpn_C1addC2")([C2,C2_input])
    # stage 3
    pool2 = MaxPooling2D(pool_size=(2, 2))(C2_output)
    C3_input = KL.Conv2D(128, (1, 1), name='fpn_C2toC3',kernel_regularizer=L12_reg)(pool2)
    #c3 out 28*28*512
    C3_output = KL.Add(name="fpn_C2addC3")([C3,C3_input])
    #stage 4
    pool34 = MaxPooling2D(pool_size=(2, 2))(C3_output)
    C4_input = KL.Conv2D(256, (1, 1),  kernel_initializer="he_normal",name='fpn_C3toC4')(pool34)
    #c4 out 14*14*1024
    C4_output = KL.Add(name="fpn_C3addC4")([C4,C4_input])
    # stage 5 C5 output 7*7*2048
    pool45 = MaxPooling2D(pool_size=(2, 2))(C4_output)
    C5_input = KL.Conv2D(512, (1, 1),  kernel_initializer="he_normal",name='fpn_C4toC5')(pool45)
    #c5 out output 7*7*2048
    C5_output = KL.Add(name="fpn_C4addC5")([C5,C5_input])
    # C5_output = layers.add([C5, C5_input])
    gpC1 = GlobalAveragePooling2D(dim_ordering='default', name='global_pool1')(C1)
    gpC2 = GlobalAveragePooling2D(dim_ordering='default', name='global_pool2')(C2_output)
    gpC3 = GlobalAveragePooling2D(dim_ordering='default', name='global_pool3')(C3_output)
    gpC4 = GlobalAveragePooling2D(dim_ordering='default', name='global_pool4')(C4_output)
    gpC5 = GlobalAveragePooling2D(dim_ordering='default', name='global_pool5')(C5_output)
    '''
    # gpCall= [gpC1,gpC2,gpC3,gpC4,gpC5,gpP2,gpP3,gpP4,gpP5]
    gpPTstage= [gpP1Tstage,gpP2Tstage,gpP3Tstage,gpP4Tstage,gpP5Tstage]
    featureAllTstage = KL.concatenate(gpPTstage)
    featureTstage = Dense(32, activation='relu', name='Dense1Tstage')(featureAllTstage)
    output_Tstage = Dense(1, activation='sigmoid', name='T_stageOutput')(featureTstage)#loss="categorical_crossentropy"
    # output for Nstage
    gpPNstage= [gpP1Nstage,gpP2Nstage,gpP3Nstage,gpP4Nstage,gpP5Nstage]
    featureAllNstage = KL.concatenate(gpPNstage)
    featureNstage = Dense(32, activation='relu', name='Dense1Nstage')(featureAllNstage)
    output_Nstage = Dense(1, activation='sigmoid', name='N_stageOutput')(featureNstage)#loss="categorical_crossentropy"

    gpCall= [gpP1,gpP2,gpP3,gpP4,gpP5]
    featureallOS = KL.concatenate(gpCall)
    featureall2 = Dense(32, activation='relu', name='Dense1')(featureallOS)
    # the gradient for main task if not allowed back propagate to the subtask
    stop_gradTstage = Lambda(lambda x: K.stop_gradient(x))(featureTstage)
    stop_gradNstage = Lambda(lambda x: K.stop_gradient(x))(featureNstage)
    # featureall1 = KL.Dropout(0.5)(featureall2)
    output_TNMstage = Dense(1, activation='sigmoid', name='TNM_stage')(featureall2)#loss="categorical_crossentropy"
    mergeallPandmultitask = KL.concatenate([featureall2,stop_gradTstage,stop_gradNstage])#,stop_grad   #####
    # mergePandmultitask =  KL.concatenate([gpP1,stop_grad])#if is multi-task, replace featureall2 by stop_grad
    output1 = KL.Dense(32,activation='relu', name='Dense2')(mergeallPandmultitask)
    # output1 = KL.Dropout(0.5,name = 'dropout')(output1)
    if len(clinical_features)>0:
        output1 =  KL.concatenate([output1, clinical_data])
        print('output is concatenated')
    clinical_features
    risk_pred = KL.Dense(num_outputs,activation='sigmoid', name='risk_pred')(output1)
    #LeakyReLU 当神经元未激活时，它仍允许赋予一个很小的梯度： f(x) = alpha * x for x < 0, f(x) = x for x >= 0.
    # risk_pred = KL.LeakyReLU(alpha = -1, name ="risk_pred")(risk_pred)
    # input_shape = (224,224,3)
    # img_input = Input(shape=input_shape,name = 'input')
    # model50 = Model(inputs = img_input, outputs =  [output_TNMstage, risk_pred], name='model50')
    if len(clinical_features)>0:
        model50 = Model(inputs = [img_input,img_input2,clinical_data],outputs = [risk_pred, output_Tstage,output_Nstage], name='model50')
    else:
        model50 = Model(inputs = [img_input,img_input2], outputs =  [risk_pred, output_Tstage,output_Nstage], name='model50')
    # model50 = Model(inputs = img_input, outputs =  risk_pred, name='model50')
    return model50
#%%
if __name__ == '__main__':
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
    os.environ['CUDA_VISIBLE_DEVICES'] = '2'
    from keras.utils import plot_model
    clinical_features = ['age','gender','cTstage','cNstage','cTNMstage']
    clinical_features =[]
    model50 = resnet_self(include_top=True)
    # print(model50.summary())
    os.chdir('/data/zlw/survival1/TMImultitask/plotmodel/') 
    plot_model(model50,to_file='./attention1120.pdf',show_shapes=True)
