
import numpy as np

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import (Flatten, Input, Activation,
                          Reshape, Dropout, Convolution2D,
                          MaxPooling2D, BatchNormalization,
                          Conv2D, GlobalAveragePooling2D,
                          Concatenate, AveragePooling2D,
                          LocallyConnected2D, Dense)

# from general.tfutil import hist_summaries_traintest, scalar_summaries_traintest

from src.subspace.builder.model_builders import make_and_add_losses
from src.subspace.keras_ext.engine import ExtendedModel
from src.subspace.keras_ext.layers import (RProjDense,
                                       RProjConv2D,
                                       RProjBatchNormalization,
                                       RProjLocallyConnected2D)
from src.subspace.keras_ext.rproj_layers_util import (OffsetCreatorDenseProj,
                                                  OffsetCreatorSparseProj,
                                                  OffsetCreatorFastfoodProj,
                                                  FastWalshHadamardProjector,
                                                  ThetaPrime, MultiplyLayer)
from src.subspace.keras_ext.util import make_image_input_preproc
from tensorflow.keras.regularizers import l2


def resnet_layer(inputs,
                 offset_creator_class,
                 vv,
                 num_filters=16,
                 kernel_size=3,
                 strides=1,
                 activation='relu',
                 batch_normalization=True,
                 conv_first=True,
                 kernel_regularizer=l2(1e-4),
                 name=None):
    """2D Convolution-Batch Normalization-Activation stack builder

    # Arguments
        inputs (tensor): input tensor from input image or previous layer
        num_filters (int): Conv2D number of filters
        kernel_size (int): Conv2D square kernel dimensions
        strides (int): Conv2D square stride dimensions
        activation (string|None): activation name
        batch_normalization (bool): whether to include batch normalization
        conv_first (bool): conv-bn-activation (True) or
            bn-activation-conv (False)

    # Returns
        x (tensor): tensor as input to the next layer
    """
    conv = RProjConv2D(offset_creator_class, vv,
                       num_filters,
                       kernel_size=kernel_size,
                       strides=strides,
                       padding='same',
                       kernel_initializer='he_normal',
                       activation=None,
                       kernel_regularizer=kernel_regularizer,
                       name=name)

    x = inputs
    if conv_first:
        x = conv(x)
        if batch_normalization:
            x = RProjBatchNormalization(offset_creator_class, vv)(x)
            # x = BatchNormalization()(x) # does this even make sense
        if activation is not None:
            x = Activation(activation)(x)
    else:
        pass
        # if batch_normalization:
        #     x = BatchNormalization()(x)
        # if activation is not None:
        #     x = Activation(activation)(x)
        # x = conv(x)
    return x


def build_LeNet_resnet(depth, weight_decay=0, vsize=100, shift_in=None, proj_type='sparse', disable_bn=False):
    im_shape = (32, 32, 3)
    n_label_vals = 10
    im_dtype = 'float32'
    batch_norm_enabled = not disable_bn

    if (depth - 2) % 6 != 0:
        raise ValueError('depth should be 6n+2 (eg 20, 32, 44 in [a])')
    num_res_blocks = int((depth - 2) / 6)

    assert proj_type in ('dense', 'sparse')
    if proj_type == 'dense':
        offset_creator_class = OffsetCreatorDenseProj
    else:
        # sparse
        offset_creator_class = OffsetCreatorSparseProj

    with tf.name_scope('inputs'):
        input_images, preproc_images = make_image_input_preproc(im_shape, im_dtype, shift_in=shift_in)
        input_labels = Input(batch_shape=(None,), dtype='int64')

    with tf.name_scope('net') as scope:
        vv = ThetaPrime(vsize)

        num_filters = 16

        x = resnet_layer(preproc_images, offset_creator_class, vv,
                         num_filters=num_filters,
                         batch_normalization=batch_norm_enabled)
        # Instantiate the stack of residual units
        for stack in range(3):
            for res_block in range(num_res_blocks):
                strides = 1
                if stack > 0 and res_block == 0:  # first layer but not first stack
                    strides = 2  # downsample
                y = resnet_layer(x,
                                 offset_creator_class,
                                 vv,
                                 num_filters=num_filters,
                                 strides=strides,
                                 name=f"Conv2D_stack{stack}_res{res_block}_l0",
                                 batch_normalization=batch_norm_enabled)
                y = resnet_layer(y,
                                 offset_creator_class,
                                 vv,
                                 num_filters=num_filters,
                                 activation=None,
                                 name=f"Conv2D_stack{stack}_res{res_block}_l1",
                                 batch_normalization=batch_norm_enabled)
                if stack > 0 and res_block == 0:  # first layer but not first stack
                    # linear projection residual shortcut connection to match
                    # changed dims
                    x = resnet_layer(x,
                                     offset_creator_class,
                                     vv,
                                     num_filters=num_filters,
                                     kernel_size=1,
                                     strides=strides,
                                     activation=None,
                                     batch_normalization=False,
                                     name=f"Conv2D_stack{stack}_res{res_block}_l2")
                x = tf.keras.layers.add([x, y])
                x = Activation('relu')(x)
            num_filters *= 2

        x = AveragePooling2D(pool_size=8)(x)
        y = Flatten()(x)
        logits = RProjDense(offset_creator_class, vv, n_label_vals,
                        activation='softmax',
                        kernel_initializer='he_normal')(y)

        model = ExtendedModel(input=input_images, output=logits)
        model.add_extra_trainable_weight(vv.var_2d)

    nontrackable_fields = ['input_images', 'preproc_images', 'input_labels', 'logits']
    for field in nontrackable_fields:
        model.add_var(field, locals()[field])

    make_and_add_losses(model, input_labels)
    return model


def resnet_layer_ff(inputs,
                 conv2d_class,
                 num_filters=16,
                 kernel_size=3,
                 strides=1,
                 activation='relu',
                 batch_normalization=True,
                 conv_first=True,
                 kernel_regularizer=l2(1e-4),
                 name=None):
    """2D Convolution-Batch Normalization-Activation stack builder

    # Arguments
        inputs (tensor): input tensor from input image or previous layer
        num_filters (int): Conv2D number of filters
        kernel_size (int): Conv2D square kernel dimensions
        strides (int): Conv2D square stride dimensions
        activation (string|None): activation name
        batch_normalization (bool): whether to include batch normalization
        conv_first (bool): conv-bn-activation (True) or
            bn-activation-conv (False)

    # Returns
        x (tensor): tensor as input to the next layer
    """
    conv = conv2d_class(num_filters,
                       kernel_size=kernel_size,
                       strides=strides,
                       padding='same',
                       kernel_initializer='he_normal',
                       activation=None,
                       kernel_regularizer=kernel_regularizer,
                       name=name)

    x = inputs
    if conv_first:
        x = conv(x)
        if batch_normalization:
            x = BatchNormalization()(x) # does this even make sense
        if activation is not None:
            x = Activation(activation)(x)
    else:
        pass
        # if batch_normalization:
        #     x = BatchNormalization()(x)
        # if activation is not None:
        #     x = Activation(activation)(x)
        # x = conv(x)
    return x

def build_resnet_fastfood(depth, weight_decay=0, vsize=100, shift_in=None, proj_type='sparse', DD=None):
    im_shape = (32, 32, 3)
    n_label_vals = 10
    im_dtype = 'float32'

    if (depth - 2) % 6 != 0:
        raise ValueError('depth should be 6n+2 (eg 20, 32, 44 in [a])')
    num_res_blocks = int((depth - 2) / 6)

    assert proj_type in ('dense', 'sparse')
    if proj_type == 'dense':
        offset_creator_class = OffsetCreatorDenseProj
    else:
        # sparse
        offset_creator_class = OffsetCreatorSparseProj

    with tf.name_scope('inputs'):
        input_images, preproc_images = make_image_input_preproc(im_shape, im_dtype, shift_in=shift_in)
        input_labels = Input(batch_shape=(None,), dtype='int64')

    def define_model(input_images, DenseLayer, ConvLayer):
        vv = ThetaPrime(vsize)

        num_filters = 16

        x = resnet_layer_ff(preproc_images, ConvLayer, num_filters=num_filters)
        # Instantiate the stack of residual units
        for stack in range(3):
            for res_block in range(num_res_blocks):
                strides = 1
                if stack > 0 and res_block == 0:  # first layer but not first stack
                    strides = 2  # downsample
                y = resnet_layer_ff(x,
                                 ConvLayer,
                                 num_filters=num_filters,
                                 strides=strides,
                                 name=f"Conv2D_stack{stack}_res{res_block}_l0")
                y = resnet_layer_ff(y,
                                 ConvLayer,
                                 num_filters=num_filters,
                                 activation=None,
                                 name=f"Conv2D_stack{stack}_res{res_block}_l1")
                if stack > 0 and res_block == 0:  # first layer but not first stack
                    # linear projection residual shortcut connection to match
                    # changed dims
                    x = resnet_layer_ff(x,
                                     ConvLayer,
                                     num_filters=num_filters,
                                     kernel_size=1,
                                     strides=strides,
                                     activation=None,
                                     batch_normalization=False,
                                     name=f"Conv2D_stack{stack}_res{res_block}_l2")
                x = tf.keras.layers.add([x, y])
                x = Activation('relu')(x)
            num_filters *= 2

        x = AveragePooling2D(pool_size=8)(x)
        y = Flatten()(x)
        logits = DenseLayer(n_label_vals,
                        activation='softmax',
                        kernel_initializer='he_normal')(y)

        model = ExtendedModel(input=input_images, output=logits)
        model.add_extra_trainable_weight(vv.var_2d)
        return model

    if not DD:
        with tf.name_scope('net_disposable'):
            # Make disposable direct model
            model_disposable = define_model(input_images, Dense, Conv2D)

            DD = np.sum([np.prod(var.get_shape().as_list()) for var in model_disposable.trainable_weights]).item()
            print(f"D {DD} {type(DD)}")
            del model_disposable


    with tf.name_scope('net'):
        # Make real RProj FWH model
        fwh_projector = FastWalshHadamardProjector(vsize, DD)

        DenseLayer = lambda *args, **kwargs: RProjDense(OffsetCreatorFastfoodProj, fwh_projector, *args, **kwargs)
        Conv2DLayer = lambda *args, **kwargs: RProjConv2D(OffsetCreatorFastfoodProj, fwh_projector, *args, **kwargs)

        model = define_model(input_images, DenseLayer, Conv2DLayer)
        fwh_projector.check_usage()

        for ww in fwh_projector.trainable_weights:
            model.add_extra_trainable_weight(ww)
        for ww in fwh_projector.non_trainable_weights:
            model.add_extra_non_trainable_weight(ww)

    nontrackable_fields = ['input_images', 'preproc_images', 'input_labels', 'logits']
    for field in nontrackable_fields:
        model.add_var(field, locals()[field])

    make_and_add_losses(model, input_labels)
    return model

# Model: "model"
# __________________________________________________________________________________________________
# Layer (type)                    Output Shape         Param #     Connected to
# ==================================================================================================
# input_1 (InputLayer)            [(None, 32, 32, 3)]  0
# __________________________________________________________________________________________________
# conv2d (Conv2D)                 (None, 32, 32, 16)   448         input_1[0][0]
# __________________________________________________________________________________________________
# batch_normalization (BatchNorma (None, 32, 32, 16)   64          conv2d[0][0]
# __________________________________________________________________________________________________
# activation (Activation)         (None, 32, 32, 16)   0           batch_normalization[0][0]
# __________________________________________________________________________________________________
# Conv2D_stack0_res0_l0 (Conv2D)  (None, 32, 32, 16)   2320        activation[0][0]
# __________________________________________________________________________________________________
# batch_normalization_1 (BatchNor (None, 32, 32, 16)   64          Conv2D_stack0_res0_l0[0][0]
# __________________________________________________________________________________________________
# activation_1 (Activation)       (None, 32, 32, 16)   0           batch_normalization_1[0][0]
# __________________________________________________________________________________________________
# Conv2D_stack0_res0_l1 (Conv2D)  (None, 32, 32, 16)   2320        activation_1[0][0]
# __________________________________________________________________________________________________
# batch_normalization_2 (BatchNor (None, 32, 32, 16)   64          Conv2D_stack0_res0_l1[0][0]
# __________________________________________________________________________________________________
# add (Add)                       (None, 32, 32, 16)   0           activation[0][0]
#                                                                  batch_normalization_2[0][0]
# __________________________________________________________________________________________________
# activation_2 (Activation)       (None, 32, 32, 16)   0           add[0][0]
# __________________________________________________________________________________________________
# Conv2D_stack0_res1_l0 (Conv2D)  (None, 32, 32, 16)   2320        activation_2[0][0]
# __________________________________________________________________________________________________
# batch_normalization_3 (BatchNor (None, 32, 32, 16)   64          Conv2D_stack0_res1_l0[0][0]
# __________________________________________________________________________________________________
# activation_3 (Activation)       (None, 32, 32, 16)   0           batch_normalization_3[0][0]
# __________________________________________________________________________________________________
# Conv2D_stack0_res1_l1 (Conv2D)  (None, 32, 32, 16)   2320        activation_3[0][0]
# __________________________________________________________________________________________________
# batch_normalization_4 (BatchNor (None, 32, 32, 16)   64          Conv2D_stack0_res1_l1[0][0]
# __________________________________________________________________________________________________
# add_1 (Add)                     (None, 32, 32, 16)   0           activation_2[0][0]
#                                                                  batch_normalization_4[0][0]
# __________________________________________________________________________________________________
# activation_4 (Activation)       (None, 32, 32, 16)   0           add_1[0][0]
# __________________________________________________________________________________________________
# Conv2D_stack0_res2_l0 (Conv2D)  (None, 32, 32, 16)   2320        activation_4[0][0]
# __________________________________________________________________________________________________
# batch_normalization_5 (BatchNor (None, 32, 32, 16)   64          Conv2D_stack0_res2_l0[0][0]
# __________________________________________________________________________________________________
# activation_5 (Activation)       (None, 32, 32, 16)   0           batch_normalization_5[0][0]
# __________________________________________________________________________________________________
# Conv2D_stack0_res2_l1 (Conv2D)  (None, 32, 32, 16)   2320        activation_5[0][0]
# __________________________________________________________________________________________________
# batch_normalization_6 (BatchNor (None, 32, 32, 16)   64          Conv2D_stack0_res2_l1[0][0]
# __________________________________________________________________________________________________
# add_2 (Add)                     (None, 32, 32, 16)   0           activation_4[0][0]
#                                                                  batch_normalization_6[0][0]
# __________________________________________________________________________________________________
# activation_6 (Activation)       (None, 32, 32, 16)   0           add_2[0][0]
# __________________________________________________________________________________________________
# Conv2D_stack1_res0_l0 (Conv2D)  (None, 16, 16, 32)   4640        activation_6[0][0]
# __________________________________________________________________________________________________
# batch_normalization_7 (BatchNor (None, 16, 16, 32)   128         Conv2D_stack1_res0_l0[0][0]
# __________________________________________________________________________________________________
# activation_7 (Activation)       (None, 16, 16, 32)   0           batch_normalization_7[0][0]
# __________________________________________________________________________________________________
# Conv2D_stack1_res0_l1 (Conv2D)  (None, 16, 16, 32)   9248        activation_7[0][0]
# __________________________________________________________________________________________________
# Conv2D_stack1_res0_l2 (Conv2D)  (None, 16, 16, 32)   544         activation_6[0][0]
# __________________________________________________________________________________________________
# batch_normalization_8 (BatchNor (None, 16, 16, 32)   128         Conv2D_stack1_res0_l1[0][0]
# __________________________________________________________________________________________________
# add_3 (Add)                     (None, 16, 16, 32)   0           Conv2D_stack1_res0_l2[0][0]
#                                                                  batch_normalization_8[0][0]
# __________________________________________________________________________________________________
# activation_8 (Activation)       (None, 16, 16, 32)   0           add_3[0][0]
# __________________________________________________________________________________________________
# Conv2D_stack1_res1_l0 (Conv2D)  (None, 16, 16, 32)   9248        activation_8[0][0]
# __________________________________________________________________________________________________
# batch_normalization_9 (BatchNor (None, 16, 16, 32)   128         Conv2D_stack1_res1_l0[0][0]
# __________________________________________________________________________________________________
# activation_9 (Activation)       (None, 16, 16, 32)   0           batch_normalization_9[0][0]
# __________________________________________________________________________________________________
# Conv2D_stack1_res1_l1 (Conv2D)  (None, 16, 16, 32)   9248        activation_9[0][0]
# __________________________________________________________________________________________________
# batch_normalization_10 (BatchNo (None, 16, 16, 32)   128         Conv2D_stack1_res1_l1[0][0]
# __________________________________________________________________________________________________
# add_4 (Add)                     (None, 16, 16, 32)   0           activation_8[0][0]
#                                                                  batch_normalization_10[0][0]
# __________________________________________________________________________________________________
# activation_10 (Activation)      (None, 16, 16, 32)   0           add_4[0][0]
# __________________________________________________________________________________________________
# Conv2D_stack1_res2_l0 (Conv2D)  (None, 16, 16, 32)   9248        activation_10[0][0]
# __________________________________________________________________________________________________
# batch_normalization_11 (BatchNo (None, 16, 16, 32)   128         Conv2D_stack1_res2_l0[0][0]
# __________________________________________________________________________________________________
# activation_11 (Activation)      (None, 16, 16, 32)   0           batch_normalization_11[0][0]
# __________________________________________________________________________________________________
# Conv2D_stack1_res2_l1 (Conv2D)  (None, 16, 16, 32)   9248        activation_11[0][0]
# __________________________________________________________________________________________________
# batch_normalization_12 (BatchNo (None, 16, 16, 32)   128         Conv2D_stack1_res2_l1[0][0]
# __________________________________________________________________________________________________
# add_5 (Add)                     (None, 16, 16, 32)   0           activation_10[0][0]
#                                                                  batch_normalization_12[0][0]
# __________________________________________________________________________________________________
# activation_12 (Activation)      (None, 16, 16, 32)   0           add_5[0][0]
# __________________________________________________________________________________________________
# Conv2D_stack2_res0_l0 (Conv2D)  (None, 8, 8, 64)     18496       activation_12[0][0]
# __________________________________________________________________________________________________
# batch_normalization_13 (BatchNo (None, 8, 8, 64)     256         Conv2D_stack2_res0_l0[0][0]
# __________________________________________________________________________________________________
# activation_13 (Activation)      (None, 8, 8, 64)     0           batch_normalization_13[0][0]
# __________________________________________________________________________________________________
# Conv2D_stack2_res0_l1 (Conv2D)  (None, 8, 8, 64)     36928       activation_13[0][0]
# __________________________________________________________________________________________________
# Conv2D_stack2_res0_l2 (Conv2D)  (None, 8, 8, 64)     2112        activation_12[0][0]
# __________________________________________________________________________________________________
# batch_normalization_14 (BatchNo (None, 8, 8, 64)     256         Conv2D_stack2_res0_l1[0][0]
# __________________________________________________________________________________________________
# add_6 (Add)                     (None, 8, 8, 64)     0           Conv2D_stack2_res0_l2[0][0]
#                                                                  batch_normalization_14[0][0]
# __________________________________________________________________________________________________
# activation_14 (Activation)      (None, 8, 8, 64)     0           add_6[0][0]
# __________________________________________________________________________________________________
# Conv2D_stack2_res1_l0 (Conv2D)  (None, 8, 8, 64)     36928       activation_14[0][0]
# __________________________________________________________________________________________________
# batch_normalization_15 (BatchNo (None, 8, 8, 64)     256         Conv2D_stack2_res1_l0[0][0]
# __________________________________________________________________________________________________
# activation_15 (Activation)      (None, 8, 8, 64)     0           batch_normalization_15[0][0]
# __________________________________________________________________________________________________
# Conv2D_stack2_res1_l1 (Conv2D)  (None, 8, 8, 64)     36928       activation_15[0][0]
# __________________________________________________________________________________________________
# batch_normalization_16 (BatchNo (None, 8, 8, 64)     256         Conv2D_stack2_res1_l1[0][0]
# __________________________________________________________________________________________________
# add_7 (Add)                     (None, 8, 8, 64)     0           activation_14[0][0]
#                                                                  batch_normalization_16[0][0]
# __________________________________________________________________________________________________
# activation_16 (Activation)      (None, 8, 8, 64)     0           add_7[0][0]
# __________________________________________________________________________________________________
# Conv2D_stack2_res2_l0 (Conv2D)  (None, 8, 8, 64)     36928       activation_16[0][0]
# __________________________________________________________________________________________________
# batch_normalization_17 (BatchNo (None, 8, 8, 64)     256         Conv2D_stack2_res2_l0[0][0]
# __________________________________________________________________________________________________
# activation_17 (Activation)      (None, 8, 8, 64)     0           batch_normalization_17[0][0]
# __________________________________________________________________________________________________
# Conv2D_stack2_res2_l1 (Conv2D)  (None, 8, 8, 64)     36928       activation_17[0][0]
# __________________________________________________________________________________________________
# batch_normalization_18 (BatchNo (None, 8, 8, 64)     256         Conv2D_stack2_res2_l1[0][0]
# __________________________________________________________________________________________________
# add_8 (Add)                     (None, 8, 8, 64)     0           activation_16[0][0]
#                                                                  batch_normalization_18[0][0]
# __________________________________________________________________________________________________
# activation_18 (Activation)      (None, 8, 8, 64)     0           add_8[0][0]
# __________________________________________________________________________________________________
# average_pooling2d (AveragePooli (None, 1, 1, 64)     0           activation_18[0][0]
# __________________________________________________________________________________________________
# flatten (Flatten)               (None, 64)           0           average_pooling2d[0][0]
# __________________________________________________________________________________________________
# dense (Dense)                   (None, 10)           650         flatten[0][0]
# ==================================================================================================
# Total params: 274,442
# Trainable params: 273,066
# Non-trainable params: 1,376
# __________________________________________________________________________________________________

