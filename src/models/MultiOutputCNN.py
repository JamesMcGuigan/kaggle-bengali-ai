import inspect
import types
from typing import Dict, List, Union, cast

from tensorflow.keras import Input, Model, regularizers
from tensorflow.keras.layers import (
    BatchNormalization,
    Conv2D,
    Dense,
    Dropout,
    Flatten,
    GlobalMaxPooling2D,
    MaxPooling2D,
    )



# noinspection DuplicatedCode
def MultiOutputCNN(
        input_shape,
        output_shape: Union[List, Dict],
        cnns_per_maxpool=4,
        maxpool_layers=4,      # increasing `maxpool_layers` prefers fewer `cnns_per_maxpool` (ideal total CNNs = 15 / 16)
        cnn_units=32,
        cnn_kernel=3,
        cnn_strides=1,
        dense_layers=1,        # `dense_layers=1` is preferred over `2` or `3`
        dense_units=256,
        activation='relu',     # 'relu' | 'crelu' | 'leaky_relu' | 'relu6' | 'softmax' | 'tanh' | 'hard_sigmoid' | 'sigmoid'
        dropout=0.25,
        regularization=False,  # `regularization=True` prefers `global_maxpool=False` and fewer dense_units - but worse results
        global_maxpool=False,  # `global_maxpool=True` prefers double the number of `dense_units` and +1 `cnns_per_maxpool`
        name='',
)  -> Model:
    function_name = cast(types.FrameType, inspect.currentframe()).f_code.co_name
    model_name    = f"{function_name}-{name}" if name else function_name
    # model_name  = seq([ function_name, name ]).filter(lambda x: x).make_string("-")  # remove dependency on pyfunctional - not in Kaggle repo without internet

    inputs = Input(shape=input_shape)
    x      = inputs

    for cnn1 in range(1,maxpool_layers+1):
        for cnn2 in range(1, cnns_per_maxpool+1):
            x = Conv2D( cnn_units * cnn1,
                        kernel_size=cnn_kernel,
                        strides=cnn_strides,
                        padding='same',
                        activation=activation)(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = BatchNormalization()(x)
        x = Dropout(dropout)(x)

    if global_maxpool:
        x = GlobalMaxPooling2D()(x)

    x = Flatten()(x)

    for nn1 in range(0,dense_layers):
        if regularization:
            x = Dense(dense_units,
                      activation=activation,
                      kernel_regularizer=regularizers.l2(0.01),
                      activity_regularizer=regularizers.l1(0.01))(x)
        else:
            x = Dense(dense_units, activation=activation)(x)

        x = BatchNormalization()(x)
        x = Dropout(dropout)(x)

    x = Flatten(name='output')(x)

    if isinstance(output_shape, dict):
        outputs = [
            Dense(output_shape, activation='softmax', name=key)(x)
            for key, output_shape in output_shape.items()
        ]
    else:
        outputs = [
            Dense(output_shape, activation='softmax', name=f'output_{index}')(x)
            for index, output_shape in enumerate(output_shape)
        ]

    model = Model(inputs, outputs, name=model_name)
    # plot_model(model, to_file=os.path.join(os.path.dirname(__file__), f"{name}.png"))
    return model
