import inspect
import types
from typing import Dict, List, Union, cast

import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Dense, Flatten
# noinspection DuplicatedCode
from tensorflow_core.python.keras.layers import BatchNormalization, Dropout


def MultiOutputApplication(
        input_shape,
        output_shape: Union[List, Dict],
        application='NASNetMobile',
        weights=None,   # None or 'imagenet'
        pooling='avg',  # None, 'avg', 'max',
        dense_units=512, # != (1295+168+11+7),
        dense_layers=2,
        dropout=0.25
)  -> Model:
    function_name = cast(types.FrameType, inspect.currentframe()).f_code.co_name
    model_name    = f"{function_name}-{application}" if application else function_name

    inputs = Input(shape=input_shape)
    x      = inputs

    if application == 'NASNetMobile':
        application_model = tf.keras.applications.nasnet.NASNetMobile(
            input_shape=input_shape,
            input_tensor=inputs,
            include_top=False,
            weights=weights,
            pooling=pooling,
            classes=1000,
        )
    else:
        raise Exception(f"MultiOutputApplication() - unknown application: {application}")

    x = application_model(x)
    x = Flatten(name='output')(x)

    for nn1 in range(0,dense_layers):
        x = Dense(dense_units, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(dropout)(x)

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
