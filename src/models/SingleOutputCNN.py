import inspect
import types
from typing import cast

from tensorflow_core.python.keras import Input, Model, regularizers
from tensorflow_core.python.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, BatchNormalization, \
    GlobalMaxPooling2D


def SingleOutputCNN(
        input_shape,
        output_shape,
        cnns_per_maxpool=1,
        maxpool_layers=1,
        dense_layers=1,
        dense_units=64,
        dropout=0.25,
        regularization=False,
        global_maxpool=False,
        name='',
)  -> Model:
    function_name = cast(types.FrameType, inspect.currentframe()).f_code.co_name
    model_name    = f"{function_name}-{name}" if name else function_name
    # model_name  = seq([ function_name, name ]).filter(lambda x: x).make_string("-")  # remove dependency on pyfunctional - not in Kaggle repo without internet

    inputs = Input(shape=input_shape)
    x      = inputs

    for cnn1 in range(0,maxpool_layers):
        for cnn2 in range(1, cnns_per_maxpool+1):
            x = Conv2D( 32 * cnn2, kernel_size=(3, 3), padding='same', activation='relu')(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = BatchNormalization()(x)
        x = Dropout(dropout)(x)

    if global_maxpool:
        x = GlobalMaxPooling2D()(x)

    x = Flatten()(x)

    for nn1 in range(0,dense_layers):
        if regularization:
            x = Dense(dense_units, activation='relu',
                      kernel_regularizer=regularizers.l2(0.01),
                      activity_regularizer=regularizers.l1(0.01))(x)
        else:
            x = Dense(dense_units, activation='relu')(x)

        x = BatchNormalization()(x)
        x = Dropout(dropout)(x)

    x = Dense(output_shape, activation='softmax')(x)

    model = Model(inputs, x, name=model_name)
    # plot_model(model, to_file=os.path.join(os.path.dirname(__file__), f"{name}.png"))
    return model
