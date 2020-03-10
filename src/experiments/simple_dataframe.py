# This is a first pass, simplest thing that could possibly work attempt
# We train three separate MINST style CNNs for each label, then combine the results
import os

import numpy as np
import pandas as pd
import tensorflow as tf

from src.dataset.DatasetDF import DatasetDF
from src.models.MinstCNN import MinstCNN
from src.util.hparam import model_compile_fit

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 0, 1, 2, 3  # Disable Tensortflow Logging

# https://stackoverflow.com/questions/36927607/how-can-i-solve-ran-out-of-gpu-memory-in-tensorflow
config  = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)


dirs = {
    "data":        "./input/bengaliai-cv19",
    "models":      "./output/models/simple_dataframe",
    "predictions": "./output/predictions",
    "logs":        "./logs",
}
for dir in dirs.values(): os.makedirs(dir, exist_ok=True)

image_shape = (137,236)
hparams = {
    # "optimizer":     "Adagrad",
    # "scheduler":     "plateau2",
    # "learning_rate": 0.1,
    "optimizer":     "RMSprop",
    "scheduler":     "constant",
    "learning_rate": 0.001,
    "min_lr":        0.001,
    "batch_size":    128,
    "patience":      3,
}
model_hparams = {
    "cnns_per_maxpool": 1,
    "maxpool_layers":   5,
    "dense_layers":     1,
    "dense_units":     64
}
model_hparams_key = "-".join( f"{key}={value}" for key,value in model_hparams.items() ).replace(' ','')
print("hparams", hparams)
print("model_hparams", model_hparams)


csv_data    = pd.read_csv(f"{dirs['data']}/train.csv")
models      = {}
model_files = {}
model_stats = {}
output_fields = [ "consonant_diacritic", "grapheme_root", "vowel_diacritic" ]
for output_field in output_fields:
    model_stats[output_field] = []
    model_files[output_field] = f"{dirs['models']}/simple_dataframe-{model_hparams_key}-{output_field}.hdf5"

for output_field in output_fields:
    # # Release GPU memory
    # cuda.select_device(0)
    # cuda.close()
    # gc.collect()

    output_shape         = csv_data[output_field].nunique()
    models[output_field] = MinstCNN(
        input_shape=(*image_shape, 1),
        output_shape=output_shape,
        name=output_field,
        **model_hparams,
    )

    if os.path.exists( model_files[output_field] ):
        models[output_field].load_weights( model_files[output_field] )

    models[output_field].summary()

    for data_id in range(0,4):
        print("------------------------------")
        print(f"Training | {output_field} | {model_hparams} | data_id: {data_id}")
        print("------------------------------")
        dataset = DatasetDF(data_id=data_id, Y_field=output_field)

        stats = model_compile_fit(
            hparams    = { **hparams, **model_hparams },
            model      = models[output_field],
            dataset    = dataset,
            model_file = model_files[output_field],
            log_dir    = f"{dirs['logs']}/simple_dataframe/{output_field}/",
            verbose    = True,
        )
        model_stats[output_field].append(stats)

    print("------------------------------")
    print(f"Completed | {output_field} | {model_hparams}")
    for stats in model_stats[output_field]: print(stats)
    print("------------------------------")


### Log Stats Results
with open(f"{dirs['predictions']}/simple_dataframe-{model_hparams_key}.log", 'w') as file:
    output = []
    output.append("------------------------------")
    for output_field in output_fields:
        output.append(f"Completed | {output_field} | {model_hparams}")
        for stats in model_stats[output_field]:
            output.append(str(stats))
        output.append("------------------------------")
    print(      "\n".join(output) )
    file.write( "\n".join(output) )
    print("wrote:", f"{dirs['predictions']}/simple_dataframe-{model_hparams_key}.log")


### Output Predictions to CSV
test_dataset = DatasetDF(data_id=0)  # contains all test data
predictions  = pd.DataFrame()
for output_field in output_fields:
    prediction = models[output_field].predict(test_dataset.X['test'])
    prediction = np.argmax( prediction, axis=-1 )
    predictions[output_field] = prediction

submission = pd.DataFrame(columns=['row_id','target'])
for index, row in predictions.iterrows():
    for output_field in output_fields:
        submission = submission.append({
            'row_id': f"Test_{index}_{output_field}",
            'target': predictions[output_field].iloc[index],
        }, ignore_index=True)

submission.to_csv(f"{dirs['predictions']}/simple_dataframe-{model_hparams_key}.csv", index=False)
print("wrote:",   f"{dirs['predictions']}/simple_dataframe-{model_hparams_key}.csv", predictions.shape)

