## Kaggle Competition Entry
# Bengali.AI Handwritten Grapheme Classification

Classify the components of handwritten Bengali

https://www.kaggle.com/c/bengaliai-cv19

<img alt='Recognize the constituents given a handwritten grapheme image' style='float:right' width=300 src="https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F1095143%2Fa9a48686e3f385d9456b59bf2035594c%2Fdesc.png?generation=1576531903599785&alt=media"/>

## Scores

| Position              | Private | Public |  Prize       | Notes |
|:----------------------|:-------:|:------:|-------------:|-------|
| 1st Place             |  0.9762 | 0.9952 | $5000        |       |
| 2nd Place             |  0.9689 | 0.9955 | $2000        |       |
| 3rd Place             |  0.9645 | 0.9945 | $1000        |       |
| Top 14                |  0.9491 | 0.9913 | Gold Medal   |       |
| Top 5%  (102)         |  0.9348 | 0.9858 | Silver Medal |       |
| Top 10% (205)         |  0.9306 | 0.9791 | Bronze Medal |       |
| [Final Writeup](https://www.kaggle.com/jamesmcguigan/bengali-ai-cnn-data-pipeline-problem-solving?scriptVersionId=35914381) |  0.9038 | 0.9381 | 1410/2059 | Late Submission  |
| [ImageDataGenerator - CNN](https://www.kaggle.com/jamesmcguigan/bengali-ai-imagedatagenerator-cnn?scriptVersionId=31218596) |  0.9010 | 0.9413 | | 4\*4 CNN + 1\*256 Dense |
| [ImageDataGenerator - CNN](https://www.kaggle.com/jamesmcguigan/bengali-ai-imagedatagenerator-cnn?scriptVersionId=31203616) |  0.8961 | 0.9482 | | 3\*5 CNN + 1\*256 Dense + Y+=grapheme |
| [ImageDataGenerator - CNN](https://www.kaggle.com/jamesmcguigan/bengali-ai-imagedatagenerator-cnn?scriptVersionId=30636537) |  0.8921 | 0.9396 | | 3\*4 CNN + 2\*256 Dense + Y+=grapheme |
| [Multi Output DF CNN](https://www.kaggle.com/jamesmcguigan/bengali-ai-multi-output-df-cnn?scriptVersionId=31204140)         |  0.8901 | 0.9402 | | 3\*5 CNN + 1\*256 Dense + Y+=grapheme - no augmentation |
| [Multi Output DF CNN](https://www.kaggle.com/jamesmcguigan/bengali-ai-multi-output-df-cnn?scriptVersionId=30830488)         |  0.8828 | 0.9337 | | 3\*4 CNN + 2\*256 Dense + Y+=grapheme - no augmentation |
| [ImageDataGenerator - CNN](https://www.kaggle.com/jamesmcguigan/bengali-ai-imagedatagenerator-cnn?scriptVersionId=31262979) |  0.8797 | 0.9198 | | 4\*4 CNN + 1\*256 Dense + Y+=grapheme + Regularization  |
| sample_submission.csv |  0.0614 | 0.0614 | Random Score |       |


## Dataset
The parquet training dataset consists of 200,840 grayscale images (in 4 files) at 137x236 resolution with a total
 filesize of 4.8GB of data.

```
./requirements.sh                                             # create ./venv/

kaggle competitions download -c bengaliai-cv19 -p ./input/
unzip ./input/bengaliai-cv19.zip -d ./input/bengaliai-cv19/

time ./src/preprocessing/write_images_to_filesystem.py        # optional
```

> Bengali is the 5th most spoken language in the world with hundreds of million of speakers. It’s the official
> language of Bangladesh and the second most spoken language in India. Considering its reach, there’s significant
> business and educational interest in developing AI that can optically recognize images of the language handwritten
> . This challenge hopes to improve on approaches to Bengali recognition.
>
> Optical character recognition is particularly challenging for Bengali. While Bengali has 49 letters (to be more
> specific 11 vowels and 38 consonants) in its alphabet, there are also 18 potential diacritics, or accents. This
> means that there are many more graphemes, or the smallest units in a written language. The added complexity results
> in ~13,000 different grapheme variations (compared to English’s 250 graphemic units).
>
> For this competition, you’re given the image of a handwritten Bengali grapheme and are challenged to separately
> classify three constituent elements in the image: grapheme root, vowel diacritics, and consonant diacritics.
>
> This dataset contains images of individual hand-written 
> [Bengali characters](https://en.wikipedia.org/wiki/Bengali_alphabet). Bengali characters (graphemes) are written by
> combining three components: a grapheme_root, vowel_diacritic, and consonant_diacritic. Your challenge is to classify 
> the components of the grapheme in each image. There are roughly 10,000 possible graphemes, of which roughly 1,000
> are represented in the training set. The test set includes some graphemes that do not exist in train but has no
> new grapheme components. It takes a lot of volunteers filling out 
> [sheets like this](https://github.com/BengaliAI/graphemePrepare/blob/master/collection/A4/form_1.jpg)
> to generate a useful amount of real data; focusing the problem on the grapheme components rather than on recognizing
> whole graphemes should make it possible to assemble a Bengali OCR system without handwriting samples for all 10,000
> graphemes.

      

## Notebooks

### Previous Competitions
- [Kaggle Competition Entry - MNIST Digit Recognizer](https://github.com/JamesMcGuigan/kaggle-digit-recognizer)

### Technical Research
- [Reading Parquet Files RAM CPU Optimization](notebooks/Reading%20Parquet%20Files%20RAM%20CPU%20Optimization.ipynb)
[[Kaggle Version](https://www.kaggle.com/jamesmcguigan/reading-parquet-files-ram-cpu-optimization)]
- [Jupyter Environment Variable os.environ](notebooks/Jupyter%20Environment%20Variable%20os.environ.ipynb) 
[[Kaggle Version](https://www.kaggle.com/jamesmcguigan/jupyter-environment-variable-os-environ)]

### Image Preprocessing
- [Image Processing](notebooks/Image%20Processing.ipynb)
[[Kaggle Version](https://www.kaggle.com/jamesmcguigan/bengali-ai-image-processing)]
- Dataset as Image Directory [[Kaggle Version](https://www.kaggle.com/jamesmcguigan/bengali-ai-dataset-as-image-directory)]

### Exploratory Data Analysis 
- [EDA Grapheme Combinations](notebooks/EDA%20Grapheme%20Combinations.ipynb)
[[Kaggle Version](https://www.kaggle.com/jamesmcguigan/bengali-ai-dataset-eda-grapheme-combinations)]
- [Unicode Visualization of the Bengali Alphabet](notebooks/Unicode%20Visualization%20of%20the%20Bengali%20Alphabet.ipynb)
[[Kaggle Version](https://www.kaggle.com/jamesmcguigan/unicode-visualization-of-the-bengali-alphabet)]

### Writeup and Submission
- [Bengali AI - CNN Data Pipeline + Problem Solving](https://www.kaggle.com/jamesmcguigan/bengali-ai-cnn-data-pipeline-problem-solving)


The Exploratory Data Analysis showed that only certain combinations of vowel/consonant diacritics where regularly
 combined with certain grapheme roots. 
 
Vowel / Consonant Combinations:
- Vowel #0 and Consonant #0 combine with everything
- Vowels #3, #5, #6, #8 have limited combinations with Consonants
- Consonant #3 is never combined except with Vowel #0
- Consonant #6 only combineds with Vowels #0 and #1

Grapheme Root Combinations:
- Vowel #0 and Consonant #0 combine with (nearly) everything
- ALL Roots combine with some Consonant #0
- Several Roots do NOT combine with Vowel #0 = [26, 28, 33, 34, 73, 82, 108, 114, 126, 152, 157, 158, 163]
- Several Roots do combine ALL Vowels = [13, 23, 64, 72, 79, 81, 96, 107, 113, 115, 133, 147]}
- Only Root #107 combines with ALL Consonants

It was further discovered that Unicode itself is encoded as a multibyte string, using a lower level of base_graphemes 
than root/vowel/consonant diacritics. Some Benglai Graphemes have multiple renderings for the same root/vowel/consonant 
combination, which is implemented in unicode by allowing duplicate base_graphemes within the encoding.

A `combination_matrix()` function was written that permitted a Unicode Visualization of the Bengali Language, 
tabulating the different combinations of grapheme_roots, vowels and consonants.
- [src/jupyter/combination_matrix.py](src/jupyter/combination_matrix.py)


## Technical Challenges

### Kaggle Kernels Only Competition

Solution: [kaggle_compile.py](./kaggle_compile.py) 
 
This codebase is approximately 2k CLOC, and builds upon my previous efforts for the 
[MINST Digit Recognizer](https://github.com/JamesMcGuigan/kaggle-digit-recognizer) competition.

Developing a professionally engineered data pipeline using only a Jupyter Notebook would be impractical, 
and my preferred development workflow is to use the IntelliJ IDE and commit my code to github.

My first attempt involved manually copy and pasting required functions into a Kaggle Script Notebook, but then I
 decided that this process could be automated with a script.
 
[./kaggle_compile.py](./kaggle_compile.py) is a simple python script compiler. It reads in a python executable
 script, parses the `import` headers for any local include files, then recursively builds a dependency tree and 
 concatenates these into single python file (much like an old-school javascript compiler). It can be called with
  either `--save` or `--commit` cli flags to automattically save to disk or commit the result to git.
  
Limitations: the script only works for `from local.module import function` syntax, and does not support `import local
.module` syntax, as calling `module.function()` inside a script would not work with script concatenation. Also the
entire python file for the dependency is included, which does not guarantee the absence of namespace conflicts, but
with awareness of this issue good coding practices it is sufficiently practical for generating Kaggle submissions.
  
There are other more robust solutions to this problem such as [stickytape](https://github.com/mwilliamson/stickytape), 
which allow for module imports, however the code for dependency files is obfuscated into a single line string
variable, which makes for an unreadable Kaggle submission. [./kaggle_compile.py](./kaggle_compile.py) produces
readable and easily editable output.


### Settings File

According to the Kaggle Winning Model Documentation Guidelines 
- https://www.kaggle.com/WinningModelDocumentationGuidelines
 
> settings.json
> This file specifies the path to the train, test, model, and output directories. Here is an example file.
> 
> This is the only place that specifies the path to these directories.
> Any code that is doing I/O should use the appropriate base paths from SETTINGS.json

This was implemented as a python script [src/settings.py](src/settings.py) which defines default settings and
 hyperparameters.


### Different Hyperparameter Configurations for Localhost, Kaggle Interactive and Kaggle Batch Mode

I discovered that it is possible to detect which environment the python code is running in using
```
os.environ.get('KAGGLE_KERNEL_RUN_TYPE', 'Localhost')  # == 'Localhost' | 'Interactive' | 'Batch'
```

This was explored in the following notebook:
- [Jupyter Environment Variable os.environ](notebooks/Jupyter%20Environment%20Variable%20os.environ.ipynb)

The main uses for this are in the [settings.py](src/settings.py) file to specify different environment filesystem paths 
(eg `./input/bengaliai-cv19/` vs `../input/bengaliai-cv19/`) as well as different default timeout and verbosity
 settings. 

A second use is tweak hyperparameter settings in Interactive Mode (ie editing a script in Kaggle), such as setting 
`epochs=0` or `epochs=1` to allow quick sanity checking of the final script and that `submission
.csv` is generated without throwing an exception. Using this flag removes the danger forgetting to revert these
changes and Batch Mode commit accidentally running using debug configuration.


### Modifing Hyperparameters using CLI 

Solution: [src/util/argparse.py](./src/util/argparse.py)

Most of the hyperparameters in this codebase have been abstracted into dictionary arguments. As an alternative to
 editing the code, or using `KAGGLE_KERNEL_RUN_TYPE` it would be nice to be able to directly
 modify them in Localhost mode on the CLI.
 
[src/util/argparse.py](./src/util/argparse.py) provides two useful functions `argparse_from_dict({})` and 
`argparse_from_dicts([{},{}])` that will autogenerate argparse CLI config from a dictionary or list of dictionaries
, with the option to edit them inplace. The result is that useful CLI flags can be implemented with a single line of
 code.
 
This is useful for passing `--timeout 1m` or `--epochs 0` when debugging locally. It can also be used as a basic
 method for hyperparameter search.


### Out of Memory Errors

Early attempts at running this codebase as a Kaggle Script often terminated in `Exit code 137` (Out of Memory), and
that the 16GB of RAM available is insufficient to both load the entire training dataset into memory as well as a
tensorflow model.

This prompted a full investigation into memory profiling and the different methods to read parquet files: 
- [Reading Parquet Files RAM CPU Optimization](notebooks/Reading%20Parquet%20Files%20RAM%20CPU%20Optimization.ipynb)
 
The dataformat of the parquet files means they cannot be read row-by-row, but only in their entirety. 
 
The optimal method chosen was to using `pd.read_parquet()` but to place this inside a generator function. Each
parquet would be read 3 times, but the generator would only store 1/3 of the raw parquet data in memory at any one
time. This reduced peak memory usage from `1.7GB` to `555MB` whilst increasing disk-IO timings from `46s` to `116s
` (`+74s`) per epoch. 

This was considered an acceptable (if overly conservative) RAM/Disk-IO tradeoff, but solved the Out of Memory errors.

Generator Source Code: 
- [src/dataset/ParquetImageDataGenerator.py](src/dataset/ParquetImageDataGenerator.py)

Additional memory optimizations involved using `float16` rather than `float64` during image normalization, and
 rescaling the image size by 50% before feeding it into the neural network model. 


### Notebook Exceeded Allowed Compute

This Kaggle competition has a `120m` maximum runtime constraint.

The simplest solution to this is to write a custom tensorflow callback that sets a timer and exits training before
the deadline. The source code for KaggleTimeoutCallback() is in: 
 - [src/callbacks/KaggleTimeoutCallback.py](src/callbacks/KaggleTimeoutCallback.py)


### Pretrained Model Weights

Another way around the timeout issue is to use pretrained model weights, run on Localhost without a timer, upload these
files to Kaggle as a private dataset, and then Commit the kaggle script using `epochs=0`. 

This requires using the `ModelCheckpoint()` callback to save the weights to a file, `glob2.glob(f"../input
/**/{os.path.basename(model_file)}")` to find the relevant file within the attached Kaggle dataset, and 
`model.load_weights()` to load them again before training. 

This was the method used to generate a score of 0.9396, which took 3h to train to convergence on a Localhost laptop.


### Submission - Notebook Threw Exception

The public test dataset comprises only 12 test images, which is deceptively small. The result is that the Kaggle
  Kernel can compile without exception, yet submitting the final `submission.csv` will generate a `Notebook Threw
  Exception` error.
  
The root cause is that the hidden test dataset is significantly larger, and presumably comparable in size to the
   training dataset. This can cause both out-of-memory errors and timeout errors if the code for generating a 
  `submission.csv` file is not performance optimized.

There is known bug in `model.predict(...)` that results in a memory leak if the function is called in a loop. The
  solution to this is to call `model.predict_on_batch(...)` which was combined with using a generator to read the test
  parquet files, and a `for loop` to manually batch the data before calling `model.predict_on_batch(...)`  
- https://github.com/keras-team/keras/issues/13118

A second issue is that the `submission.csv` has a funny format, which is not a 4 column table, but rather in
 `{image_id}_{column}, {value}` format. My initial naive code implementation used a nested `for loop` to write the
  values of `submission.csv` cell by cell. Rewriting this using a vectorized `df.apply()` solved the 
  `Notebook Threw Exception` error.

Optimized code for generating submission.csv: 
- [src/util/csv.py](src/util/csv.py)


### Model Generating Random Results

The random score of the `sample_submission.csv` is 0.0614, and several of my submission returned this result.

Debugging this discovered several code issues:
- Normalizing the data for training, but not for testing
- Incorrect ordering of keys in `submission.csv` - https://www.kaggle.com/c/human-protein-atlas-image-classification/discussion/69366 
- Training using `loss_weights=sqrt(output_shape.values())`
- Using `tf.keras.backend.set_floatx('float16')` then retraining on pretrained model weights



## Image Preprocessing

The following transforms where conducted as preprocessing:
- Invert:  set background to 0 and text to 255
- Resize:  reduce image size by 50% using `skimage.measure.block_reduce((1,2,2,1), np.max)`
- Rescale: rescale brightness of maximum pixel to 255
- Denoise: set background pixels less than maximum mean (42) to 0
- Center:  crop image background border, and recenter image
- Normalize: Convert pixel value range from `0 - 255` to `0.0 - 1.0`

This provided a clean and standarized format for both training and test data.

An exploration and testing of Image Preprocessing was conducted in Jupyter Notebook
- [Image Processing](notebooks/Image%20Processing.ipynb)

The final code for image transforms
- [src/dataset/Transforms.py](src/dataset/Transforms.py)


### Image Augmentation
`ImageDataGenerator` was used to provide Image Augmentation with the following settings:
 
```
datagen_args = {
    # "rescale":          1./255,  # "normalize": True is default in Transforms
    # "brightness_range": 0.5,   # Preprocessing normalized brightness
    "zoom_range":         0.2,
    "width_shift_range":  0.1,     # Preprocessing already centered image
    "height_shift_range": 0.1,     # Preprocessing already centered image
    "rotation_range":     45/2,
    "shear_range":        45/2,
    "fill_mode":         'constant',
    "cval": 0,
    # "featurewise_center": True,             # No visible effect in plt.imgshow()
    # "samplewise_center": True,              # No visible effect in plt.imgshow()
    # "featurewise_std_normalization": True,  # No visible effect in plt.imgshow() | requires .fit()
    # "samplewise_std_normalization": True,   # No visible effect in plt.imgshow() | requires .fit()
    # "zca_whitening": True,                  # Kaggle, insufficent memory
}
```

Due the memory constraints of not being able to load the entire training dataset into a single dataframe, 
and the performance issues of saving the test/train datasets to a directory tree of image files,
a ImageDataGenerator subclass was written with a custom `ParquetImageDataGenerator.flow_from_parquet()`  
method. This used the generator pattern as described above to read from the parquet files, and run 
preprocessing `Transforms` on batches of data, whilst keep the minimum data possible in RAM.
- [src/dataset/ParquetImageDataGenerator.py](src/dataset/ParquetImageDataGenerator.py)


## Writing Images to Filesystem
- [src/preprocessing/write_images_to_filesystem.py](src/preprocessing/write_images_to_filesystem.py)
- [https://www.kaggle.com/jamesmcguigan/bengali-ai-dataset-as-image-directory](https://www.kaggle.com/jamesmcguigan/bengali-ai-dataset-as-image-directory)

As an experiment, a script was written to write the images in the parquet file to the filesystem after Image
 Preprocessing. 

This took 11 minutes of runtime to complete on Kaggle, generating 200,840 image files.

In theory this would have allowed `ImageDataGenerator.flow_from_dictectory()` to be used, and this directory tree could
 have been imported as an external dataset. On localhost, having 200k files in a single directory tree produces
 performance issues, and this method was abandoned in favour of `ParquetImageDataGenerator.flow_from_parquet()`


## Neural Network Models

### Single Output CNN
- [src/pipelines/simple_triple_df_cnn.py](src/pipelines/simple_triple_df_cnn.py)
- [src/models/SingleOutputCNN.py](src/models/SingleOutputCNN.py)

First attempt, simplest thing that could possibly work, was to port the CNN code used for MINST,
load the data for each parquet file into a pandas DataFrame and train 3 separate neural networks, 
one for each output variable.

The main issue with this simplistic approach is that the data must be read three times, and the entire training
 process takes three times as long.

Unable to obtain a Kaggle Score using this method.
 

### Multi Output CNN
- [src/pipelines/multi_output_df_cnn.py](src/pipelines/multi_output_df_cnn.py)
- [src/models/MultiOutputCNN.py](src/models/MultiOutputCNN.py)

The next evolution was to write a single CNN neural network with multiple outputs, thus reducing training time by 3x.

The model was trained using all 4 of the columns provided in the `train.csv`
- grapheme_root
- vowel_diacritic
- consonant_diacritic
- grapheme


The CNN Architecture uses multiple layers of:
- Conv2D(3x3)
- MaxPooling2D()
- BatchNormalization()
- Flatten()
- Dropout()
- Dense()
- Dropout()


TODO:
- split the grapheme into constituent unicode bytes and retrain using softmax Multi-Hot-Encoding
- Test to see if this method can generate a Kaggle score using only Image Preprocessing 


### ImageDataGenerator CNN
- [src/pipelines/image_data_generator_cnn.py](src/pipelines/image_data_generator_cnn.py)
- [src/models/MultiOutputCNN.py](src/models/MultiOutputCNN.py)
- [https://www.kaggle.com/jamesmcguigan/bengali-ai-imagedatagenerator-cnn?scriptVersionId=30636537](https://www.kaggle.com/jamesmcguigan/bengali-ai-imagedatagenerator-cnn?scriptVersionId=30636537)
- [data_output/scores/0.9396/image_data_generator_cnn-cnns_per_maxpool=3-maxpool_layers=4-dense_layers=2-dense_units=256-regularization=False-global_maxpool=False-submission.log](data_output/scores/0.9396/image_data_generator_cnn-cnns_per_maxpool=3-maxpool_layers=4-dense_layers=2-dense_units=256-regularization=False-global_maxpool=False-submission.log)

This approach used `ImageDataGenerator()` for Image Augmentation with a custom 
`ParquetImageDataGenerator.flow_from_parquet()` subclass function to read the parquet files in a memory optimized
 fashion

The following hyperparameters generated a 0.9396 Kaggle Score
```
model_hparams = {
    "cnns_per_maxpool":   3,
    "maxpool_layers":     4,
    "dense_layers":       2,
    "dense_units":      256,
    "regularization": False,
    "global_maxpool": False,
}
train_hparams = {
    "optimizer":     "RMSprop",
    "scheduler":     "constant",
    "learning_rate": 0.001,
    "best_only":     True,
    "batch_size":    128,     
    "patience":      10,
    "epochs":        999,
    "loss_weights":  False,
}
```

`EarlyStopping(patience=10)` was used with a 75/25 train/test split (3 parquet files for train, 1 for test), and
 took 3.4 hours to converge on Localhost after 77 epochs


Increasing dense_units from `256` -> `512`:
- converged in less epochs: `77` -> `66`
- slightly quicker to train: `3.4h` -> `2.5h`
- had slightly worse accuracy: `0.969` -> `0.9346` (`val_grapheme_root_accuracy`) 

TODO: 
- Run a full hyperparameter search to discover if it is possible to obtain a higher score


### ImageDataGenerator Keras Application

An alternative to a custom designed CNN is to use a published CNN network architecture from `tf.keras.applications`

This requires removing the first and last layers of the application, then adding additional Dense layers before
 splitting the output.

The first application investigated was NASNetMobile, however compared to the custom CNN model: 
- converged in less epochs: `77` -> `53`
- much slower to train: `3.4h` -> `4.15h`
- had much worse accuracy: `0.969` -> `0.873` (`val_grapheme_root_accuracy`) 

TODO:
- Test the full range of `tf.keras.applications`
- Test if running for a fixed number of epochs produces more consistent results


## Hyperparameter Search

Additional Visualizations of Hyperparameter Search
- [notebooks/Hyperparamer Search.ipynb](notebooks/Hyperparamer%20Search.ipynb)

Previous research on MINST produced the following shortlist of Optimizer / Learning Rate / Scheduler combinations 
```
"optimized_scheduler": {
    "Adagrad_triangular": { "learning_rate": 0.1,    "optimizer": "Adagrad",  "scheduler": "CyclicLR_triangular"  },
    "Adagrad_plateau":    { "learning_rate": 0.1,    "optimizer": "Adagrad",  "scheduler": "plateau2"      },
    "Adam_triangular2":   { "learning_rate": 0.01,   "optimizer": "Adam",     "scheduler": "CyclicLR_triangular2" },
    "Nadam_plateau":      { "learning_rate": 0.01,   "optimizer": "Nadam",    "scheduler": "plateau_sqrt"  },
    "Adadelta_plateau":   { "learning_rate": 1.0,    "optimizer": "Adadelta", "scheduler": "plateau10"     },
    "SGD_triangular2":    { "learning_rate": 1.0,    "optimizer": "SGD",      "scheduler": "CyclicLR_triangular2" },
    "RMSprop_constant":   { "learning_rate": 0.001,  "optimizer": "RMSprop",  "scheduler": "constant"      },
}
```
- https://github.com/JamesMcGuigan/kaggle-digit-recognizer

### Optimizers and Learning Rates
- [src/pipelines/image_data_generator_cnn_search.py](src/pipelines/image_data_generator_cnn_search.py)
- [data_output/submissions/image_data_generator_cnn_search_train/results.csv](data_output/submissions/image_data_generator_cnn_search_train/results.csv)

| Time | Epocs | Loss  | Optimizer | Scheduler | Learning Rate | Notes                    |
|-----:|------:|:------|----------:|----------:|:--------------|--------------------------|
| 7580 | 54	   | 0.637 | Adadelta  | plateau10 | 1.000         | Lowest loss, least epochs, middle speed | 
| 9558 | 68	   | 0.759 | Adam      | constant  | 0.001         | Lowest loss with constant optimizer | 
| 4514 | 32	   | 0.818 | RMSProp   | constant  | 0.001         | Fastest with decent loss            | 

### CNN Model Hparams
- [src/pipelines/image_data_generator_cnn_search.py](src/pipelines/image_data_generator_cnn_search.py)
- [data_output/submissions/image_data_generator_cnn_search_model/results.csv](data_output/submissions/image_data_generator_cnn_search_model/results.csv)

A hyperparameter grid search was run varying the number of layers within the CNN model

regularization
- `regularization=False` almost always outperforms `regularization=True`
- `regularization=True` prefers fewer dense units

global_maxpool
- `regularization=True` prefers `global_maxpool=False` (but not vice veras)
- `global_maxpool=True` prefers double the number of `dense_units` and +1 `cnns_per_maxpool`

cnns
- increasing `maxpool_layers` prefers fewer `cnns_per_maxpool` (ideal total CNNs = 15 / 16) 

dense units
- `dense_layers=1` is preferred over `2` or `3`

Fastest with high score 
- maxpool_layers=5 | cnns_per_maxpool=3 | dense_layers=1 | dense_units=256 | global_maxpool=False | regularization=False 

Shortlist:
- maxpool_layers=5 | cnns_per_maxpool=3 | dense_layers=1 | dense_units=512 | global_maxpool=True  | regularization=False
- maxpool_layers=4 | cnns_per_maxpool=4 | dense_layers=1 | dense_units=256 | global_maxpool=False | regularization=False
- maxpool_layers=4 | cnns_per_maxpool=4 | dense_layers=1 | dense_units=256 | global_maxpool=False | regularization=True


Results:
- Optimizing architecture from `3\*4 CNN + 2\*256 Dense` -> `3\*5 CNN + 1\*256 Dense` improves kaggle score `0.8921/0.9396` -> `0.8961/0.9482`
- Adding regularization reduces score from `0.8961/0.9482` -> `0.8797/0.9198`
- Removing `Y+=grapheme` improves private score from `0.8961/0.9482` -> `0.9010/0.9413`

TODO:
- Train Y on unicode grapheme components
- Hyperparameter search on X_transforms + Augmentation params
- Kaggle trained models still timeout, so pretrained models might produce better results  

