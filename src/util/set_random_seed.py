# Source: https://stackoverflow.com/questions/32419510/how-to-get-reproducible-results-in-keras
# NOTE:   This code almost but doesn't completely produce deterministic results - hopefully good enough for development

# DOCS:   https://keras.io/getting-started/faq/#how-can-i-obtain-reproducible-results-using-keras-during-development
# Moreover, when using the TensorFlow backend and running on a GPU, some operations have non-deterministic outputs,
# in particular tf.reduce_sum(). This is due to the fact that GPUs run many operations in parallel,
# so the order of execution is not always guaranteed. Due to the limited precision of floats,
# even adding several numbers together may give slightly different results depending on the order in which you add them.
# You can try to avoid the non-deterministic operations, but some may be created automatically by TensorFlow to compute
# the gradients, so it is much simpler to just run the code on the CPU.
# For this, you can set the CUDA_VISIBLE_DEVICES environment variable to an empty string, for example:

RANDOM_SEED = 42
import os
import random

import numpy as np
import tensorflow as tf

os.environ['PYTHONHASHSEED'] = str(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

