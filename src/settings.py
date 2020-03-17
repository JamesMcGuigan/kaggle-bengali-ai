# DOCS: https://www.kaggle.com/WinningModelDocumentationGuidelines
import os

settings = {}

settings['hparam_defaults'] = {
    "optimizer":     "RMSprop",
    "scheduler":     "constant",
    "learning_rate": 0.001,
    "min_lr":        0.001,
    "split":         0.2,
    "batch_size":    128,
    "fraction":      1.0,
    "patience": {
        'Localhost':    5,
        'Interactive':  0,
        'Batch':        5,
    }[os.environ.get('KAGGLE_KERNEL_RUN_TYPE','Localhost')],
    "loops": {
        'Localhost':   1,
        'Interactive': 1,
        'Batch':       1,
    }[os.environ.get('KAGGLE_KERNEL_RUN_TYPE','Localhost')],

    # Timeout = 120 minutes | allow 30 minutes for testing submit | TODO: unsure of KAGGLE_KERNEL_RUN_TYPE on Submit
    "timeout": {
        'Localhost':   "24h",
        'Interactive': "5m",
        'Batch':       "110m",
    }.get(os.environ.get('KAGGLE_KERNEL_RUN_TYPE','Localhost'), "110m")
}

settings['verbose'] = {
    "tensorboard": {
        {
            'Localhost':   True,
            'Interactive': False,
            'Batch':       False,
        }[os.environ.get('KAGGLE_KERNEL_RUN_TYPE','Localhost')]
    },
    "fit": {
        'Localhost':   1,
        'Interactive': 2,
        'Batch':       2,
    }[os.environ.get('KAGGLE_KERNEL_RUN_TYPE','Localhost')]
}

if os.environ.get('KAGGLE_KERNEL_RUN_TYPE'):
    settings['dir'] = {
        "data":        "../input/bengaliai-cv19",
        "features":    "./input_features/bengaliai-cv19/",
        "models":      "./models",
        "submissions": "./",
        "logs":        "./logs",
    }
else:
    settings['dir'] = {
        "data":        "./input/bengaliai-cv19",
        "features":    "./input_features/bengaliai-cv19/",
        "models":      "./data_output/models",
        "submissions": "./data_output/submissions",
        "logs":        "./logs",
    }
for dirname in settings['dir'].values(): os.makedirs(dirname, exist_ok=True)

