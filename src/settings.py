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
    "patience":      10,
    "fraction":      1.0,
    "loops":         2,
}
settings['verbose'] = {
    "fit": {
        'Localhost':   1,
        'Interactive': 2,
        'Batch':       0,
    }[os.environ.get('KAGGLE_KERNEL_RUN_TYPE','Localhost')]
}

if os.environ.get('KAGGLE_KERNEL_RUN_TYPE'):
    settings['dir'] = {
        "data":        "../input/bengaliai-cv19",
        "models":      "./models",
        "submissions": "./",
        "logs":        "./logs",
    }
else:
    settings['dir'] = {
        "data":        "./input/bengaliai-cv19",
        "models":      "./data_output/models",
        "submissions": "./data_output/submissions",
        "logs":        "./logs",
    }
for dirname in settings['dir'].values(): os.makedirs(dirname, exist_ok=True)

