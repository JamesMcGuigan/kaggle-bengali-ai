# DOCS: https://www.kaggle.com/WinningModelDocumentationGuidelines
import os

import simplejson

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
        'Localhost':    10,
        'Interactive':  0,
        'Batch':        10,
    }.get(os.environ.get('KAGGLE_KERNEL_RUN_TYPE','Localhost'), 10),

    "loops": {
        'Localhost':   1,
        'Interactive': 1,
        'Batch':       1,
    }.get(os.environ.get('KAGGLE_KERNEL_RUN_TYPE','Localhost'), 1),

    # Timeout = 120 minutes | allow 30 minutes for testing submit | TODO: unsure of KAGGLE_KERNEL_RUN_TYPE on Submit
    "timeout": {
        'Localhost':   "24h",
        'Interactive': "5m",
        'Batch':       "110m",
    }.get(os.environ.get('KAGGLE_KERNEL_RUN_TYPE','Localhost'), "110m")

}

settings['verbose'] = {

    "tensorboard": {
        'Localhost':   True,
        'Interactive': False,
        'Batch':       False,
    }.get(os.environ.get('KAGGLE_KERNEL_RUN_TYPE','Localhost'), False),

    "fit": {
        'Localhost':   1,
        'Interactive': 2,
        'Batch':       2,
    }.get(os.environ.get('KAGGLE_KERNEL_RUN_TYPE','Localhost'), 2)

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

####################
if __name__ == '__main__':
    for dirname in settings['dir'].values():
        try:    os.makedirs(dirname, exist_ok=True)  # BUGFIX: read-only filesystem
        except: pass
    for key,value in settings.items():       print(f"settings['{key}']:".ljust(30), str(value))

    if os.environ.get('KAGGLE_KERNEL_RUN_TYPE'):
        with open('settings.json', 'w') as file:
            print( 'settings', simplejson.dumps(settings, indent=4*' '))
            simplejson.dump(settings, file, indent=4*' ')
