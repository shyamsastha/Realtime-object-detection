from concurrent import futures
import multiprocessing
import ctypes
import logging
import time
import sys
import os
import yaml
import numpy as np

from lib.mpfps import FPS

"""
Execute ssd_mobilenet_v1, ssd_mobilenet_v2, ssdlite_mobilenet_v2
Repository:
https://github.com/naisy/realtime_object_detection

About repogitory: Forked from GustavZ's github.
https://github.com/GustavZ/realtime_object_detection

Updates:
- Support ssd_mobilenet_v2, ssdlite_mobilenet_v2

- Add Multi-Processing visualization. : Detection and visualization are asynchronous.

- Drop unused files.

- Parallel run to complete JIT. : Improve startup time from 90sec to 78sec.
- Add time details.             : To understand the processing time well.

- Separate split and non-split code.     : Remove unused session from split code.
- Remove Session from load frozen graph. : Reduction of memory usage.

- Flexible sleep_interval.          : Maybe speed up on high spec PC.
- FPS separate to multi-processing. : Speed up.
- FPS streaming calculation.        : Flat fps.
- FPS is average of fps_interval.   : Flat fps. (in fps_stream)
- FPS updates every 0.2 sec.        : Flat fps. (in fps_snapshot)

- solve: Multiple session cannot launch problem. tensorflow.python.framework.errors_impl.InternalError: Failed to create session.
"""

def load_config():
    """
    LOAD CONFIG FILE
    Convert config.yml to DICT.
    """
    cfg = None
    if (os.path.isfile('config.yml')):
        with open("config.yml", 'r') as ymlfile:
            cfg = yaml.load(ymlfile)
    else:
        raise FileNotFoundError(("File not found: config.yml"))
    return cfg

def log_format(debug_mode):
    """
    LOG FORMAT
    If debug_mode, show detailed log
    """
    if debug_mode:
        np.set_printoptions(precision=5, suppress=True, threshold=np.inf)  # suppress scientific float notation
        logging.basicConfig(level=logging.DEBUG,
                            format='[%(levelname)s] time:%(created).8f pid:%(process)d pn:%(processName)-10s tid:%(thread)d tn:%(threadName)-10s fn:%(funcName)-10s %(message)s',
        )
    return

def download_model():
    """
    Download Model form TF's Model Zoo
    """
    model_file = model_name + '.tar.gz'
    download_base = 'http://download.tensorflow.org/models/object_detection/'
    if not os.path.isfile(model_path):
        print('Model not found. Downloading it now.')
        opener = urllib.request.URLopener()
        opener.retrieve(download_base + model_file, model_file)
        tar_file = tarfile.open(model_file)
        for file in tar_file.getmembers():
            file_name = os.path.basename(file.name)
            if 'frozen_inference_graph.pb' in file_name:
                tar_file.extract(file, os.getcwd() + '/models/')
        os.remove(os.getcwd() + '/' + model_file)
    else:
        print('Model found. Proceed.')

def main():
    try:
        """
        LOAD SETUP VARIABLES
        """
        cfg = load_config()
        debug_mode = cfg['debug_mode']
        model_type = cfg['model_type']

        """
        LOG FORMAT MODE
        """
        log_format(debug_mode)

        """
        START DETECTION, FPS, FPS PRINT
        """
        fps = FPS(cfg)
        fps_counter_proc = fps.start_counter()
        fps_console_proc = fps.start_console()
        if model_type == 'ssd_mobilenet_v1':
            from lib.detection_ssd_mobilenet_v1 import SSDMobileNetV1
            ssd = SSDMobileNetV1()
            ssd.start(cfg)
        elif model_type == 'ssd_mobilenet_v2':
            from lib.detection_ssd_mobilenet_v2 import SSDMobileNetV2
            ssd = SSDMobileNetV2()
            ssd.start(cfg)
        else:
            raise IOError(("Unknown model_type."))
        fps_counter_proc.join()
        fps_console_proc.join()
    except Exception as e:
        import traceback
        traceback.print_exc()
    finally:
        pass

if __name__ == '__main__':
    main()

