#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Execute ssd_mobilenet_v1.
Repository:
https://github.com/naisy/realtime_object_detection

This code based on GustavZ's github.
https://github.com/GustavZ/realtime_object_detection
"""
import numpy as np
import os
import tensorflow as tf
import copy
import yaml
import cv2
import tarfile
import six.moves.urllib as urllib
from tensorflow.core.framework import graph_pb2

# Protobuf Compilation (once necessary)
#os.system('protoc object_detection/protos/*.proto --python_out=.')

from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils_cv2 as vis_util
from stuff.helper import WebcamVideoStream
from lib.load_graph import LoadFrozenGraph
from lib.mpvariable import MPVariable
from lib.mpfps import FPS

import logging
import time

from concurrent import futures
import multiprocessing
import ctypes

import sys
import types
PY2 = sys.version_info[0] == 2
PY3 = sys.version_info[0] == 3
if PY2:
    import copy_reg
elif PY3:
    import copyreg as copy_reg

def _pickle_method(m):
    if m.im_self is None:
        return getattr, (m.im_class, m.im_func.func_name)
    else:
        return getattr, (m.im_self, m.im_func.func_name)

copy_reg.pickle(types.MethodType, _pickle_method)

# Used in processes
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

# Used in Main process
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



# Used in Visualize process
def load_labelmap(cfg):
    """
    LOAD LABEL MAP
    """
    logging.debug("enter")
    print('Loading label map')
    label_path = cfg['label_path']
    num_classes = cfg['num_classes']

    label_map = label_map_util.load_labelmap(label_path)
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=num_classes, use_display_name=True)
    category_index = label_map_util.create_category_index(categories)
    return category_index


# Used in GPU process
def process_gpu():
    logging.debug("enter")

    cfg = load_config()
    video_input         = cfg['video_input']
    visualize           = cfg['visualize']
    vis_text            = cfg['vis_text']
    execution_seconds   = cfg['execution_seconds']
    width               = cfg['width']
    height              = cfg['height']
    fps_interval        = cfg['fps_interval']
    allow_memory_growth = cfg['allow_memory_growth']
    det_interval        = cfg['det_interval']
    det_th              = cfg['det_th']
    model_name          = cfg['model_name']
    model_path          = cfg['model_path']
    label_path          = cfg['label_path']
    num_classes         = cfg['num_classes']
    split_model         = cfg['split_model']
    log_device          = cfg['log_device']
    ssd_shape           = cfg['ssd_shape']

    print("GPU - Building Graph")
    load_frozen_graph = LoadFrozenGraph(cfg)
    graph, score, expand = load_frozen_graph.load_graph()

    # Session Config: allow seperate GPU/CPU adressing and limit memory allocation
    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=log_device)
    config.gpu_options.allow_growth=allow_memory_growth
    with graph.as_default():
        with tf.Session(config=config) as sess:
            # Define Input and Ouput tensors
            image_tensor = graph.get_tensor_by_name('image_tensor:0')
            detection_boxes = graph.get_tensor_by_name('detection_boxes:0')
            detection_scores = graph.get_tensor_by_name('detection_scores:0')
            detection_classes = graph.get_tensor_by_name('detection_classes:0')
            num_detections = graph.get_tensor_by_name('num_detections:0')
            score_out = graph.get_tensor_by_name('Postprocessor/convert_scores:0')
            expand_out = graph.get_tensor_by_name('Postprocessor/ExpandDims_1:0')
            score_in = graph.get_tensor_by_name('Postprocessor/convert_scores_1:0')
            expand_in = graph.get_tensor_by_name('Postprocessor/ExpandDims_1_1:0')
            gpu_opts = [score_out, expand_out]

            video_stream = WebcamVideoStream(video_input, width, height).start()
            print('Starting Detection')
            try:
                while video_stream.isActive():
                    top_in_time = time.time()
                    if not MPVariable.running.value:
                        break
                    # read video frame, expand dimensions and convert to rgb
                    image = video_stream.read()
                    image_expanded = np.expand_dims(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), axis=0)
                    gpu_feeds = {image_tensor: image_expanded}
                    if visualize:
                        gpu_extras = {"image":image, "top_in_time":top_in_time} # for visualization frame
                    else:
                        gpu_extras = {"top_in_time":top_in_time}
                    # sess.run
                    gpu_extras.update({"GPU_in_time":time.time()})
                    results = sess.run(gpu_opts, feed_dict=gpu_feeds)
                    gpu_extras.update({"GPU_out_time":time.time()})
                    # send result to conn
                    gpu_out_conn.send({"results":results,"extras":gpu_extras})
            except Exception as e:
                import traceback
                traceback.print_exc()
            finally:
                MPVariable.running.value = False
                video_stream.stop()
                gpu_out_conn.close()
                cpu_in_conn.close()
                cpu_out_conn.close()
                visualize_in_conn.close()

# Used in CPU process
def process_cpu():
    logging.debug("enter")

    cfg = load_config()
    video_input         = cfg['video_input']
    visualize           = cfg['visualize']
    vis_text            = cfg['vis_text']
    execution_seconds   = cfg['execution_seconds']
    width               = cfg['width']
    height              = cfg['height']
    fps_interval        = cfg['fps_interval']
    allow_memory_growth = cfg['allow_memory_growth']
    det_interval        = cfg['det_interval']
    det_th              = cfg['det_th']
    model_name          = cfg['model_name']
    model_path          = cfg['model_path']
    label_path          = cfg['label_path']
    num_classes         = cfg['num_classes']
    split_model         = cfg['split_model']
    log_device          = cfg['log_device']
    ssd_shape           = cfg['ssd_shape']

    print("CPU - Building Graph")
    load_frozen_graph = LoadFrozenGraph(cfg)
    graph, score, expand = load_frozen_graph.load_graph()

    # Session Config: allow seperate GPU/CPU adressing and limit memory allocation
    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=log_device)
    config.gpu_options.allow_growth=allow_memory_growth
    with graph.as_default():
        with tf.Session(config=config) as sess:
            # Define Input and Ouput tensors
            image_tensor = graph.get_tensor_by_name('image_tensor:0')
            detection_boxes = graph.get_tensor_by_name('detection_boxes:0')
            detection_scores = graph.get_tensor_by_name('detection_scores:0')
            detection_classes = graph.get_tensor_by_name('detection_classes:0')
            num_detections = graph.get_tensor_by_name('num_detections:0')
            score_out = graph.get_tensor_by_name('Postprocessor/convert_scores:0')
            expand_out = graph.get_tensor_by_name('Postprocessor/ExpandDims_1:0')
            score_in = graph.get_tensor_by_name('Postprocessor/convert_scores_1:0')
            expand_in = graph.get_tensor_by_name('Postprocessor/ExpandDims_1_1:0')
            cpu_opts = [detection_boxes, detection_scores, detection_classes, num_detections]

            #skip_counter = 0
            try:
                while MPVariable.running.value:
                    g = cpu_in_conn.recv()
                    if g is None:
                        break
                    score, expand, cpu_extras = g["results"][0], g["results"][1], g["extras"]
                    cpu_feeds = {score_in: score, expand_in: expand}
                    cpu_extras.update({"CPU_in_time":time.time()})
                    results = sess.run(cpu_opts, feed_dict=cpu_feeds)
                    cpu_extras.update({"CPU_out_time":time.time()})
                    # send result to conn
                    cpu_out_conn.send({"results":results, "extras":cpu_extras})
                    #frame_counter.value += 1
            except Exception as e:
                import traceback
                traceback.print_exc()
            finally:
                MPVariable.running.value = False
                gpu_out_conn.close()
                cpu_in_conn.close()
                cpu_out_conn.close()
                visualize_in_conn.close()
    return

# Used in VISUALIZE process
def process_visualize():
    logging.debug("enter")

    cfg = load_config()
    visualize           = cfg['visualize']
    vis_text            = cfg['vis_text']
    fps_interval        = cfg['fps_interval']
    det_interval        = cfg['det_interval']
    det_th              = cfg['det_th']
    debug_mode          = cfg['debug_mode']

    category_index = load_labelmap(cfg)
    cur_frame = 0
    try:
        while MPVariable.running.value:
            c = visualize_in_conn.recv()
            if c is None:
                break
            boxes, scores, classes, num, extras = c["results"][0], c["results"][1], c["results"][2], c["results"][3], c["extras"]
            # Visualization of the results of a detection.
            viz_in_time = time.time()
            if visualize:
                image = extras["image"]
                vis_util.visualize_boxes_and_labels_on_image_array(
                    image,
                    np.squeeze(boxes),
                    np.squeeze(classes).astype(np.int32),
                    np.squeeze(scores),
                    category_index,
                    use_normalized_coordinates=True,
                    line_thickness=8)
                if vis_text:
                    if not debug_mode:
                        cv2.putText(image, "fps: {:.1f}".format(MPVariable.fps.value), (10,30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (77, 255, 9), 2)
                    else:
                        """ FOR PERFORMANCE DEBUG """
                        cv2.putText(image, "fps: {:.1f} 0.2sec".format(MPVariable.fps_snapshot.value), (10,30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (77, 255, 9), 2)
                        cv2.putText(image, "fps: {:.1f} {}sec".format(MPVariable.fps.value, fps_interval), (10,60),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (77, 255, 9), 2)
                cv2.imshow('object_detection', image)
                # Exit Option
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    MPVariable.running.value = False
                    c = visualize_in_conn.recv()
                    break
            else:
                # Exit after max frames if no visualization
                if cur_frame%det_interval==0:
                    for box, score, _class in zip(np.squeeze(boxes), np.squeeze(scores), np.squeeze(classes)):
                        if score > det_th:
                            label = category_index[_class]['name']
                            print("label: {}\nscore: {}\nbox: {}".format(label, score, box))
                cur_frame += 1
            viz_out_time = time.time()
            MPVariable.frame_counter.value += 1
            top_in_time = extras["top_in_time"]
            gpu_proc_time = extras["GPU_out_time"] - extras["GPU_in_time"]
            cpu_proc_time = extras["CPU_out_time"] - extras["CPU_in_time"]
            viz_proc_time = viz_out_time - viz_in_time
            lost_proc_time = viz_out_time - top_in_time - gpu_proc_time - cpu_proc_time - viz_proc_time
            total_proc_time = viz_out_time - top_in_time
            MPVariable.gpu_proc_time.value += gpu_proc_time
            MPVariable.cpu_proc_time.value += cpu_proc_time
            MPVariable.viz_proc_time.value += viz_proc_time
            MPVariable.lost_proc_time.value += lost_proc_time
            MPVariable.total_proc_time.value += total_proc_time
            if debug_mode:
                sys.stdout.write("snapshot FPS:{: ^5.1f} total:{: ^10.5f} gpu:{: ^10.5f} cpu:{: ^10.5f} viz:{: ^10.5f} lost:{: ^10.5f}\n".format(
                    MPVariable.fps.value, total_proc_time, gpu_proc_time, cpu_proc_time, viz_proc_time, lost_proc_time))

    except Exception as e:
        import traceback
        traceback.print_exc()
    finally:
        MPVariable.running.value = False
        cv2.destroyAllWindows()
        gpu_out_conn.close()
        cpu_in_conn.close()
        cpu_out_conn.close()
        visualize_in_conn.close()

    return


# Used in STOP process
def process_stop():
    logging.debug("enter")
    cfg = load_config()
    execution_seconds = cfg['execution_seconds']

    sleep_interval = 1.0 / 1.0 # Wakeup and time checks N times per sec.
    if not visualize:
        while MPVariable.running.value and MPVariable.frame_counter.value == 0:
            """ wait until the first frame done. """
            time.sleep(sleep_interval)
        start_time = time.time()
        while MPVariable.running.value:
            time.sleep(sleep_interval)
            now_time = time.time()
            if execution_seconds <= now_time - start_time:
                break
        MPVariable.running.value = False
    return

'''
process function list
'''
PROCESS_LIST=['process_gpu', 'process_cpu', 'process_visualize', 'process_stop']
def do_process(target):
    logging.debug("enter")

    if target == 'process_gpu':
        process_gpu()
        return 'end '+target
    if target == 'process_cpu':
        process_cpu()
        return 'end '+target
    if target == 'process_visualize':
        process_visualize()
        return 'end '+target
    if target == 'process_stop':
        process_stop()
        return 'end '+target

    return


gpu_out_conn, cpu_in_conn = multiprocessing.Pipe()
cpu_out_conn, visualize_in_conn = multiprocessing.Pipe()

def main():
    logging.debug("enter")
    try:
        """
        LOAD SETUP VARIABLES
        """
        cfg = load_config()
        debug_mode = cfg['debug_mode']

        """
        LOG FORMAT MODE
        """
        log_format(debug_mode)

        """
        START DETECTION, FPS, FPS PRINT
        """
        fps = FPS(cfg)
        t = fps.start()

        """
        START OBJECT DETECTION
        """
        with futures.ProcessPoolExecutor(max_workers=len(PROCESS_LIST)) as executer:
            mappings = {executer.submit(do_process,pname): pname for pname in PROCESS_LIST}
            for i in futures.as_completed(mappings):
                target = mappings[i]
                result = i.result()
                print(result)
        t.join()

    except Exception as e:
        print('error! executer failed.')
        import traceback
        traceback.print_exc()
    finally:
        MPVariable.running.value = False
        gpu_out_conn.close()
        cpu_in_conn.close()
        cpu_out_conn.close()
        visualize_in_conn.close()
        print("executer end")

    return


if __name__ == '__main__':
    main()
