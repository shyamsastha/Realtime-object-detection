#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 21 12:01:40 2017
@author: GustavZ

This code based on object_detection.py of GustavZ.
https://github.com/GustavZ/realtime_object_detection

Updates:

- FPS separate to multi-processing
- FPS streaming calculation
- FPS is average of fps_interval (in fps_stream)
- FPS updates every 0.2 sec. (in fps_snapshot)

- solve: Multiple session cannot launch problem. tensorflow.python.framework.errors_impl.InternalError: Failed to create session.
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
from stuff.helper import WebcamVideoStream, SessionWorker

from concurrent import futures
import multiprocessing
import ctypes
import logging
import time
import sys

import copy_reg
import types

def _pickle_method(m):
    if m.im_self is None:
        return getattr, (m.im_class, m.im_func.func_name)
    else:
        return getattr, (m.im_self, m.im_func.func_name)

copy_reg.pickle(types.MethodType, _pickle_method)

def load_config():
    logging.debug("enter")
    global cfg
    if (os.path.isfile('config.yml')):
        with open("config.yml", 'r') as ymlfile:
            cfg = yaml.load(ymlfile)
    else:
        with open("config.sample.yml", 'r') as ymlfile:
            cfg = yaml.load(ymlfile)
    return cfg

cfg = load_config()
video_input         = cfg['video_input']
visualize           = cfg['visualize']
vis_text            = cfg['vis_text']
max_frames          = cfg['max_frames']
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
debug_mode          = cfg['debug_mode']

if debug_mode:
    np.set_printoptions(precision=5, suppress=True, threshold=np.inf)  # suppress scientific float notation
    logging.basicConfig(level=logging.DEBUG,
                        format='[%(levelname)s] time:%(created).8f pid:%(process)d pn:%(processName)-10s tid:%(thread)d tn:%(threadName)-10s fn:%(funcName)-10s %(message)s',
    )

# Download Model form TF's Model Zoo
def download_model():
    model_file = model_name + '.tar.gz'
    download_base = 'http://download.tensorflow.org/models/object_detection/'
    if not os.path.isfile(model_path):
        print('Model not found. Downloading it now.')
        opener = urllib.request.URLopener()
        opener.retrieve(download_base + model_file, model_file)
        tar_file = tarfile.open(model_file)
        for file in tar_file.getmembers():
          file_name = os.path.basename(file.name)
          if 'toy_frozen_inference_graph.pb' in file_name:
            tar_file.extract(file, os.getcwd() + '/models/')
        os.remove(os.getcwd() + '/' + model_file)
    else:
        print('Model found. Proceed.')

# helper function for split model
def _node_name(n):
  if n.startswith("^"):
    return n[1:]
  else:
    return n.split(":")[0]

# Load a (frozen) Tensorflow model into memory.
def load_frozenmodel():
    print('Loading frozen model into memory')
    if not split_model:
        detection_graph = tf.Graph()
        with detection_graph.as_default():
          od_graph_def = tf.GraphDef()
          with tf.gfile.GFile(model_path, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
        return detection_graph, None, None

    else:
        # load a frozen Model and split it into GPU and CPU graphs
        # Hardcoded for ssd_mobilenet
        input_graph = tf.Graph()
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = allow_memory_growth
        with tf.Session(graph=input_graph, config=config):
            if ssd_shape == 600:
                shape = 7326
            else:
                shape = 1917
            score = tf.placeholder(tf.float32, shape=(None, shape, num_classes), name="Postprocessor/convert_scores")
            expand = tf.placeholder(tf.float32, shape=(None, shape, 1, 4), name="Postprocessor/ExpandDims_1")
            for node in input_graph.as_graph_def().node:
                if node.name == "Postprocessor/convert_scores":
                    score_def = node
                if node.name == "Postprocessor/ExpandDims_1":
                    expand_def = node

        detection_graph = tf.Graph()
        with detection_graph.as_default():
          od_graph_def = tf.GraphDef()
          with tf.gfile.GFile(model_path, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            dest_nodes = ['Postprocessor/convert_scores','Postprocessor/ExpandDims_1']

            edges = {}
            name_to_node_map = {}
            node_seq = {}
            seq = 0
            for node in od_graph_def.node:
              n = _node_name(node.name)
              name_to_node_map[n] = node
              edges[n] = [_node_name(x) for x in node.input]
              node_seq[n] = seq
              seq += 1

            for d in dest_nodes:
              assert d in name_to_node_map, "%s is not in graph" % d

            nodes_to_keep = set()
            next_to_visit = dest_nodes[:]
            while next_to_visit:
              n = next_to_visit[0]
              del next_to_visit[0]
              if n in nodes_to_keep:
                continue
              nodes_to_keep.add(n)
              next_to_visit += edges[n]

            nodes_to_keep_list = sorted(list(nodes_to_keep), key=lambda n: node_seq[n])

            nodes_to_remove = set()
            for n in node_seq:
              if n in nodes_to_keep_list: continue
              nodes_to_remove.add(n)
            nodes_to_remove_list = sorted(list(nodes_to_remove), key=lambda n: node_seq[n])

            keep = graph_pb2.GraphDef()
            for n in nodes_to_keep_list:
              keep.node.extend([copy.deepcopy(name_to_node_map[n])])

            remove = graph_pb2.GraphDef()
            remove.node.extend([score_def])
            remove.node.extend([expand_def])
            for n in nodes_to_remove_list:
              remove.node.extend([copy.deepcopy(name_to_node_map[n])])

            with tf.device('/gpu:0'):
              tf.import_graph_def(keep, name='')
            with tf.device('/cpu:0'):
              tf.import_graph_def(remove, name='')

        return detection_graph, score, expand


def load_labelmap():
    print('Loading label map')
    label_map = label_map_util.load_labelmap(label_path)
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=num_classes, use_display_name=True)
    category_index = label_map_util.create_category_index(categories)
    return category_index


def detection(detection_graph, category_index, score, expand):
    print("Building Graph")
    # Session Config: allow seperate GPU/CPU adressing and limit memory allocation
    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=log_device)
    config.gpu_options.allow_growth=allow_memory_growth
    with detection_graph.as_default():
        with tf.Session(graph=detection_graph,config=config) as sess:
            # Define Input and Ouput tensors
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
            detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
            detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
            detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name('num_detections:0')
            if split_model:
                score_out = detection_graph.get_tensor_by_name('Postprocessor/convert_scores:0')
                expand_out = detection_graph.get_tensor_by_name('Postprocessor/ExpandDims_1:0')
                score_in = detection_graph.get_tensor_by_name('Postprocessor/convert_scores_1:0')
                expand_in = detection_graph.get_tensor_by_name('Postprocessor/ExpandDims_1_1:0')
                # Threading
                gpu_worker = SessionWorker("GPU",detection_graph,config)
                cpu_worker = SessionWorker("CPU",detection_graph,config)
                gpu_opts = [score_out, expand_out]
                cpu_opts = [detection_boxes, detection_scores, detection_classes, num_detections]
                gpu_counter = 0
                cpu_counter = 0
            # Start Video Stream
            video_stream = WebcamVideoStream(video_input,width,height).start()
            cur_frame = 0
            print("Press 'q' to Exit")
            print('Starting Detection')
            while video_stream.isActive():
                # actual Detection
                if split_model:
                    # split model in seperate gpu and cpu session threads
                    if gpu_worker.is_sess_empty():
                        # read video frame, expand dimensions and convert to rgb
                        image = video_stream.read()
                        image_expanded = np.expand_dims(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), axis=0)
                        # put new queue
                        gpu_feeds = {image_tensor: image_expanded}
                        if visualize:
                            gpu_extras = image # for visualization frame
                        else:
                            gpu_extras = None
                        gpu_worker.put_sess_queue(gpu_opts,gpu_feeds,gpu_extras)

                    g = gpu_worker.get_result_queue()
                    if g is None:
                        # gpu thread has no output queue. ok skip, let's check cpu thread.
                        gpu_counter += 1
                    else:
                        # gpu thread has output queue.
                        gpu_counter = 0
                        score,expand,image = g["results"][0],g["results"][1],g["extras"]

                        if cpu_worker.is_sess_empty():
                            # When cpu thread has no next queue, put new queue.
                            # else, drop gpu queue.
                            cpu_feeds = {score_in: score, expand_in: expand}
                            cpu_extras = image
                            cpu_worker.put_sess_queue(cpu_opts,cpu_feeds,cpu_extras)

                    c = cpu_worker.get_result_queue()
                    if c is None:
                        # cpu thread has no output queue. ok, nothing to do. continue
                        cpu_counter += 1
                        time.sleep(0.005)
                        continue # If CPU RESULT has not been set yet, no fps update
                    else:
                        cpu_counter = 0
                        boxes, scores, classes, num, image = c["results"][0],c["results"][1],c["results"][2],c["results"][3],c["extras"]
                else:
                    # default session
                    image = video_stream.read()
                    image_expanded = np.expand_dims(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), axis=0)
                    boxes, scores, classes, num = sess.run(
                            [detection_boxes, detection_scores, detection_classes, num_detections],
                            feed_dict={image_tensor: image_expanded})

                # Visualization of the results of a detection.
                if visualize:
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
                            cv2.putText(image,"fps: {:.1f}".format(fps.value), (10,30),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (77, 255, 9), 2)
                        else:
                            """ FOR PERFORMANCE DEBUG """
                            cv2.putText(image,"fps: {:.1f} 0.2sec".format(fps_spike.value), (10,30),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (77, 255, 9), 2)
                            cv2.putText(image,"fps: {:.1f} {}sec".format(fps.value, fps_interval), (10,60),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (77, 255, 9), 2)
                    cv2.imshow('object_detection', image)
                    # Exit Option
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                else:
                    # Exit after max frames if no visualization
                    for box, score, _class in zip(np.squeeze(boxes), np.squeeze(scores), np.squeeze(classes)):
                        if cur_frame%det_interval==0 and score > det_th:
                            label = category_index[_class]['name']
                            print("label: {}\nscore: {}\nbox: {}".format(label, score, box))
                    if cur_frame >= max_frames:
                        break
                    cur_frame += 1
                frame_counter.value += 1

    # End everything
    if split_model:
        gpu_worker.stop()
        cpu_worker.stop()
    video_stream.stop()
    cv2.destroyAllWindows()


def object_detection():
    try:
        download_model()
        graph, score, expand = load_frozenmodel()
        category = load_labelmap()
        detection(graph, category, score, expand)
    except Exception as e:
        import traceback
        traceback.print_exc()
    finally:
        running.value = False
    return

# Used in FPS process
def process_fps():
    """
    frame_counter.value: Frame counter value shared by processes.
    update fps by 0.2 sec
    """
    logging.debug("enter")

    cfg = load_config()
    fps_interval = cfg['fps_interval']
    debug_mode = cfg['debug_mode']

    sleep_interval = 1.0 / 50.0 # Wakeup and time checks N times per sec.
    update_interval = 1.0 / 5.0 # FPS calculate N times per sec. (must update_interval >= sleep_interval)
    fps_stream_length = fps_interval # FPS stream seconds length. FPS is the frames per second processed during the most recent this time.
    fps_stream = [] # Array of fps_snapshot during fps_stream_length
    fps_snapshot = None # One array (frames_in_interval, interval_seconds, realtime) per update_interval

    if debug_mode:
        """ FOR PERFORMANCE DEBUG """
        spike_fps = 0
        min_spike_fps = 10000
        max_spike_fps = 0
        min_spike_snapshot = []
        max_spike_snapshot = []
        """ """
    try:
        launch_time = time.time()
        while running.value and frame_counter.value == 0:
            time.sleep(sleep_interval)
        print("Time to first image:{}".format(time.time() - launch_time))

        previos_work_time = time.time()
        while running.value:
            time.sleep(sleep_interval)
            now_time = time.time()
            if now_time >= previos_work_time + update_interval:
                ### FPS update by update_interval ###
                snapshot_frames = frame_counter.value
                frame_counter.value = 0

                """
                FPS stream
                """
                snapshot_seconds = now_time - previos_work_time
                fps_snapshot = (snapshot_frames, snapshot_seconds, now_time)
                fps_stream += [fps_snapshot]
                if debug_mode:
                    """
                    FOR PERFORMANCE DEBUG
                    FPS snapshot calculation
                    """
                    spike_fps = snapshot_frames/snapshot_seconds
                    fps_spike.value = spike_fps # FPS of snapshot. for visualize
                    if min_spike_fps >= spike_fps:
                        min_spike_fps = spike_fps
                        min_spike_snapshot += [fps_snapshot]
                        print("min_spike:{:.1f} {}".format(spike_fps, fps_snapshot))
                    if max_spike_fps <= spike_fps:
                        max_spike_fps = spike_fps
                        max_spike_snapshot += [fps_snapshot]
                        print("max_spike:{:.1f} {}".format(spike_fps, fps_snapshot))
                    """ """

                while running.value:
                    (min_frame, min_seconds, min_time) = fps_stream[0]
                    if now_time - min_time > fps_stream_length:
                        """
                        drop old snapshot
                        """
                        fps_stream.pop(0)
                    else:
                        """
                        goto FPS calculate
                        """
                        break
                if(len(fps_stream) > 0):
                    """
                    FPS streaming calculation
                    count frames and seconds in stream
                    """
                    np_fps_stream = np.array(fps_stream)
                    np_fps_stream = np_fps_stream[:,:2]
                    np_fps = np.sum(np_fps_stream, axis=0) # [total_frames, total_seconds] duaring fps_stream_length

                    """
                    insert local values to shared variables
                    """
                    fps_frames.value = int(np_fps[0]) # for console output
                    fps_seconds.value = np_fps[1]     # for console output
                    fps.value = np_fps[0]/np_fps[1]   # for visualize and console
                else:
                    fps_frames.value = 0
                    fps_seconds.value = -1 # for toooooooo slow fps check. if -1 sec appears, fps_stream_length should set more long time.
                    fps.value = 0
                previos_work_time = now_time
    except Exception as e:
        import traceback
        traceback.print_exc()
    finally:
        running.value = False
        if debug_mode:
            print("min_spike_fps:{:.1f}".format(min_spike_fps))
            print("{}".format(min_spike_snapshot))
            print("max_spike_fps:{:.1f}".format(max_spike_fps))
            print("{}".format(max_spike_snapshot))

    return

# Used in FPS console process
def process_fps_console():
    """
    print fps by fps_interval sec.
    """
    logging.debug("enter")
    cfg = load_config()
    fps_interval = cfg['fps_interval']

    sleep_interval = 1.0 / 50.0 # Wakeup and time checks N times per sec.

    try:
        while running.value and frame_counter.value == 0:
            time.sleep(sleep_interval)

        previos_work_time = time.time()
        while running.value:
            time.sleep(sleep_interval)
            now_time = time.time()
            if now_time >= previos_work_time + fps_interval:
                """
                FPS console by fps_interval
                """
                frames = fps_frames.value
                seconds = fps_seconds.value
                print("FPS:{:.1f} Frames:{} Seconds:{}".format(fps.value, fps_frames.value, fps_seconds.value))
                previos_work_time = now_time
    except Exception as e:
        import traceback
        traceback.print_exc()
    finally:
        running.value = False
    return


PROCESS_LIST=['process_fps','process_fps_console']
def do_process(target):
    logging.debug("enter")

    if target == 'process_fps':
        process_fps()
        return 'end '+target
    if target == 'process_fps_console':
        process_fps_console()
        return 'end '+target

    return

running = multiprocessing.Value(ctypes.c_bool,True)
frame_counter = multiprocessing.Value(ctypes.c_int,0)
fps = multiprocessing.Value(ctypes.c_float,0.0)
fps_frames = multiprocessing.Value(ctypes.c_int,0)
fps_seconds = multiprocessing.Value(ctypes.c_float,0.0)
if debug_mode:
    fps_spike = multiprocessing.Value(ctypes.c_float,0.0) # FOR PAFORMANCE DEBUG

def main():
    try:
        with futures.ProcessPoolExecutor(max_workers=len(PROCESS_LIST)) as executer:
            mappings = {executer.submit(do_process,pname): pname for pname in PROCESS_LIST}
            """ start object_detection """
            object_detection()
            for i in futures.as_completed(mappings):
                target = mappings[i]
                result = i.result()
                print(result)

    except Exception as e:
        print('error! executer failed.')
        import traceback
        traceback.print_exc()
    finally:
        print("executer end")


if __name__ == '__main__':
    main()
