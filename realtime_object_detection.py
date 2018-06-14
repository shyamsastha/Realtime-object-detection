#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This code based on object_detection.py of GustavZ.
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
import logging
import time

from concurrent import futures
import multiprocessing
import ctypes


# Used in GPU/CPU process
## LOAD CONFIG PARAMS ##
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
debug_mode          = cfg['debug_mode']

if debug_mode:
    np.set_printoptions(precision=5, suppress=True, threshold=np.inf)  # suppress scientific float notation
    logging.basicConfig(level=logging.DEBUG,
                        format='[%(levelname)s] time:%(created).8f pid:%(process)d pn:%(processName)-10s tid:%(thread)d tn:%(threadName)-10s fn:%(funcName)-10s %(message)s',
    )


# Used in Main process
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
          if 'frozen_inference_graph.pb' in file_name:
            tar_file.extract(file, os.getcwd() + '/models/')
        os.remove(os.getcwd() + '/' + model_file)
    else:
        print('Model found. Proceed.')

# Used in GPU/CPU process
# helper function for split model
def _node_name(n):
  if n.startswith("^"):
    return n[1:]
  else:
    return n.split(":")[0]

# Used in GPU/CPU process
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


# Used in Visualize process
def load_labelmap():
    print('Loading label map')
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

    print("Building Graph")
    detection_graph, score, expand = load_frozenmodel()

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
            score_out = detection_graph.get_tensor_by_name('Postprocessor/convert_scores:0')
            expand_out = detection_graph.get_tensor_by_name('Postprocessor/ExpandDims_1:0')
            score_in = detection_graph.get_tensor_by_name('Postprocessor/convert_scores_1:0')
            expand_in = detection_graph.get_tensor_by_name('Postprocessor/ExpandDims_1_1:0')
            gpu_opts = [score_out, expand_out]

            video_stream = WebcamVideoStream(video_input,width,height).start()
            print('Starting Detection')
            #skip_counter = 0
            try:
                while video_stream.isActive():
                    if not running.value:
                        break
                    # read video frame, expand dimensions and convert to rgb
                    image = video_stream.read()
                    image_expanded = np.expand_dims(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), axis=0)
                    gpu_feeds = {image_tensor: image_expanded}
                    if visualize:
                        gpu_extras = image  # for visualization frame
                    else:
                        gpu_extras = None
                    # sess.run
                    results = sess.run(gpu_opts,feed_dict=gpu_feeds)
                    # send result to conn
                    gpu_out_conn.send({"results":results,"extras":gpu_extras})
            except Exception as e:
                import traceback
                traceback.print_exc()
            finally:
                running.value = False
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

    print("Building Graph")
    detection_graph, score, expand = load_frozenmodel()

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
            score_out = detection_graph.get_tensor_by_name('Postprocessor/convert_scores:0')
            expand_out = detection_graph.get_tensor_by_name('Postprocessor/ExpandDims_1:0')
            score_in = detection_graph.get_tensor_by_name('Postprocessor/convert_scores_1:0')
            expand_in = detection_graph.get_tensor_by_name('Postprocessor/ExpandDims_1_1:0')
            cpu_opts = [detection_boxes, detection_scores, detection_classes, num_detections]

            #skip_counter = 0
            try:
                while running.value:
                    g = cpu_in_conn.recv()
                    if g is None:
                        break
                    score,expand,cpu_extras = g["results"][0],g["results"][1],g["extras"]
                    cpu_feeds = {score_in: score, expand_in: expand}
                    results = sess.run(cpu_opts,feed_dict=cpu_feeds)
                    # send result to conn
                    cpu_out_conn.send({"results":results,"extras":cpu_extras})
                    #frame_counter.value += 1
            except Exception as e:
                import traceback
                traceback.print_exc()
            finally:
                running.value = False
                gpu_out_conn.close()
                cpu_in_conn.close()
                cpu_out_conn.close()
                visualize_in_conn.close()
    return

# Used in VISUALIZE process
def process_visualize():
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

    category_index = load_labelmap()
    cur_frame = 0
    try:
        while running.value:
            c = visualize_in_conn.recv()
            if c is None:
                break
            boxes, scores, classes, num, image = c["results"][0],c["results"][1],c["results"][2],c["results"][3],c["extras"]
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
                    running.value = False
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
            frame_counter.value += 1
    except Exception as e:
        import traceback
        traceback.print_exc()
    finally:
        running.value = False
        cv2.destroyAllWindows()
        gpu_out_conn.close()
        cpu_in_conn.close()
        cpu_out_conn.close()
        visualize_in_conn.close()

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
                    fps_seconds.value = -1 # for toooooooo slow fps check. if -1 sec appears, fps_stream_length should more long time.
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


# Used in STOP process
def process_stop():
    logging.debug("enter")
    cfg = load_config()
    execution_seconds   = cfg['execution_seconds']

    sleep_interval = 1.0 / 1.0 # Wakeup and time checks N times per sec.
    if not visualize:
        while running.value and frame_counter.value == 0:
            """ wait until the first frame done. """
            time.sleep(sleep_interval)
        start_time = time.time()
        while running.value:
            time.sleep(sleep_interval)
            now_time = time.time()
            if execution_seconds <= now_time - start_time:
                break
        running.value = False
    return

'''
process function list
'''
PROCESS_LIST=['process_gpu','process_cpu','process_visualize','process_fps','process_fps_console','process_stop']
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
    if target == 'process_fps':
        process_fps()
        return 'end '+target
    if target == 'process_fps_console':
        process_fps_console()
        return 'end '+target
    if target == 'process_stop':
        process_stop()
        return 'end '+target

    return


gpu_out_conn, cpu_in_conn = multiprocessing.Pipe()
cpu_out_conn, visualize_in_conn = multiprocessing.Pipe()

running = multiprocessing.Value(ctypes.c_bool,True)
frame_counter = multiprocessing.Value(ctypes.c_int,0)
fps = multiprocessing.Value(ctypes.c_float,0.0)
fps_frames = multiprocessing.Value(ctypes.c_int,0)
fps_seconds = multiprocessing.Value(ctypes.c_float,0.0)
if debug_mode:
    fps_spike = multiprocessing.Value(ctypes.c_float,0.0) # FOR PAFORMANCE DEBUG

def main():
    logging.debug("enter")
    download_model()

    try:
        with futures.ProcessPoolExecutor(max_workers=len(PROCESS_LIST)) as executer:
            mappings = {executer.submit(do_process,pname): pname for pname in PROCESS_LIST}
            for i in futures.as_completed(mappings):
                target = mappings[i]
                result = i.result()
                print(result)

    except Exception as e:
        print('error! executer failed.')
        import traceback
        traceback.print_exc()
    finally:
        running.value = False
        gpu_out_conn.close()
        cpu_in_conn.close()
        cpu_out_conn.close()
        visualize_in_conn.close()
        print("executer end")

    return


if __name__ == '__main__':
    main()
