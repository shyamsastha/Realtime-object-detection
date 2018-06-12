#!/usr/bin/env python2
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

logging.basicConfig(level=logging.DEBUG,
                    format='[%(levelname)s] time:%(created).8f pid:%(process)d pn:%(processName)-10s tid:%(thread)d tn:%(threadName)-10s fn:%(funcName)-10s %(message)s',
)


# Used in GPU/CPU process
## LOAD CONFIG PARAMS ##
def read_config():
    logging.debug("enter")
    global cfg
    if (os.path.isfile('config.yml')):
        with open("config.yml", 'r') as ymlfile:
            cfg = yaml.load(ymlfile)
    else:
        with open("config.sample.yml", 'r') as ymlfile:
            cfg = yaml.load(ymlfile)
    return cfg

cfg = read_config()
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

    cfg = read_config()
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

    cfg = read_config()
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

    cfg = read_config()
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
    before_time = time.time()
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
                    cv2.putText(image,"fps: {:.1f}".format(1.0/(time.time() - before_time)), (10,30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (77, 255, 9), 2)
                    before_time = time.time() # start next FPS time count
                cv2.imshow('object_detection', image)
                # Exit Option
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    running.value = False
                    c = visualize_in_conn.recv()
                    break
            else:
                # Exit after max frames if no visualization
                if frame_counter.value%det_interval==0:
                    for box, score, _class in zip(np.squeeze(boxes), np.squeeze(scores), np.squeeze(classes)):
                        if score > det_th:
                            label = category_index[_class]['name']
                            print("label: {}\nscore: {}\nbox: {}".format(label, score, box))
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
    logging.debug("enter")
    cfg = read_config()
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

    sleep_interval = 0.1
    fps_value = 0.0
    try:
        while running.value:
            slept_time = 0
            before_sleep_time = time.time()
            wakeup_time = before_sleep_time + fps_interval
            while running.value:
                time.sleep(sleep_interval)
                now_time = time.time()
                if now_time >= wakeup_time:
                    break
            slept_time = now_time - before_sleep_time
            fps_value = float(frame_counter.value)/float(slept_time)
            print("FPS:{:.1f}, Frames:{} Sec.:{}".format(fps_value, frame_counter.value, slept_time))
            frame_counter.value=0
    except Exception as e:
        import traceback
        traceback.print_exc()

    return

# Used in STOP process
def process_stop():
    logging.debug("enter")
    cfg = read_config()
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
    if not visualize:
        time.sleep(execution_seconds)
        running.value = False
    return

'''
プロセスによる実行関数の振り分け定義
'''
#PROCESS_LIST=['process_gpu','process_cpu','process_visualize','process_fps','process_stop']
PROCESS_LIST=['process_gpu','process_cpu','process_visualize','process_fps','process_stop']
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
    if target == 'process_stop':
        process_stop()
        return 'end '+target

    return


gpu_out_conn, cpu_in_conn = multiprocessing.Pipe()
cpu_out_conn, visualize_in_conn = multiprocessing.Pipe()

running = multiprocessing.Value(ctypes.c_bool,True)
frame_counter = multiprocessing.Value(ctypes.c_int,0)

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
