from object_detection.utils import label_map_util
import numpy as np
from object_detection.utils import visualization_utils_cv2 as vis_util
from stuff.helper import WebcamVideoStream, SessionWorker

from lib.load_graph import LoadFrozenGraph
from lib.mpvariable import MPVariable
import time
import sys
import cv2
import logging
import tensorflow as tf


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

def without_visualization(category_index, boxes, scores, classes, cur_frame, det_interval, det_th):
    # Exit after max frames if no visualization
    for box, score, _class in zip(np.squeeze(boxes), np.squeeze(scores), np.squeeze(classes)):
        if cur_frame%det_interval==0 and score > det_th:
            label = category_index[_class]['name']
            print("label: {}\nscore: {}\nbox: {}".format(label, score, box))

def visualization(category_index, image, boxes, scores, classes, debug_mode, vis_text, fps_interval):
    # Visualization of the results of a detection.
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
            cv2.putText(image,"fps: {:.1f}".format(MPVariable.fps.value), (10,30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (77, 255, 9), 2)
        else:
            """ FOR PERFORMANCE DEBUG """
            cv2.putText(image,"fps: {:.1f} 0.2sec".format(MPVariable.fps_snapshot.value), (10,30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (77, 255, 9), 2)
            cv2.putText(image,"fps: {:.1f} {}sec".format(MPVariable.fps.value, fps_interval), (10,60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (77, 255, 9), 2)
    return image


def ssd_mobilenet_v1_without_split(cfg, category_index, graph):
    video_input         = cfg['video_input']
    visualize           = cfg['visualize']
    vis_text            = cfg['vis_text']
    max_frames          = cfg['max_frames']
    allow_memory_growth = cfg['allow_memory_growth']
    width               = cfg['width']
    height              = cfg['height']
    det_interval        = cfg['det_interval']
    det_th              = cfg['det_th']
    num_classes         = cfg['num_classes']
    log_device          = cfg['log_device']
    fps_interval        = cfg['fps_interval']
    debug_mode          = cfg['debug_mode']

    # Session Config: allow seperate GPU/CPU adressing and limit memory allocation
    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=log_device)
    config.gpu_options.allow_growth = allow_memory_growth

    # Define Input and Ouput tensors
    image_tensor = graph.get_tensor_by_name('image_tensor:0')
    detection_boxes = graph.get_tensor_by_name('detection_boxes:0')
    detection_scores = graph.get_tensor_by_name('detection_scores:0')
    detection_classes = graph.get_tensor_by_name('detection_classes:0')
    num_detections = graph.get_tensor_by_name('num_detections:0')

    with graph.as_default():
        with tf.Session(config=config) as sess:
            # Start Video Stream
            video_stream = WebcamVideoStream(video_input,width,height).start()
            cur_frame = 0
            is_first_frame_done = False
            print("Press 'q' to Exit")
            print('Starting Detection')
            while video_stream.isActive():
                top_in_time = time.time()
                # default session
                image = video_stream.read()
                image_expanded = np.expand_dims(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), axis=0) # np.expand_dims is faster than []
                proc_in_time = time.time()
                boxes, scores, classes, num = sess.run(
                    [detection_boxes, detection_scores, detection_classes, num_detections],
                    feed_dict={image_tensor: image_expanded})
                proc_out_time = time.time()

                viz_in_time = time.time()
                if visualize:
                    image = visualization(category_index, image, boxes, scores, classes, debug_mode, vis_text, fps_interval)
                    cv2.imshow('object_detection', image)
                    # Exit Option
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                else:
                    without_visualization(category_index, boxes, scores, classes, cur_frame, det_interval, det_th)
                    if cur_frame >= max_frames:
                        break
                    cur_frame += 1
                viz_out_time = time.time()

                proc_time = proc_out_time - proc_in_time
                viz_proc_time = viz_out_time - viz_in_time
                lost_proc_time = viz_out_time - top_in_time - proc_time - viz_proc_time
                total_proc_time = viz_out_time - top_in_time
                MPVariable.viz_proc_time.value += viz_proc_time
                MPVariable.lost_proc_time.value += lost_proc_time
                MPVariable.total_proc_time.value += total_proc_time
                if debug_mode:
                    sys.stdout.write("snapshot FPS:{: ^5.1f} total:{: ^10.5f} proc:{: ^10.5f} viz:{: ^10.5f} lost:{: ^10.5f}\n".format(
                        MPVariable.fps.value, total_proc_time, proc_time, viz_proc_time, lost_proc_time))
                MPVariable.frame_counter.value += 1

    # End everything
    video_stream.stop()
    cv2.destroyAllWindows()
    return

def ssd_mobilenet_v1_with_split(cfg, category_index, graph, score, expand):
    video_input         = cfg['video_input']
    visualize           = cfg['visualize']
    vis_text            = cfg['vis_text']
    max_frames          = cfg['max_frames']
    allow_memory_growth = cfg['allow_memory_growth']
    width               = cfg['width']
    height              = cfg['height']
    det_interval        = cfg['det_interval']
    det_th              = cfg['det_th']
    num_classes         = cfg['num_classes']
    log_device          = cfg['log_device']
    fps_interval        = cfg['fps_interval']
    debug_mode          = cfg['debug_mode']

    # Session Config: allow seperate GPU/CPU adressing and limit memory allocation
    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=log_device)
    config.gpu_options.allow_growth = allow_memory_growth
    #config.gpu_options.per_process_gpu_memory_fraction = 0.01 # 80MB memory is enough to run on TX2

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

    """
    START WORKER THREAD
    """
    gpu_worker = SessionWorker("GPU", graph, config)
    cpu_worker = SessionWorker("CPU", graph, config)
    gpu_opts = [score_out, expand_out]
    cpu_opts = [detection_boxes, detection_scores, detection_classes, num_detections]
    gpu_counter = 0
    cpu_counter = 0
    sleep_interval = 0.005
    """
    PUT DUMMY DATA INTO GPU WORKER FOR JIT DONE
    """
    gpu_feeds = {image_tensor:  [np.zeros((300, 300, 3))]}
    gpu_extras = {}
    gpu_worker.put_sess_queue(gpu_opts, gpu_feeds, gpu_extras)
    """
    PUT DUMMY DATA INTO CPU WORKER FOR JIT DONE
    """
    score = np.zeros((1, 1917, num_classes))
    expand = np.zeros((1, 1917, 1, 4))
    cpu_feeds = {score_in: score, expand_in: expand}
    cpu_extras = {}
    cpu_worker.put_sess_queue(cpu_opts, cpu_feeds, cpu_extras)
    """
    WAIT UNTIL JIT-COMPILE DONE
    """
    while True:
        g = gpu_worker.get_result_queue()
        if g is None:
            time.sleep(sleep_interval)
        else:
            break
    while True:
        c = cpu_worker.get_result_queue()
        if c is None:
            time.sleep(sleep_interval)
        else:
            break

    # Start Video Stream
    video_stream = WebcamVideoStream(video_input, width, height).start()
    cur_frame = 0
    is_first_frame_done = False
    print("Press 'q' to Exit")
    print('Starting Detection')
    while video_stream.isActive():
        top_in_time = time.time()
        # split model in seperate gpu and cpu session threads
        if gpu_worker.is_sess_empty(): # must need for speed
            # read video frame, expand dimensions and convert to rgb
            image = video_stream.read()
            image_expanded = np.expand_dims(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), axis=0) # np.expand_dims is faster than []
            #image_expanded = np.expand_dims(image, axis=0) # BGR image for input. Of couse, bad accuracy in RGB trained model, but speed up.
            # put new queue
            gpu_feeds = {image_tensor: image_expanded}
            if visualize:
                gpu_extras = {"image":image, "top_in_time":top_in_time} # for visualization frame
            else:
                gpu_extras = {"top_in_time":top_in_time}
            gpu_worker.put_sess_queue(gpu_opts, gpu_feeds, gpu_extras)
        g = gpu_worker.get_result_queue()
        if g is None:
            # gpu thread has no output queue. ok skip, let's check cpu thread.
            gpu_counter += 1
        else:
            # gpu thread has output queue.
            gpu_counter = 0
            score, expand, extras = g["results"][0], g["results"][1], g["extras"]

            if cpu_worker.is_sess_empty():
                # When cpu thread has no next queue, put new queue.
                # else, drop gpu queue.
                cpu_feeds = {score_in: score, expand_in: expand}
                cpu_extras = extras
                cpu_worker.put_sess_queue(cpu_opts, cpu_feeds, cpu_extras)

        c = cpu_worker.get_result_queue()
        if c is None:
            # cpu thread has no output queue. ok, nothing to do. continue
            cpu_counter += 1
            time.sleep(sleep_interval)
            continue # If CPU RESULT has not been set yet, no fps update
        else:
            cpu_counter = 0
            boxes, scores, classes, num, extras = c["results"][0], c["results"][1], c["results"][2], c["results"][3], c["extras"]

            viz_in_time = time.time()
            if visualize:
                image = extras["image"]
                image = visualization(category_index, image, boxes, scores, classes, debug_mode, vis_text, fps_interval)
                cv2.imshow('object_detection', image)
                # Exit Option
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                without_visualization(category_index, boxes, scores, classes, cur_frame, det_interval, det_th)
                if cur_frame >= max_frames:
                    break
                cur_frame += 1
            viz_out_time = time.time()

            if MPVariable.frame_counter.value == 0 and MPVariable.fps.value > 0:
                sleep_interval = 1.0 / MPVariable.fps.value / 10.0
                gpu_worker.sleep_interval = sleep_interval
                cpu_worker.sleep_interval = sleep_interval

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
            MPVariable.frame_counter.value += 1

    # End everything
    gpu_worker.stop()
    cpu_worker.stop()
    video_stream.stop()
    cv2.destroyAllWindows()
    return

def ssd_mobilenet_v1(cfg):
    """
    LOAD FROZEN_GRAPH
    """
    load_frozen_graph = LoadFrozenGraph(cfg)
    graph, score, expand = load_frozen_graph.load_graph()

    """
    LOAD LABEL MAP
    """
    category_index = load_labelmap(cfg)
    split_model = cfg['split_model']

    print("Building Graph")
    try:
        if split_model:
            ssd_mobilenet_v1_with_split(cfg, category_index, graph, score, expand)
        else:
            ssd_mobilenet_v1_without_split(cfg, category_index, graph)
    except Exception as e:
        import traceback
        traceback.print_exc()
    finally:
        MPVariable.running.value = False
        return

