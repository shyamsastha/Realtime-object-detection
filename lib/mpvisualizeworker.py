import multiprocessing
import numpy as np
import cv2
import time
from lib.mpio import start_receiver
from lib.mpvariable import MPVariable
from lib.load_label_map import LoadLabelMap
from tf_utils import visualization_utils_cv2 as vis_util

import sys
PY2 = sys.version_info[0] == 2
PY3 = sys.version_info[0] == 3
if PY2:
    import Queue
elif PY3:
    import queue as Queue


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


class MPVisualizeWorker():

    def __init__(self, cfg, in_con):
        """ """ """ """ """ """ """ """ """ """ """
        START WORKER PROCESS
        """ """ """ """ """ """ """ """ """ """ """
        m = multiprocessing.Process(target=self.execution, args=(cfg, in_con))
        m.start()
        return

    def execution(self, cfg, in_con):
        q_in = Queue.Queue()

        """ """ """ """ """ """ """ """ """ """ """
        GET CONFIG
        """ """ """ """ """ """ """ """ """ """ """
        VISUALIZE            = cfg['visualize']
        MAX_VIS_FPS          = cfg['max_vis_fps']
        VIS_TEXT             = cfg['vis_text']
        FPS_INTERVAL         = cfg['fps_interval']
        SPLIT_MODEL          = cfg['split_model']
        DEBUG_MODE           = cfg['debug_mode']
        """ """

        """ """ """ """ """ """ """ """ """ """ """
        LOAD LABEL MAP
        """ """ """ """ """ """ """ """ """ """ """
        llm = LoadLabelMap()
        category_index = llm.load_label_map(cfg)
        """ """

        """ """ """ """ """ """ """ """ """ """ """
        START RECEIVE THREAD
        """ """ """ """ """ """ """ """ """ """ """
        start_receiver(in_con, q_in, MPVariable.vis_drop_frames)

        vis_frames = 0
        try:
            while MPVariable.running.value:
                if q_in.empty():
                    time.sleep(MPVariable.sleep_interval.value)
                    continue

                q = q_in.get(block=False)
                if q is None:
                    MPVariable.running.value = False
                    q_in.task_done()
                    break

                #if q['mode'] == MPVariable.DROP_MODE:
                #    q_in.task_done()
                #    continue

                vis_in_time = time.time()
                boxes, scores, classes, num, extras = q['results'][0], q['results'][1], q['results'][2], q['results'][3], q['extras']
                image = extras['image']
                image = visualization(category_index, image, boxes, scores, classes, DEBUG_MODE, VIS_TEXT, FPS_INTERVAL)
                """
                SHOW
                """
                cv2.imshow("Object Detection", image)
                # Press q to quit
                if cv2.waitKey(1) & 0xFF == 113: #ord('q'):
                    q_in.task_done()
                    break

                q_in.task_done()
                MPVariable.vis_frame_counter.value += 1
                vis_out_time = time.time()
                """
                PROCESSING TIME
                """
                vis_proc_time = vis_out_time - vis_in_time
                MPVariable.vis_proc_time.value += vis_proc_time
        except:
            import traceback
            traceback.print_exc()
        finally:
            MPVariable.running.value = False

        return


