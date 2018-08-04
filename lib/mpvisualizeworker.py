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


def visualization(category_index, image, boxes, scores, classes, debug_mode, vis_text, fps_interval,
                  fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.4, fontThickness=1, masks=None,):
    # Visualization of the results of a detection.
    vis_util.visualize_boxes_and_labels_on_image_array(
        image,
        np.squeeze(boxes),
        np.squeeze(classes).astype(np.int32),
        np.squeeze(scores),
        category_index,
        instance_masks=masks,
        use_normalized_coordinates=True,
        line_thickness=8)
    if vis_text:
        display_str = []
        max_text_width = 0
        max_text_height = 0
        if not debug_mode:
            display_str.append("fps: {:.1f}".format(MPVariable.fps.value))
        else:
            display_str.append("fps: {:.1f} 0.2sec".format(MPVariable.fps_snapshot.value))
            display_str.append("fps: {:.1f} {}sec".format(MPVariable.fps.value, fps_interval))

        """ DRAW BLACK BOX AND TEXT """
        [(text_width, text_height), baseLine] = cv2.getTextSize(text=display_str[0], fontFace=fontFace, fontScale=fontScale, thickness=fontThickness)
        x_left = int(baseLine)
        y_top = int(baseLine)
        for i in range(len(display_str)):
            [(text_width, text_height), baseLine] = cv2.getTextSize(text=display_str[i], fontFace=fontFace, fontScale=fontScale, thickness=fontThickness)
            if max_text_width < text_width:
                max_text_width = text_width
            if max_text_height < text_height:
                max_text_height = text_height
        """ DRAW BLACK BOX """
        cv2.rectangle(image, (x_left - 2, int(y_top)), (int(x_left + max_text_width + 2), int(y_top + len(display_str)*max_text_height*1.2+baseLine)), color=(0, 0, 0), thickness=-1)
        """ DRAW FPS, TEXT """
        for i in range(len(display_str)):
            cv2.putText(image, display_str[i], org=(x_left, y_top + int(max_text_height*1.2 + (max_text_height*1.2 * i))), fontFace=fontFace, fontScale=fontScale, thickness=fontThickness, color=(77, 255, 9))

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
        """
        cv2.imshow()
        """

        """
        q_in: {'image':image, 'vis_in_time':time.time()}
        """
        q_in = Queue.Queue()

        """ """ """ """ """ """ """ """ """ """ """
        START RECEIVE THREAD
        """ """ """ """ """ """ """ """ """ """ """
        start_receiver(in_con, q_in, MPVariable.vis_drop_frames)

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

                image = q['image']
                vis_in_time = q['vis_in_time']
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


