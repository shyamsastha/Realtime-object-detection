import multiprocessing
import numpy as np
import cv2
import time
from lib.mpio import start_receiver
from lib.mpvariable import MPVariable
from lib.load_label_map import LoadLabelMap
from tf_utils import visualization_utils_cv2 as vis_util
from skimage import measure

import sys
PY2 = sys.version_info[0] == 2
PY3 = sys.version_info[0] == 3
if PY2:
    import Queue
elif PY3:
    import queue as Queue

def to_layer(background, foreground, background_alpha=1.0, foreground_alpha=0.75, gamma=0):
    '''
    Marge Two Images
    args:
        background: OpenCV Background Image
        foreground: OpenCV Foreground Image
        background_alpha: Background Alpha Value
        foreground_alpha: Foreground Alpha Value
        gamma: Gamma Value
    return:
        result: Marged Image
    '''

    result = cv2.addWeighted(background, background_alpha, foreground, foreground_alpha, gamma)
    return result

def overdraw(background, foreground, mask=None):
    print("back:{} fore:{}".format(background.shape,foreground.shape))
    """
    resize foreground image with keep aspect ratio
    """
    # Select the pasting position with x, y
    rows, cols = foreground.shape[:2]
    f_x = 0

    # By multiplying each BRG channel by alpha_mask, the opaque part of the foreground creates new_background of [0, 0, 0]
    new_background = cv2.merge(list(map(lambda f_x:f_x * mask, cv2.split(background))))
    # Combine images by converting BGRA to BGR and new_background
    background = cv2.add(foreground, background)
    return background

def blending(background, foreground):
    # Add the masked foreground and background.
    #outImage = cv2.add(foreground, background)
    outImage = foreground + background
    return outImage

def visualization(category_index, image, boxes, scores, classes, debug_mode, vis_text, fps_interval,
                  fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.4, fontThickness=1, masks=None,):
    # Visualization of the results of a detection.
    vis_util.visualize_boxes_and_labels_on_image_array(
        image,
        boxes,
        scores,
        classes,
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

def deeplab_visualization(label_names, image, seg_map, debug_mode, vis_text, fps_interval,
                          fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.4, fontThickness=1, use_normalized_coordinates=False):
    rows, cols = image.shape[:2]
    line_thickness = 8
    MIN_AREA             = 500
    color=(0, 255, 0)
    
    """
    DRAW DETECTED BOX AND TEXT
    """
    map_labeled = measure.label(seg_map, connectivity=1)
    for region in measure.regionprops(map_labeled):
        if region.area > MIN_AREA:
            """
            LABEL NAME
            """
            display_str = label_names[seg_map[tuple(region.coords[0])]]

            """
            BOX COORDINATE
            """
            xmin = region.bbox[1]
            xmax = region.bbox[3]
            ymin = region.bbox[0]
            ymax = region.bbox[2]
            if use_normalized_coordinates:
                (left, right, top, bottom) = (xmin * cols, xmax * rows,
                                              ymin * cols, ymax * rows)
            else:
                (left, right, top, bottom) = (xmin, xmax, ymin, ymax)

            """
            DRAW OBJECT BOX
            """
            points = np.array([[left, top], [left, bottom], [right, bottom], [right, top], [left, top]])
            cv2.polylines(image, np.int32([points]),
                          isClosed=False, thickness=line_thickness, color=color, lineType=cv2.LINE_AA)
            """
            CALCULATE STR WIDTH AND HEIGHT
            """
            display_str_height = cv2.getTextSize(text=display_str, fontFace=fontFace, fontScale=fontScale, thickness=fontThickness)[0][1]
            total_display_str_height = (1 + 2 * 0.05) * display_str_height

            if top > total_display_str_height:
                text_bottom = top
            else:
                text_bottom = bottom + total_display_str_height

            [(text_width, text_height), baseLine] = cv2.getTextSize(text=display_str, fontFace=fontFace, fontScale=fontScale, thickness=fontThickness)
            margin = np.ceil(0.05 * text_height)

            """
            DRAW TEXTBOX AND TEXT
            """
            [(text_width, text_height), baseLine] = cv2.getTextSize(text=display_str, fontFace=fontFace, fontScale=fontScale, thickness=fontThickness)
            margin = np.ceil(0.05 * text_height)

            cv2.rectangle(image, (int(left), int(text_bottom - 3 * baseLine - text_height - 2 * margin)), (int(left + text_width), int(text_bottom - baseLine)), color=color, thickness=-1)
            cv2.putText(image, display_str, org=(int(left + margin), int(text_bottom - text_height - margin)), fontFace=fontFace, fontScale=fontScale, thickness=fontThickness, color=(0, 0, 0))
  
            text_bottom -= text_height - 2 * margin


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


