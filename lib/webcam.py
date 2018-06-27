#!/usr/bin/env python
# -*- coding: utf-8 -*-
import cv2
import threading

class WebcamVideoStream:
    """
    Reference:
    https://www.pyimagesearch.com/2015/12/21/increasing-webcam-fps-with-python-and-opencv/
    """

    def __init__(self, src, width, height):
        # initialize the video camera stream and read the first frame
        self.stream = cv2.VideoCapture(src)
        if not self.stream.isOpened():
            # camera failed
            raise IOError(("Couldn't open video file or webcam."))
        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        (self.grabbed, self.frame) = self.stream.read()
        # initialize the variable used to indicate if the thread should
        # check camera stream shape
        real_width = int(self.stream.get(3))
        real_height = int(self.stream.get(4))
        print("Start video stream with shape: {},{}".format(real_width, real_height))
        self.running = True

    def __del__(self):
        if self.stream.isOpened():
            self.stream.release()
        return

    def start(self):
        # start the thread to read frames from the video stream
        t = threading.Thread(target=self.update, args=())
        t.setDaemon(True)
        t.start()
        return self

    def update(self):
        try:
            # keep looping infinitely until the stream is closed
            while self.running:
                # otherwise, read the next frame from the stream
                (self.grabbed, self.frame) = self.stream.read()
        except:
            import traceback
            traceback.print_exc()
            self.running = False
        finally:
            # if the thread indicator variable is set, stop the thread
            self.stream.release()
        return

    def read(self):
        # return the frame most recently read
        return self.frame

    def stop(self):
        self.running = False
