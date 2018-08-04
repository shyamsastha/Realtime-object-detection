#!/usr/bin/env python
# -*- coding: utf-8 -*-
import cv2
import threading
import time
import os

class VideoReader:
    vid = None
    out = None
    running = False

    def __init__(self):
        return

    def __del__(self):
        if self.vid.isOpened():
            self.vid.release()
        if self.out is not None:
            self.out.release()

        return

    def mkdir(self, path):
        if not os.path.exists(path):
            os.makedirs(path)
        return

    def start(self, src, width, height, output_prefix='output', save_to_movie=False):
        """
        output_1532580366.27.avi
        output_file[:-4] # remove .avi from filename
        """
        output_file = 'movie/' + output_prefix + '_' + str(time.time()) + '.avi'
        # initialize the video camera vid and read the first frame
        self.vid = cv2.VideoCapture(src)
        if not self.vid.isOpened():
            # camera failed
            raise IOError(("Couldn't open video file or webcam."))
        self.vid.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.vid.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.ret, self.frame = self.vid.read()
        if not self.ret:
            self.vid.release()
            raise IOError(("Couldn't open video frame."))

        # initialize the variable used to indicate if the thread should
        # check camera vid shape
        self.real_width = int(self.vid.get(3))
        self.real_height = int(self.vid.get(4))
        print("Start video stream with shape: {},{}".format(self.real_width, self.real_height))
        self.running = True

        """ save to movie """
        if save_to_movie:
            self.mkdir('movie')
            fps = self.vid.get(cv2.CAP_PROP_FPS)
            fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
            self.out = cv2.VideoWriter(output_file, int(fourcc), fps, (int(self.real_width), int(self.real_height)))
        return self

    def getSize(self):
        return (self.real_width, self.real_height)

    def read(self):
        # return the frame most recently read
        self.ret, self.frame = self.vid.read()
        if not self.ret:
            self.stop()
            return None
        return self.frame

    def save(self, frame):
        # save to avi
        self.out.write(frame)
        return

    def stop(self):
        self.running = False
        if self.vid.isOpened():
            self.vid.release()
        if self.out is not None:
            self.out.release()
