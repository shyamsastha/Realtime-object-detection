#!/usr/bin/env python
# -*- coding: utf-8 -*-
import cv2
import threading
import time
import os
import sys
from stat import *

def walktree(dir_path, callback):
    """
    Reference:
    https://stackoverflow.com/questions/3204782/how-to-check-if-a-file-is-a-directory-or-regular-file-in-python

    recursively descend the directory tree rooted at top,
    calling the callback function for each regular file
    """
    for f in os.listdir(dir_path):
        filepath = os.path.join(dir_path, f)
        mode = os.stat(filepath)[ST_MODE]
        if S_ISDIR(mode):
            # It's a directory, recurse into it
            walktree(filepath, callback)
        elif S_ISREG(mode):
            # It's a file
            if filepath.endswith(".jpeg"):
                # call the callback function
                callback(filepath)
            elif filepath.endswith(".jpg"):
                # call the callback function
                callback(filepath)
            elif filepath.endswith(".png"):
                # call the callback function
                callback(filepath)
            else:
                # Unknown file type, print a message
                print('{} - skip'.format(filepath))
        else:
            # Unknown file type, print a message
            print('{} - unknown'.format(filepath))
    return

class ImageReader:

    def __init__(self):
        self.running = False
        self.image_files = []
        self.image_files_len = 0
        self.current_files_index = -1
        self.detection_counter = {}
        return

    def __del__(self):
        return

    def mkdir(self, path):
        if not os.path.exists(path):
            os.makedirs(path)
        return

    def start(self, dir_path, output_dir='output_image', save_to_file=False):
        if not os.path.exists(dir_path):
            raise ValueError('File not found: '+dir_path)
        walktree(dir_path, self.add_image_files)
        self.image_files_len = len(self.image_files)
        self.OUTPUT_DIR = output_dir

        print("Start image reader")
        self.running = True

        # OpenCV imageFormat is either 1 or 0 or -1
        #1 for loading as RGB image (strips alfa component of RGBA file)
        #0 for loading as grayscale image
        #-1 for loading as is (includes alfa component of RGBA file)
        self.imageFormat=1

        """ save to file """
        if save_to_file:
            self.mkdir(output_dir)
        return self

    def getSize(self):
        return (self.real_width, self.real_height)

    def add_image_files(self, filepath):
        self.image_files.append(filepath)
        return

    def getNextFilePath(self):
        self.current_files_index += 1
        if self.image_files_len <= self.current_files_index:
            ret = False
            frame = None
            return ret, frame
        ret = True
        filepath = self.image_files[self.current_files_index]
        return ret, filepath

    def read(self):
        ret, filepath = self.getNextFilePath()
        if not ret:
            self.stop()
            return None, None

        frame = cv2.imread(filepath, self.imageFormat)
        self.real_height, self.real_width = frame.shape[:2]

        return frame, filepath

    def save(self, cv_bgr, filepath):
        dir_path, filename = os.path.split(filepath)
        self.mkdir(self.OUTPUT_DIR+"/"+dir_path)
        # save to file
        cv2.imwrite(self.OUTPUT_DIR+"/"+filepath, cv_bgr)
        return

    def save_detection_image(self, int_label, cv_bgr, filepath):
        self.mkdir(self.OUTPUT_DIR+"/"+str(int_label))

        dir_path, filename = os.path.split(filepath)
        if not filename in self.detection_counter:
            self.detection_counter.update({filename: 0})
        self.detection_counter[filename] += 1
        # remove .jpg/.jpeg/.png and get filename
        if filename.endswith(".jpeg"):
            filehead = filename[:-5]
            filetype = ".jpeg"
        elif filename.endswith(".jpg"):
            filehead = filename[:-4]
            filetype = ".jpg"
        elif filename.endswith(".png"):
            filehead = filename[:-4]
            filetype = ".png"

        # save to file
        cv2.imwrite(self.OUTPUT_DIR+"/"+str(int_label)+"/"+filehead+"_"+str(self.detection_counter[filename])+filetype, cv_bgr)
        return

    def stop(self):
        self.running = False
