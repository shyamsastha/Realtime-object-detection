# Multi-Threading for Realtime Object Detection

## The first split model was created by @wkelongws.
He did nice work!<br>
For realtime object detection, this is the most important part.<br>
[https://github.com/tensorflow/models/issues/3270](https://github.com/tensorflow/models/issues/3270)<br>
Processing image:<br>
![](./document/ProcessingFlow-P1.png)<br>

## The code has been summarized on github by @GustavZ.
He did nice work!<br>
The code uploaded up by him is easily executable and very nice.<br>

## From here multi-threading begins.
The Hack solution's sess.run() can be divided into threads.<br>
![](./document/ProcessingFlow-P2.png)<br>
As a result, the main thread only checks the queue, and the GPU part and the CPU part continue to operate, respectively.<br>
<br>
Data communication between threads uses queue.<br>
<br>
Normally, because Python is influenced by GIL, multi-threading is not good design.<br>
But those that work with the C library are released from GIL.<br>
Therefore, sess.run() has a merit greater than the disadvantage of multi-threading.<br>

## Multi-Processing FPS.
Because the FPS counter is written in python, it remakes in multi-processing.<br>
Shared variables for multi-processing seem to work faster with ctypes.<br>
![](./document/ProcessingFlow-P3.png)<br>
Design of Multi-processing FPS.<br>
![](./document/ProcessingFlow-P4.png)<br>
Short time FPS is greatly influenced by the progress of CPU Worker.<br>
Therefore, in order to make flat FPS as much as possible, count a long time to get the average FPS.<br>
However, I would like to update FPS in short time.<br>
To achieve this, take a small chunk of FPS and put it in the array.<br>
This is an idea of FPS streaming.<br>
![](./document/ProcessingFlow-P5.png)<br>

## Multi-Processing Visualization.
I made GPU and CPU parts multi-processing and I could not get good results. It was slower than multi-threading.<br>
The slow cause is in interprocess communication.<br>

When input image size is 1280 x 720 on PC, FPS is greatly different depending on whether it is visualized or not. With this large image size, visualization is a big bottleneck.<br>
So I designed to use multi-processing to limit visualization FPS.<br>
<br>
Pipe is in thread because it blocks processing at both transmission and reception.<br>
Data communication between main process's main thread and sender thread or visualization process's main thread and recever thread uses queue because these are to cross the thread.<br>
![](./document/ProcessingFlow-P6.png)<br>
Visualization is very heavy. Therefore, if the receiver queue is full, it drops the data sent. This is VDrop.(Visualization worker's frame drop.)<br>
![](./document/ProcessingFlow-P7.png)<br>
Continuing stacking without dropping the frame will be Out of Memory.<br>

## Current process and thread design.
![](./document/ProcessingFlow-P8.png)<br>

## My failed Multi-Processing design.
![](./document/ProcessingFlow-P9.png)<br>

