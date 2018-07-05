# Tensorflow realtime_object_detection on Jetson TX2/TX1

## About this repository
forked from GustavZ/realtime_object_detection: [https://github.com/GustavZ/realtime_object_detection](https://github.com/GustavZ/realtime_object_detection)
And focused on ssd_mobilenet_v1.

## Getting Started:
- login Jetson TX2 `ssh -C -Y ubuntu@xxx.xxx.xxx.xxx`
- edit `config.yml` for your environment. (Ex. video_input: 0 # for PC)
- run `python run_ssd_mobilenet_v1.py` realtime object detection (Multi-Threading)
- wait few minuts.
- Multi-Threading is better performance than Multi-Processing. Multi-Processing bottleneck is interprocess communication.
<br />

## Requirements:
```
pip install --upgrade pyyaml
```
Also, OpenCV >= 3.1 and Tensorflow >= 1.4 (1.6 is good)

## config.yml
##### Camera
This is OpenCV argument.
* USB Webcam on PC
```
video_input: 0
```
* USB Webcam on TX2
```
video_input: 1
```
* Onboard camera on TX2
```
video_input: "nvcamerasrc ! video/x-raw(memory:NVMM), width=(int)1280, height=(int)720,format=(string)I420, framerate=(fraction)30/1 ! nvvidconv flip-method=0 ! video/x-raw, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink"
```

#####  Without Visualization
I do not know why, but in TX2 force_gpu_compatible: True it will be faster.
* on TX2
```
force_gpu_compatible: True
visualize: False
```
* on PC
```
force_gpu_compatible: False
visualize: False
```

##### With Visualization
Visualization is heavy. Visualization FPS possible to limit.<br>
Display FPS: Detection FPS.<br>
* default is with Single-Processing and show every frames.
```
visualize: True
vis_worker: False
max_vis_fps: 0
vis_text: True
```
* Visualization FPS limit with Single-Processing
```
visualize: True
vis_worker: False
max_vis_fps: 30
vis_text: True
```
* Visualization FPS limit with Multi-Processing
```
visualize: True
vis_worker: True
max_vis_fps: 30
vis_text: True
```

## Console Log
```
FPS:25.8  Frames:130 Seconds: 5.04248   | 1FRAME total: 0.11910   cap: 0.00013   gpu: 0.03837   cpu: 0.02768   lost: 0.05293   send: 0.03834   | VFPS:25.4  VFrames:128 VDrops: 1 
```
FPS: detection fps. average fps of fps_interval (5sec). <br>
Frames: detection frames in fps_interval. <br>
Seconds: fps_interval running time. <br>

<hr>

1FRAME<br>
total: 1 frame's processing time. 0.1 means delay and 10 fps if it is single-threading. In multi-threading, this value means delay. <br>
cap: time of capture camera image and transform for model input. <br>
gpu: sess.run() time of gpu part. <br>
cpu: sess.run() time of cpu part. <br>
lost: time of overhead, something sleep etc. <br>
send: time of multi-processing queue, block and pipe time. <br>

<hr>

VFPS: visualization fps. <br>
VFrames: visualization frames in fps_interval. <br>
VDrops: When multi-processing visualization is bottleneck, drops. <br>

## Updates:
- Add Multi-Processing visualization. : Detection and visualization are asynchronous.

- Drop unused files.

- Add force_gpu_compatible option. : ssd_mobilenet_v1_coco 34.5 FPS without vizualization 1280x720 on TX2.

- Multi-Processing version corresponds to python 3.6 and python 2.7.
- Launch speed up.              : Improve startup time from 90sec to 78sec.
- Add time details.             : To understand the processing time well.

- Separate split and non-split code.     : Remove unused session from split code.
- Remove Session from load frozen graph. : Reduction of memory usage.

- Flexible sleep_interval.          : Maybe speed up on high performance PC.
- FPS separate to multi-processing. : Speed up.
- FPS streaming calculation.        : Flat fps.
- FPS is average of fps_interval.   : Flat fps.
- FPS updates every 0.2 sec.        : Flat fps.

- solve: Multiple session cannot launch problem. tensorflow.python.framework.errors_impl.InternalError: Failed to create session.

## My Setup:
* Jetson TX2
  * JetPack 3.2/3.2.1
    * Python 3.6
    * OpenCV 3.4.1/Tensorflow 1.6.0
    * OpenCV 3.4.1/Tensorflow 1.6.1
    * OpenCV 3.4.1/Tensorflow 1.7.0 (slow)
    * OpenCV 3.4.1/Tensorflow 1.7.1 (slow)
    * OpenCV 3.4.1/Tensorflow 1.8.0 (slow)
  * JetPack 3.1
    * Python 3.6
    * OpenCV 3.3.1/Tensorflow 1.4.1
    * OpenCV 3.4.0/Tensorflow 1.5.0
    * OpenCV 3.4.1/Tensorflow 1.6.0
    * OpenCV 3.4.1/Tensorflow 1.6.1 (Main)
* Jetson TX1
  * JetPack 3.2
    * Python 3.6
    * OpenCV 3.4.1/Tensorflow 1.6.0
<br />

## NVPMODEL
| Mode | Mode Name | Denver 2 | Frequency | ARM A57 | Frequency | GPU Frequency |
|:--|:--|:--|:--|:--|:--|:--|
| 0 | Max-N | 2 | 2.0 GHz | 4 | 2.0 GHz | 1.30 GHz |
| 1 | Max-Q | 0 | | 4 | 1.2 GHz | 0.85 GHz |
| 2 | Max-P Core-All | 2 | 1.4 GHz | 4 | 1.4 GHz | 1.12 GHz |
| 3 | Max-P ARM | 0 | | 4 | 2.0 GHz | 1.12 GHz |
| 4 | Max-P Denver | 2 | 2.0 GHz | 0 | | 1.12 GHz |

Max-N
```
sudo nvpmodel -m 0
sudo ./jetson_clocks.sh
```

Max-P ARM(Default)
```
sudo nvpmodel -m 3
sudo ./jetson_clocks.sh
```

Show current mode
```
sudo nvpmodel -q --verbose
```

## Current max Performance on `ssd_mobilenet_v1` (with visualization 160x120):
| FPS | Multi | Mode | CPU | Watt | Ampere | Volt-Ampere | Model | classes |
|:--|:--|:--|:--|:--|:--|:--|:--|:--|
| 40 | Multi-Threading | Max-N | 27-55% | 15.6W | 0.27A | 27.8VA | roadsign_frozen_inference_graph_v1_2nd_4k.pb | 4 |
| 36 | Multi-Threading | Max-P ARM | 50-59% | 12.1W | 0.21A | 21.9VA | roadsign_frozen_inference_graph_v1_2nd_4k.pb | 4 |
| 35 | Multi-Processing | Max-N | 0-64% | 14.7W | 0.25A | 25.4VA | roadsign_frozen_inference_graph_v1_2nd_4k.pb | 4 |
| 33 | Multi-Processing | Max-P ARM | 49-55% | 11.6W | 0.20A | 20.1VA | roadsign_frozen_inference_graph_v1_2nd_4k.pb | 4 |

TX1 Multi-Threading is 25-26 FPS.

![](./document/ssd_mobilenet_160x120.png)<br>


## Youtube
#### Robot Car and Realtime Object Detection
[![TX2](https://img.youtube.com/vi/FoRKFw6xoAY/1.jpg)](https://www.youtube.com/watch?v=FoRKFw6xoAY)

#### Object Detection vs Semantic Segmentation on TX2
[![TX2](https://img.youtube.com/vi/p4EeF0LGcw8/1.jpg)](https://www.youtube.com/watch?v=p4EeF0LGcw8)
#### Realtime Object Detection on TX2
[![TX2](https://img.youtube.com/vi/554GqG21c8M/1.jpg)](https://www.youtube.com/watch?v=554GqG21c8M)
#### Realtime Object Detection on TX1
[![TX1](https://img.youtube.com/vi/S4tozDI5ncY/3.jpg)](https://www.youtube.com/watch?v=S4tozDI5ncY)

Movie's FPS is little bit slow down. Because run ssd_movilenet_v1 with desktop capture.<br>
Capture command:<br>
```
gst-launch-1.0 -v ximagesrc use-damage=0 ! nvvidconv ! 'video/x-raw(memory:NVMM),alignment=(string)au,format=(string)I420,framerate=(fraction)25/1,pixel-aspect-ratio=(fraction)1/1' ! omxh264enc !  'video/x-h264,stream-format=(string)byte-stream' ! h264parse ! avimux ! filesink location=capture.avi
```

## Training ssd_mobilenet with own data
[https://github.com/naisy/train_ssd_mobilenet](https://github.com/naisy/train_ssd_mobilenet)