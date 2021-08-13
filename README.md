# paddle_inference_ros
### 介绍
该功能包可以帮助开发者在ROS中使用Paddle inference部署基于飞桨的CV模型。
### Description
This package is in order to helping developers using paddle inference to deploy deep learning CV models bases on paddlepaddle in ROS.

### Software Architecture
- paddle_inference_ros(ros package)
    - scripts
        - camera.py(camera_node)
        - pp_infer.py(ppinfer_node)
        - download_model.sh

### Requirements
- ubuntu 18.04
- ROS Melodic
- python3.6.9(with system)
- paddlepaddle-gpu 2.1.1+ (install [paddle-inference prebulided whl](https://www.paddlepaddle.org.cn/documentation/docs/zh/guides/09_hardware_support/hardware_info_cn.html#paddle-inference) or you can also follow [my blog](https://blog.csdn.net/qq_45779334/article/details/118611953) to install it.)

## Get Start
### 1.Rebulid cv_bridge based on python3
```
$ mkdir -p paddle_ros_ws/src && cd paddle_ros_ws/src
$ catkin_init_workspace
$ git clone https://gitee.com/irvingao/vision_opencv.git
$ cd ../
$ catkin_make install -DPYTHON_EXECUTABLE=/usr/bin/python3
```
```
$ vim ~/.bashrc
```
Add:
```
source ~/paddle_ros_ws/devel/setup.bash
source ~/paddle_ros_ws/install/setup.bash --extend
```
Test:
```
$ python3
```
```
import cv_bridge
from cv_bridge.boost.cv_bridge_boost import getCvType
```
**The package of cv_bridge has been built successfully if shows as follow:**
```
Python 3.6.9 (default, Jan 26 2021, 15:33:00) 
[GCC 8.4.0] on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> import cv_bridge
>>> from cv_bridge.boost.cv_bridge_boost import getCvType
>>> 
```

### 2.Bulid paddle_inference_ros package
```
cd src
$ git clone https://gitee.com/irvingao/paddle_inference_ros.git
$ cd paddle_inference_ros/scripts
$ chmod +x *
$ cd ../../..
$ catkin_make -DPYTHON_EXECUTABLE=/usr/bin/python3
```

### 3.Run Detection
Run These commands in seperate terminal:
```
$ roscore
$ cd src/paddle_inference_ros/scripts $$ ./download_model.sh
$ rosrun paddle_inference_ros camera.py
$ rosrun paddle_inference_ros pp_infer.py
```