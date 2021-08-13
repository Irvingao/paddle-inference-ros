#!/usr/bin/env python3
# coding:utf-8

import cv2
import numpy as np

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

from paddle.inference import Config
from paddle.inference import PrecisionType
from paddle.inference import create_predictor
import yaml
import time

# ————————————————图像预处理函数———————————————— #

def resize(img, target_size):
    """resize to target size"""
    if not isinstance(img, np.ndarray):
        raise TypeError('image type is not numpy.')
    im_shape = img.shape
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])
    im_scale_x = float(target_size) / float(im_shape[1])
    im_scale_y = float(target_size) / float(im_shape[0])
    img = cv2.resize(img, None, None, fx=im_scale_x, fy=im_scale_y)
    return img

def normalize(img, mean, std):
    img = img / 255.0
    mean = np.array(mean)[np.newaxis, np.newaxis, :]
    std = np.array(std)[np.newaxis, np.newaxis, :]
    img -= mean
    img /= std
    return img

def preprocess(img, img_size):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    img = resize(img, img_size)
    resize_img = img
    img = img[:, :, ::-1].astype('float32')  # bgr -> rgb
    img = normalize(img, mean, std)
    img = img.transpose((2, 0, 1))  # hwc -> chw
    return img[np.newaxis, :], resize_img

# ——————————————————————模型配置、预测相关函数—————————————————————————— #
def predict_config(model_file, params_file):
    '''
    函数功能：初始化预测模型predictor
    函数输入：模型结构文件，模型参数文件
    函数输出：预测器predictor
    '''
    # 根据预测部署的实际情况，设置Config
    config = Config()
    # 读取模型文件
    config.set_prog_file(model_file)
    config.set_params_file(params_file)
    # Config默认是使用CPU预测，若要使用GPU预测，需要手动开启，设置运行的GPU卡号和分配的初始显存。
    config.enable_use_gpu(400, 0)
    # 可以设置开启IR优化、开启内存优化。
    config.switch_ir_optim()
    config.enable_memory_optim()
    # config.enable_tensorrt_engine(workspace_size=1 << 30, precision_mode=PrecisionType.Float32,max_batch_size=1, min_subgraph_size=5, use_static=False, use_calib_mode=False)
    predictor = create_predictor(config)
    return predictor

def predict(predictor, img):
    
    '''
    函数功能：初始化预测模型predictor
    函数输入：模型结构文件，模型参数文件
    函数输出：预测器predictor
    '''
    input_names = predictor.get_input_names()
    for i, name in enumerate(input_names):
        input_tensor = predictor.get_input_handle(name)
        input_tensor.reshape(img[i].shape)
        input_tensor.copy_from_cpu(img[i].copy())
    # 执行Predictor
    predictor.run()
    # 获取输出
    results = []
    # 获取输出
    output_names = predictor.get_output_names()
    for i, name in enumerate(output_names):
        output_tensor = predictor.get_output_handle(name)
        output_data = output_tensor.copy_to_cpu()
        results.append(output_data)
    return results

# ——————————————————————后处理函数—————————————————————————— #
def draw_bbox_image(frame, result, label_list, threshold=0.5):
    
    for res in result:
        cat_id, score, bbox = res[0], res[1], res[2:]
        if score < threshold:
    	    continue
        xmin, ymin, xmax, ymax = bbox
        cv2.rectangle(frame, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (255,0,255), 2)
        label_id = label_list[int(cat_id)]
        print('label is {}, bbox is {}'.format(label_id, bbox))
        try:
            # #cv2.putText(图像, 文字, (x, y), 字体, 大小, (b, g, r), 宽度)
            cv2.putText(frame, label_id, (int(xmin), int(ymin-2)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 2)
            cv2.putText(frame, str(round(score,2)), (int(xmin-35), int(ymin-2)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
        except KeyError:
            pass

def callback(data):
    global bridge, predictor, im_size, im_shape, scale_factor, label_list
    cv_img = bridge.imgmsg_to_cv2(data, "bgr8")
    img_data, cv_img = preprocess(cv_img, im_size)        
    # 预测
    result = predict(predictor, [im_shape, img_data, scale_factor])
    draw_bbox_image(cv_img, result[0], label_list, threshold=0.1)
    cv2.imshow("cv_img", cv_img)

    cv2.waitKey(1)

if __name__ == '__main__':
    import sys 
    print(sys.version) # 查看python版本
    
    # 初始化节点
    rospy.init_node('ppinfer_node', anonymous=True)
    bridge = CvBridge()

    # 模型文件路径(最好写绝对路径)
    model_dir = '/home/nano/workspace/paddle_ros_ws/src/py3_infer/scripts/yolov3_r50vd_dcn_270e_coco/'
    # 从infer_cfg.yml中读出label
    infer_cfg = open(model_dir + 'infer_cfg.yml')
    data = infer_cfg.read()
    yaml_reader = yaml.load(data)
    label_list = yaml_reader['label_list']
    print(label_list)

    # 配置模型参数
    model_file = model_dir + "model.pdmodel"
    params_file = model_dir + "model.pdiparams"
    
    # 图像尺寸相关参数初始化
    try:
        img = bridge.imgmsg_to_cv2(data, "bgr8")
    except AttributeError:
        img = np.zeros((224,224,3), np.uint8)
    im_size = 224
    scale_factor = np.array([im_size * 1. / img.shape[0], im_size * 1. / img.shape[1]]).reshape((1, 2)).astype(np.float32)
    im_shape = np.array([im_size, im_size]).reshape((1, 2)).astype(np.float32)

    # 初始化预测模型
    predictor = predict_config(model_file, params_file)

    rospy.Subscriber('/image_view/image_raw', Image, callback)

    rospy.spin()