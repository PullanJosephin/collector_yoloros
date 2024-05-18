#!/bin/python
# -*- coding: utf-8 -*-
#使用 D435i +yolov5 进行目标检测 
import random
import rospy
import geometry_msgs
from sensor_msgs.msg import PointCloud
from geometry_msgs.msg import Point32
from std_msgs.msg import Header
from std_msgs.msg import Bool
import pyrealsense2 as rs
import numpy as np
import json
import torch
import sys,os
import cv2
from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages
from utils.general import (check_img_size,non_max_suppression, scale_segments)
from utils.torch_utils import select_device,time_sync
from utils.plots import Annotator, colors
from utils.augmentations import  letterbox
from pathlib import Path
pipeline = rs.pipeline()  #定义流程pipeline
config = rs.config()   #定义配置config
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 60)  #配置depth流
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 60)   #配置color流
profile = pipeline.start(config)  #流程开始
align_to = rs.stream.color  #与color流对齐
align = rs.align(align_to)
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

weights=ROOT/'runs/train1/exp4/weights/best_openvino_model'  # model.pt path(s)
#weights='runs/train1/exp4/weights/best.pt'
data=ROOT/'yolov5/eggs/data.yaml'  # dataset.yaml path
imgsz=[640,640]  # inference size (height, width)
conf_thres=0.25  # confidence threshold
iou_thres=0.45  # NMS IOU threshold
max_det=1000  # maximum detections per image
device=''  # cuda device, i.e. 0 or 0,1,2,3 or cpu#使用openino加速时不用cpu
classes=None  # filter by class: --class 0, or --class 0 2 3  detect all:None 过滤类 具体看coco128.yaml
agnostic_nms=False  # class-agnostic NMS
augment=False  # augmented inference
line_thickness=2  # bounding box thickness (pixels)
hide_labels=False  # hide labels
hide_conf=False  # hide confidences
half=False  # use FP16 half-precision inference
dnn=False  # use OpenCV DNN for ONNX inference


rospy.init_node('yolo5_node', anonymous=True)#节点初始化

target_pub = rospy.Publisher('target_positions',PointCloud, queue_size=10)#发布话题

def get_aligned_images():
    frames = pipeline.wait_for_frames()  #等待获取图像帧
    aligned_frames = align.process(frames)  #获取对齐帧
    aligned_depth_frame = aligned_frames.get_depth_frame()  #获取对齐帧中的depth帧
    color_frame = aligned_frames.get_color_frame()   #获取对齐帧中的color帧

    ############### 相机参数的获取 #######################
    intr = color_frame.profile.as_video_stream_profile().intrinsics   #获取相机内参
    depth_intrin = aligned_depth_frame.profile.as_video_stream_profile().intrinsics  #获取深度参数（像素坐标系转相机坐标系会用到）
    camera_parameters = {'fx': intr.fx, 'fy': intr.fy,
                         'ppx': intr.ppx, 'ppy': intr.ppy,
                         'height': intr.height, 'width': intr.width,
                         'depth_scale': profile.get_device().first_depth_sensor().get_depth_scale()
                         }
    # 保存内参到本地
    with open('./intr7insics.json', 'w') as fp:
        json.dump(camera_parameters, fp)
    #######################################################
    
    depth_image = np.asanyarray(aligned_depth_frame.get_data())  #深度图（默认16位）
    depth_image_8bit = cv2.convertScaleAbs(depth_image, alpha=0.03)  #深度图（8位）
    depth_image_3d = np.dstack((depth_image_8bit,depth_image_8bit,depth_image_8bit))  #3通道深度图
    color_image = np.asanyarray(color_frame.get_data())  # RGB图
    
    #返回相机内参、深度参数、彩色图、深度图、齐帧中的depth帧
    return intr, depth_intrin, color_image, depth_image, aligned_depth_frame

def pixlim(pixx,topnum,lowernum):#区间限制
    if (pixx) > topnum-1 :
        pixx = topnum-1
    if pixx <lowernum :
        pixx= lowernum
    return pixx

def get_3d_camera_coordinate(depth_pixel, depth_data, depth_intrin,randnum):
    x = depth_pixel[0]
    y = depth_pixel[1]
    distance_list = []
    mid_pos = [x, y] #确定索引深度的中心像素位置
    min_val = min(10, 10) #确定深度搜索范围
    for i in range(randnum):
        bias = random.randint(-min_val//4, min_val//4)
        #限制
        dist = depth_data[pixlim(int(mid_pos[1] + bias),480,0),pixlim(int(mid_pos[0] + bias),640,0)]
        if dist:
            distance_list.append(dist)
    distance_list = np.array(distance_list)
    distance_list = np.sort(distance_list)[randnum//2-randnum//4:randnum//2+randnum//4] #冒泡排序+中值滤波
    dis = np.mean(distance_list)/1000
    camera_coordinate = rs.rs2_deproject_pixel_to_point(depth_intrin, depth_pixel, dis)

    return dis, camera_coordinate

if __name__ == "__main__":


    # Load model
    device = select_device(device)

    
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Dataloader

    while not rospy.is_shutdown():
        t1 = time_sync()#记录开始识别时间
        intr, depth_intrin, rgb, depth, aligned_depth_frame = get_aligned_images() #获取对齐的图像与相机内参
        im0 = rgb
        # Padded resize
        im = letterbox(im0, imgsz, stride, auto=pt)[0]
        # Convert
        im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        im = np.ascontiguousarray(im)

        bs = 1  # batch_size

        # Run inference
        model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup
        key = cv2.waitKey(1)

        im = torch.from_numpy(im).to(model.device)
        im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        # Inference
            pred = model(im, augment=augment, visualize=False)           
        #pred = model(im, augment=augment, visualize = False)

        # NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

        # Process predictions
        t2 = time_sync()#record the time when stopped inference
        target_positions=[]#清空检测到的目标
        egg_points=PointCloud()
        egg_points.header.stamp = rospy.Time.now()
        egg_points.header.frame_id = 'camera_link'

        for i, det in enumerate(pred):  # per image
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))
            if len(det):
                
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_segments(im.shape[2:], det[:, :4], im0.shape).round()

                # Write results
                for *xyxy, conf, cls in reversed(det):
                        c = int(cls)  # integer class
                        label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                        annotator.box_label(xyxy, label, color=colors(c, True))
                        cx = ((xyxy[2]-xyxy[0])/2)+xyxy[0]
                        cy = ((xyxy[3]-xyxy[1])/2)+xyxy[1]
                        cx=int(cx.cpu())
                        cy=int(cy.cpu()*0.88)#fix of the openvino module
                        im0 = cv2.line(im0,(cx+10,cy),(cx-10,cy),(0, 255, 0),2)#center cross
                        im0 = cv2.line(im0,(cx,cy+10),(cx,cy-10),(0, 255, 0),2)
                        depth_pixel = [cx, cy] 
                        dis, camera_coordinate = get_3d_camera_coordinate(depth_pixel, depth, depth_intrin,24)
                        if  not np.isnan(dis) and dis<0.30 :#too far or no depth info will not be a valuable target
                            target_positions.append(camera_coordinate)
                        cv2.putText(im0, '%.02fm' % (dis), (cx, cy-20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (125, 107, 35), 2)
                for pos in target_positions:
                    egg_points.points.append(Point32(x=-pos[0], y=-pos[1], z=pos[2]))          
            target_pub.publish(egg_points)#publish eggposition pointcloud
            #print(egg_points)  
            fps = 1.0 / (t2 - t1)
            fps_text = "FPS: {:.2f}".format(fps)#calculate inference fps and display it
            cv2.putText(im0, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        # Stream results
           
            im0 = annotator.result()
            cv2.imshow("im0", im0)
            cv2.waitKey(1)  # 1 millisecond
    pipeline.stop()
cv2.destroyAllWindows()