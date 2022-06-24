import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--model', type=str, default="res_fat.pt", help='Model File Path') #burada tensorrt de olabilir pt de uzantı olarak
parser.add_argument('--conf', type=float, default=0.45, help='confidence threshold')
args = parser.parse_args()  

import cv2
import json
import glob 
import numpy as np
import shutil
import time
import torch
import logging
import sys

#import jetson.inference
#import jetson.utils

#cam = jetson.utils.gstCamera(640, 480, "/dev/video0")
cam = cv2.VideoCapture(0)
model = torch.hub.load('yolov5', 'custom', path='crowdhuman_yolov5m.pt', source='local')

counter = 0
while True:
    counter += 1

    #frame_cuda, w, h = cam.CaptureRGBA()
    #frame_mapped = jetson.utils.cudaAllocMapped(width = 640, height = 480, format = frame_cuda.format)
    #jetson.utils.cudaOverlay(frame_cuda, frame_mapped, 0, 0)
    #frame = np.array(jetson.utils.cudaToNumpy(frame_mapped)[:,:,:3], np.uint8)
    #jetson.utils.cudaDeviceSynchronize()

    ret, frame = cam.read()

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    output = model(frame, size = 480)

    rendered = output.render()[0]
    cv2.imshow("rendered", cv2.cvtColor(rendered, cv2.COLOR_BGR2RGB))
    cv2.waitKey(1)