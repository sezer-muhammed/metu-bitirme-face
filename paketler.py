from turtle import forward
from soupsieve import match
import torch
import sys
from pathlib import Path
import os

from sqlite3 import Time
import torch
import glob 
from random import shuffle
import numpy as np
import argparse
import torch
import torch.backends.cudnn as cudnn
import cv2
from pathlib import Path
import os
from time import time

sys.path.insert(0, './yolov5')
from yolov5.models.common import DetectMultiBackend

from deep_sort.utils.parser import get_config
from deep_sort.deep_sort import DeepSort

import face_recognition

class ids_info():
  def __init__(self, model_path, tracker_name, face_database_path):
    self.model = torch.hub.load("yolov5", 'custom', path=model_path, source='local')
    self.model.conf = 0.45
    cfg = get_config()
    cfg.merge_from_file("deep_sort.yaml")
    self.deepsort = DeepSort(tracker_name,
                        "cuda", #! Performans için değiştirilmesi gerekebilir
                        max_dist=cfg.DEEPSORT.MAX_DIST,
                        max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                        max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT, nn_budget=cfg.DEEPSORT.NN_BUDGET,
                        )
    self.faces = face_database_path

    self.residents = []
    self.residents_name = []
    for resident in glob.glob("faces/residents/*.jpg"):
      image = cv2.imread(resident)
      image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
      name = resident.split("\\")[-1].split("_")[0] #! UBUNTUDA BUNU DÜZELT
      face_encoding = face_recognition.face_encodings(image)[0]
      self.residents.append(face_encoding)
      self.residents_name.append(name)

  def Detect(self, frame):
    self.frame = frame
    self.result = self.model(frame)
    self.yolo_detections = self.result.pred[0].cpu()

  def Tracker(self):
    clss = self.result.xywh[0][:, 5].cpu()
    xywh = self.result.xywh[0][:, 0:4].cpu()[clss == 1]
    confs = self.result.xywh[0][:, 4].cpu()[clss == 1]
    clss = self.result.xywh[0][:, 5].cpu()[clss == 1]
    self.tracker_detections = np.array(self.deepsort.update(xywh, confs, clss, self.frame))
    if len(self.tracker_detections.shape) == 1:
      self.tracker_detections = np.expand_dims(self.tracker_detections, axis = 0)
    if self.tracker_detections.size == 0:
      self.tracker_detections = np.empty((0, 6))
    return self.tracker_detections

  def Regularize(self):
    self.face_locations = self.tracker_detections[:, [1, 2, 3, 0]]
    #TODO Match body and head, then find ymax for each head

  def Face_Detect(self):
    self.face_encodings = face_recognition.face_encodings(self.frame, self.face_locations)
    self.face_names = []    
    for face_encode in self.face_encodings:
      matches = face_recognition.compare_faces(self.residents, face_encode)
      face_distances = face_recognition.face_distance(self.residents, face_encode)
      best_match_index = np.argmin(face_distances)
      name = "unknown"
      if matches[best_match_index]:
        name = self.residents_name[best_match_index]
      self.face_names.append(name)
    return self.face_names

  def Returner(self):
    