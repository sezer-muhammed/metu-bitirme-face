import torch
import sys
from pathlib import Path
import os

from time import time
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
from yolov5.models.common import DetectMultiBackend, AutoShape

from deep_sort.utils.parser import get_config
from deep_sort.deep_sort import DeepSort

import face_recognition

from shapely import geometry

class face_detection:
  def __init__(self, id, face, bbox_face, bbox_body):
      self.id = id
      self.frame_no = 0
      self.faces = {}
      self.last_name = "unknown"
      self.lock = False
      self.Update(face, bbox_face, bbox_body)

  def Update(self, face, bbox_face, bbox_body):
    self.frame_no += 1
    self.bbox_face = bbox_face
    self.bbox_body = bbox_body
    if face == "empty":
      face = self.last_name
    self.last_name = face

    for name in self.faces:
      if name == "unknown":
        if self.lock:
          self.faces[name] = -100
        else: 
          self.faces[name] = min(max(-100, self.faces[name] - 6), 30)
      else:
        self.faces[name] = min(max(0, self.faces[name] - 4), 30)
        if self.faces[name] > 20:
          self.lock = True


    if face in self.faces.keys():
      self.faces[face] += 8
    else:
      self.faces[face] = 8

class ids_info():
  def __init__(self, model_path, tracker_name, face_database_path, print_time, conf, polygons, logger):
    self.polygons = polygons

    self.print_time = print_time
    self.model = torch.hub.load("yolov5", 'custom', path=model_path, source='local', force_reload=True)
    self.model.conf = conf
    self.model.agnostic = True
    self.logger = logger


    cfg = get_config()
    cfg.merge_from_file("deep_sort.yaml")
    self.deepsort = DeepSort(tracker_name,
                        "cuda", #! Performans için değiştirilmesi gerekebilir
                        max_dist=cfg.DEEPSORT.MAX_DIST,
                        max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                        max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT, nn_budget=cfg.DEEPSORT.NN_BUDGET,
                        )
    self.faces = face_database_path

    self.Detections = []

    self.residents = []
    self.residents_name = []

    self.flags = [0, 0, 0, 0, 0] #forbidden enterance, too long waited people, .., .., ..

    start = time()
    for resident in glob.glob("faces/*/*.jpg"):
      image = cv2.imread(resident)
      image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
      name = resident.split("/")[-1].split("_")[0] #! UBUNTUDA BUNU DÜZELT
      logger.info(f"Trying to upload image {resident}. If there is a problem please delete this file.")
      face_encoding = face_recognition.face_encodings(image)[0]
      self.residents.append(face_encoding)
      self.residents_name.append(name)
    if self.print_time:
      self.logger.info(f"Uploading Resident Images Done. {round(time() - start, 4)}")

  def return_owners(self):
    clss = self.result.xywh[0][:, 5].cpu()
    return self.result.xyxy[0].cpu()[clss > 1]

  def Detect(self, frame):
    self.loop_start = time()
    start = time()
    self.frame = frame
    self.result = self.model(frame, size = 480)
    self.yolo_detections = self.result.pred[0].cpu()
    if self.print_time:
      self.logger.info(f"Detectin Objects Done. {round(time() - start, 4)}")

  def Tracker(self):
    start = time()
    clss = self.result.xywh[0][:, 5].cpu()
    xywh = self.result.xywh[0][:, 0:4].cpu()[clss == 1]
    confs = self.result.xywh[0][:, 4].cpu()[clss == 1]
    clss = self.result.xywh[0][:, 5].cpu()[clss == 1]
    self.tracker_detections = np.array(self.deepsort.update(xywh, confs, clss, self.frame))
    if len(self.tracker_detections.shape) == 1:
      self.tracker_detections = np.expand_dims(self.tracker_detections, axis = 0)
    if self.tracker_detections.size == 0:
      self.tracker_detections = np.empty((0, 6))
    if self.print_time:
      self.logger.info(f"DeepSort Matching Done. {round(time() - start, 4)}")
    return self.tracker_detections

  def Regularize(self):
    start = time()
    self.face_locations = self.tracker_detections[:, [1, 2, 3, 0]]

    clss = self.result.xywh[0][:, 5].cpu()
    bodies = self.result.xywh[0][:, 0:4].cpu()[clss == 0].numpy()
    self.flags[0] = 0
    flag_counter = 0
    forbidden_points = []
    for body in bodies:
      x = body[0]
      y = body[3] / 2 + body[1]
      point = geometry.Point(x, y)
      flag_counter += 1
      for area in self.polygons:
        if area.contains(point):
          self.flags[0] = flag_counter
          forbidden_points.append([int(x), int(y)])
    if self.print_time:
      self.logger.info(f"Regularization of Data Done. {round(time() - start, 4)}")
    return forbidden_points


  def Face_Detect(self):
    start = time()
    if self.face_locations.size == 0:
      self.face_names = []
      return self.face_names
    self.face_names = ["empty"] * self.face_locations.shape[0]
    random = np.random.randint(0, self.face_locations.shape[0])
    self.face_encodings = face_recognition.face_encodings(self.frame, [self.face_locations[random]])
    
    for face_encode in self.face_encodings:
      matches = face_recognition.compare_faces(self.residents, face_encode, tolerance = 0.65)
      face_distances = face_recognition.face_distance(self.residents, face_encode)
      best_match_index = np.argmin(face_distances)
      name = "unknown"
      if matches[best_match_index]:
        name = self.residents_name[best_match_index]
      self.face_names[random] = name
    if self.print_time:
      self.logger.info(f"Face ID Extracted and Matched Done. {round(time() - start, 4)}, {self.face_names}")
    return self.face_names

  def Returner(self):
    start = time()
    new_detections = []
    matched_indexes = []

    for det in self.Detections:
      result = np.where(self.tracker_detections[:, 4] == det.id)[0]
      if result.size != 0:
        result = result[0]
        matched_indexes.append(result)
        det.Update(self.face_names[result], self.tracker_detections[result, 0:4], [])
        new_detections.append(det)

    self.Detections = new_detections

    counter = -1
    for face_name, tracked_det in zip(self.face_names, self.tracker_detections):
      counter += 1
      if counter in matched_indexes:
        continue
      det = face_detection(tracked_det[4], face_name, tracked_det[0:4], [])
      self.Detections.append(det)

    if self.print_time:
      self.logger.info(f"Detections Done. {round(time() - start, 4)}")
      self.logger.info(f"Total Loop Time: {time() - self.loop_start:5.3f}, FPS: {round(1 / (time() - self.loop_start),2)}\n")
      #print("----------------------------------------------------------------------------------")
    return self.Detections, self.flags
