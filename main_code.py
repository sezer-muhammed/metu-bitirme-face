import argparse
from struct import unpack

parser = argparse.ArgumentParser()

parser.add_argument('--model', type=str, default="bitirme_face_detection_yolov5n.pt", help='Model File Path')
parser.add_argument('--tracker', type=str, default="osnet_x0_25", help='Tracker Name')
parser.add_argument('--faces', type=str, default="faces", help='Face Database Path')

parser.add_argument('--source', type=str, default="", help='Video Path or 0 for webcam')
parser.add_argument('--conf', type=float, default=0.45, help='confidence threshold')
#parser.add_argument('--int-input', default=3, type=int, help='bounding box thickness (pixels)')
#parser.add_argument('--binary', default=False, action='store_true', help='hide confidences')
parser.add_argument('--verbose', action='store_true', help='Show Methods Time')
args = parser.parse_args()

from paketler import ids_info
import cv2
import json
import glob 
import numpy as np
from shapely import geometry

bounds = np.loadtxt("faces/bounds.txt")
pet_corners = bounds[-1, :]
forbiddens = bounds[:-1, :]

forbidden_poly = []
for poly in forbiddens:
  poly = poly[poly != -1].reshape((-1, 2))
  forbidden_poly.append(geometry.Polygon(poly))
  print("Forbidden Area Added:", forbidden_poly[-1].wkt)

videos = glob.glob("../face_videos/WIN_20220429_10_34_15_Pro.mp4")

for video in videos:
  manager = ids_info(args.model, args.tracker, args.faces, args.verbose, args.conf, forbidden_poly)

  cam = cv2.VideoCapture(video)

  fps = cam.get(cv2.CAP_PROP_FPS)
  w = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
  h = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))

  saver = cv2.VideoWriter(f"runs/{video.split('.')[0]}_filled.mp4", cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
  counter = 0
  print(f"====={video}========")

  pet_counter = 0
  last_id = 0
  last_counter = [0,0,0,0]

  while True:
    counter += 1

    _, frame = cam.read()
    if _ == False:
      break


    arucoDict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)
    arucoParams = cv2.aruco.DetectorParameters_create()
    aruco_frame = frame[int(pet_corners[1]):int(pet_corners[3]), int(pet_corners[0]):int(pet_corners[2])]
    (corners, ids, rejected) = cv2.aruco.detectMarkers(aruco_frame, arucoDict,parameters=arucoParams)

    if ids is not None:
      for corner, id in zip(corners, ids):
        pet_counter = 20
        pet_counter = max(pet_counter - 1, 0)
        last_counter = corner
        last_id = id
        cv2.rectangle(frame, (int(np.min(last_counter[0][:, 0]) + pet_corners[0]), int(np.min(last_counter[0][:, 1]) + pet_corners[1])), (int(np.max(last_counter[0][:, 0]) + pet_corners[0]), int(np.max(last_counter[0][:, 1]) + pet_corners[1])), (25, 0, 250), 2)

    if pet_counter > 0:
      cv2.putText(frame, f"AruCo ID: {last_id}", (10, 700), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 1)


    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


    manager.Detect(frame)
    dets = manager.Tracker()
    forbidden_points = manager.Regularize()
    names = manager.Face_Detect()
    Detections, flags = manager.Returner()

    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    
    for point in forbidden_points:
      cv2.circle(frame, (point[0], point[1]), 10, (0, 255, 255), -1)

    for i, det in enumerate(Detections):
      face_name = max(det.faces, key = det.faces.get)
      rect_color = (hash(face_name + "B")%255, hash(face_name + "G")%255, hash(face_name + "R")%255)
      cv2.rectangle(frame, (det.bbox_face[0], det.bbox_face[1]), (det.bbox_face[2], det.bbox_face[3]), rect_color, 3)
      cv2.putText(frame, f"ID: {det.id}, {face_name}", (det.bbox_face[0], det.bbox_face[3]), cv2.FONT_HERSHEY_COMPLEX, 0.4, (0, 0, 255), 1)
      info_color = (0, 255, 30)

      if det.frame_no > 300 and face_name == "unknown" and counter % int(fps * 2) < int(fps):
        info_color = (25, 10, 255)

      elif face_name != "unknown":
        info_color = rect_color

      info_text = f"ID: {det.id:3.0f}, Life: {round(det.frame_no / fps, 2):7.2f} Seconds"
      cv2.putText(frame, info_text, (10, (i + 1) * 20), cv2.FONT_HERSHEY_COMPLEX, 0.5, info_color, 1)

      for key in det.faces:
        lenght = 10 + cv2.getTextSize(info_text, cv2.FONT_HERSHEY_COMPLEX, 0.5, 1)[0][0]
        sorted_dict = dict(sorted(det.faces.items(), key=  lambda item: item[1],  reverse = True))
        sorted_dict = json.dumps(sorted_dict)
        cv2.putText(frame, f"| {sorted_dict}", (lenght, (i + 1) * 20), cv2.FONT_HERSHEY_COMPLEX, 0.5, info_color, 1)

    cv2.imshow("frame", frame)
    saver.write(frame)
    cv2.waitKey(1)
