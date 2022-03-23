from matplotlib.pyplot import magma
from paketler import ids_info
import cv2
import json
import glob 

videos = glob.glob("*.mp4")

for video in videos:
  manager = ids_info("../crowdhuman_yolov5m (1).pt", "osnet_x0_25", "faces")

  cam = cv2.VideoCapture(video)

  fps = cam.get(cv2.CAP_PROP_FPS)
  w = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
  h = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))

  saver = cv2.VideoWriter(f"{video.split('.')[0]}Filled.mp4", cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
  counter = 0

  while True:
    counter += 1
    _, frame = cam.read()
    if _ == False:
      break
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    #frame = cv2.resize(frame, (640, 360))

    manager.Detect(frame)
    dets = manager.Tracker()
    manager.Regularize()
    names = manager.Face_Detect()
    Detections = manager.Returner()

    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    for i, det in enumerate(Detections):
      face_name = max(det.faces, key = det.faces.get)
      rect_color = (hash(face_name + "B")%255, hash(face_name + "G")%255, hash(face_name + "R")%255)
      cv2.rectangle(frame, (det.bbox_face[0], det.bbox_face[1]), (det.bbox_face[2], det.bbox_face[3]), rect_color, 5)
      cv2.putText(frame, f"ID: {det.id}, {face_name}", (det.bbox_face[0], det.bbox_face[3]), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 0, 255), 1)
      info_color = (0, 255, 30)
      if det.frame_no > 300 and face_name == "unknown" and counter % int(fps * 2) < int(fps):
        info_color = (25, 10, 255)
      elif face_name != "unknown":
        info_color = rect_color
      info_text = f"ID: {det.id:3.0f}, Life: {round(det.frame_no / fps, 2):7.2f} Seconds"
      cv2.putText(frame, info_text, (10, (i + 1) * 35), cv2.FONT_HERSHEY_COMPLEX, 1, info_color, 2)
      for key in det.faces:
        lenght = 10 + cv2.getTextSize(info_text, cv2.FONT_HERSHEY_COMPLEX, 1, 2)[0][0]
        sorted_dict = dict(sorted(det.faces.items(), key=  lambda item: item[1],  reverse = True))
        sorted_dict = json.dumps(sorted_dict)
        cv2.putText(frame, f"| {sorted_dict}", (lenght, (i + 1) * 35), cv2.FONT_HERSHEY_COMPLEX, 1, info_color, 2)
    cv2.imshow("frame", frame)
    saver.write(frame)
    cv2.waitKey(1)