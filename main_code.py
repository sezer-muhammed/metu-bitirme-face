from matplotlib.pyplot import magma
from paketler import ids_info
import cv2

manager = ids_info("crowdhuman_yolov5m (1).pt", "osnet_x0_25", "faces")

cam = cv2.VideoCapture("example_video (1).mp4")

fps = cam.get(cv2.CAP_PROP_FPS)
w = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
#saver = cv2.VideoWriter("Output.mp4", cv2.VideoWriter_fourcc(*'mp4v'), fps, (640, 360))

while True:
  _, frame = cam.read()
  if _ == False:
    break
  frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
  frame = cv2.resize(frame, (640, 360))

  manager.Detect(frame)
  dets = manager.Tracker()
  manager.Regularize()
  names = manager.Face_Detect()
  dets = manager.Returner()

  print(len(dets))

  frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
  for det, name in zip(dets, names):
    cv2.rectangle(frame, (det[0], det[1]), (det[2], det[3]), (0, 255, 100), 3)
    cv2.putText(frame, f"ID: {det[4]}, {name}", (det[0], det[3]), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 1)

  #cv2.imshow("frame", frame)
  #saver.write(frame)
  cv2.waitKey(1)