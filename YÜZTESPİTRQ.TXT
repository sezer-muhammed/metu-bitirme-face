Yüz tespiti için oluşturulacak class

Detection: ID, YÜZ ID{}, BBOX YÜZ, FRAMENO, BBOX BEDEN, 
Detections: Detection[]

Face Database (jpg): Residents Folder, Intruders Folder (Dynamic) (optional)
jpg name: "sevval_0.jpg"

__init__:
* Model path, conf, img-size
* Tracker path (eğer varsa)
* Face Database Path

Detect:
* Input: Frame
* Output: Detected BBOX in YOLO xyxy format (Nx6 Tensor)

Tracker:
* Input: Frame, Detected BBOX in YOLO xyxy format (Nx6 Tensor)
* Output: Detected BBOX in Tracker xyxy and xywh format (Nx6 Tensor)

Regularize:
* Input: Detected BBOX in Tracker xyxy format (Nx6 Tensor)
* Output: Required Format for Each Element
	- X_min, X_max, Y_min, Y_max, ID Array for Face_detection Input.
	- Body BBOX y_min for distance calculation
	- Face - Body Match

Face Detect:
* Input: Frame, X_min, X_max, Y_min, Y_max, ID Array for Face_detection Input, Face Database Path
* Output: Matched Faces' Array

Returner:
* Input: Matched Faces' Array, Detected BBOX in Tracker xyxy format (Nx6 Tensor), Body BBOX y_min for distance calculation, Detections
* Output: Detections