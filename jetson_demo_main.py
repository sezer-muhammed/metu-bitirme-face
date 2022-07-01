import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--model', type=str, default="fat-480.pt", help='Model File Path') #burada tensorrt de olabilir pt de uzantı olarak
parser.add_argument('--tracker', type=str, default="osnet_x0_25", help='Tracker Name')
parser.add_argument('--faces', type=str, default="faces", help='Face Database Path')

parser.add_argument('--conf', type=float, default=0.45, help='confidence threshold')

parser.add_argument('--verbose', action='store_true', help='Show Methods Time')
args = parser.parse_args()  

from paketler import ids_info
import cv2
import json
import glob 
import numpy as np
from shapely import geometry
import gdown
import shutil
import time

import logging
import sys

import jetson.inference
import jetson.utils

logger = logging.getLogger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s') #logger objesinin formatını belirtiyor, bu formatta save edecek infoları

stdout_handler = logging.StreamHandler(sys.stdout)
stdout_handler.setLevel(logging.DEBUG)
stdout_handler.setFormatter(formatter)

file_handler = logging.FileHandler('logs.log')
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(formatter) #buralar hep logger kısmı kurulumu

logger.addHandler(file_handler)
logger.addHandler(stdout_handler)

logger.info('Home security system started. Downloading informations from drive.')

#Bu kısım mevcut klasörü silip google drivedan yenisini indiriyor
#Dikkat edin saatte 2-3 kere yapma hakkınız var, kullanacağınızda commenti kaldırın

#try:
#  shutil.rmtree("faces")
#  time.sleep(0.5)
#except:
#  pass
#url = "https://drive.google.com/drive/folders/1mme8YyDRcOB37gWqyWho-G5wi30xNcnu"
#gdown.download_folder(url, quiet=False, use_cookies=False)

bounds = np.loadtxt("faces/bounds.txt") #metin belgesini np arrayi olarak açıyor bu yüzden her satırda eşit sayıda sayı olmalı, -1'lerin sebebi o
pet_corners = bounds[-1, :] #son satır pet cornersi belirtiyor, aruco tespiti yapılacak bölge
forbiddens = bounds[:-1, :] #diğer satırlar ise yasaklı alanlar

forbidden_poly = []
for poly in forbiddens:
    poly = poly[poly != -1].reshape((-1, 2))
    forbidden_poly.append(geometry.Polygon(poly))
    logger.info(f"Forbidden Area Added: {str(forbidden_poly[-1].wkt)}") #yasaklı alanları shapely kütüphanesi ile oluşturuyorum ve log'lara kaydediyorum 


flags = {"Pet Detection":False, "Someone Here For a Long Time":False, "Someone in Forbidden Area":False, "Camera Sabotage":False} #flaglar

home_population = {"Left":0, "Entered":0} #flaglar
counter = 0
pet_counter = 0
forbidden_counter = 0
last_id = 0
last_counter = [0,0,0,0]
last_ids = np.array([])

manager = ids_info(args.model, args.tracker, args.faces, args.verbose, args.conf, forbidden_poly, logger) #yapay zeka ve düzenleme Class'ını çağırır

cam = jetson.utils.gstCamera(640, 480, "/dev/video0")



while True:
    counter += 1

    frame_cuda, w, h = cam.CaptureRGBA()
    frame_mapped = jetson.utils.cudaAllocMapped(width = 640, height = 480, format = frame_cuda.format)
    jetson.utils.cudaOverlay(frame_cuda, frame_mapped, 0, 0)
    frame = np.array(jetson.utils.cudaToNumpy(frame_mapped)[:,:,:3], np.uint8)
    jetson.utils.cudaDeviceSynchronize()



    flags["Camera Sabotage"] = False
    if np.sum(frame[360,200:400]) < 1000:
        flags["Camera Sabotage"] = True #Frame'in bir satırının toplam değeri azsa (karanlıksa) sabotaj varsayıyor, bunu daha pratik çözemedim

    arucoDict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)
    arucoParams = cv2.aruco.DetectorParameters_create()
    aruco_frame = frame[int(pet_corners[1]):int(pet_corners[3]), int(pet_corners[0]):int(pet_corners[2])]
    (corners, ids, rejected) = cv2.aruco.detectMarkers(aruco_frame, arucoDict,parameters=arucoParams) #Aruco tespiti yapılıyor

    pet_counter = max(pet_counter - 1, 0) #pet counter bir kere tespit yapıldıktan sonra 10 oluyor ve her loop'ta azalıyor, tekrar tespit olunca gene 10 oluyor böylece 1 tespit 4 saniye boyunca kalıyor

    #bu for döngüsü ise güncel bir tespit varsa çizdiriyor
    if ids is not None:
        for corner, id in zip(corners, ids):
            pet_counter = 10
            last_counter = corner
            last_id = id
            cv2.rectangle(frame, (int(np.min(last_counter[0][:, 0]) + pet_corners[0]), int(np.min(last_counter[0][:, 1]) + pet_corners[1])), (int(np.max(last_counter[0][:, 0]) + pet_corners[0]), int(np.max(last_counter[0][:, 1]) + pet_corners[1])), (25, 0, 250), 2) 

    #pet counter 0 değilse pet tespit edilmiş varsayılıyor ve son aruco tespitinin ID'si bastırılıyor
    flags["Pet Detection"] = False
    if pet_counter > 0:
        flags["Pet Detection"] = True
        cv2.putText(frame, f"AruCo ID: {last_id}", (10, 300), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 1)

    #Frame RGB Haline geliyor

    manager.Detect(frame) #Obje tespiti
    dets = manager.Tracker() #Tracker modeli
    forbidden_points = manager.Regularize() #Düzenleme ve yasak alana giriş bilgi edinimi
    names = manager.Face_Detect() #Yüz tespiti
    Detections, _ = manager.Returner() #Detection objelerinin elde edilmesi 
    residents = manager.return_owners()
    #Bu kısımda ellenmesi gereken bir şey yok, tüm işlemler detections ve forbidden_points üzerinden olmalı 


    #Filtreme amaçlı olarak forbidden counter da pet counter'a benzer bir şekilde çalışıyor
    forbidden_counter = max(forbidden_counter - 1, 0)
    if forbidden_counter == 0:
        flags["Someone in Forbidden Area"] = False
    else:
        flags["Someone in Forbidden Area"] = True
    for point in forbidden_points:
        forbidden_counter = 10
        cv2.circle(frame, (point[0], point[1]), 10, (0, 255, 255), -1)

    current_ids = [] #Bu kısım eve girenleri algılamak için bir önceki ID'leri ve konumlarını kaydediyor, Bir sonrakinde kaybolan ID'lerin konumu eve yakında eve giriş sayılıyor
    for i, det in enumerate(Detections):
        current_ids.append([det.id, det.bbox_face[3]])
        face_name = max(det.faces, key = det.faces.get)
        rect_color = (hash(face_name + "B")%255, hash(face_name + "G")%255, hash(face_name + "R")%255)
        cv2.rectangle(frame, (det.bbox_face[0], det.bbox_face[1]), (det.bbox_face[2], det.bbox_face[3]), rect_color, 3)
        cv2.putText(frame, f"ID: {det.id}, {face_name}", (det.bbox_face[0], det.bbox_face[3]), cv2.FONT_HERSHEY_COMPLEX, 0.4, (0, 0, 255), 1) #Yüzleri çizdiriyor
        info_color = (0, 255, 30)

        flags["Someone Here For a Long Time"] = False
        if det.frame_no > 10 and face_name == "unknown" and counter % int(2 * 2) < int(2):
            flags["Someone Here For a Long Time"] = True
            info_color = (25, 10, 255) #uzun süre durma durumunda flag açılıyor

        elif face_name != "unknown":
            info_color = rect_color

        info_text = f"ID: {det.id:3.0f}, Life: {round(det.frame_no / 2, 2):7.2f} Seconds"
        cv2.putText(frame, info_text, (10, (i + 1) * 20), cv2.FONT_HERSHEY_COMPLEX, 0.5, info_color, 1) #Tespit edilen kişiler hakkında info printleniyor

        lenght = 10 + cv2.getTextSize(info_text, cv2.FONT_HERSHEY_COMPLEX, 0.5, 1)[0][0]
        sorted_dict = dict(sorted(det.faces.items(), key=  lambda item: item[1],  reverse = True))
        sorted_dict = json.dumps(sorted_dict)
        cv2.putText(frame, f"| {sorted_dict}", (lenght, (i + 1) * 20), cv2.FONT_HERSHEY_COMPLEX, 0.5, info_color, 1) #isimler printleniyor

        if det.frame_no == 1 and det.bbox_face[3] > 380:
            home_population["Left"] += 1
            sorted_lefter = dict(sorted(home_population.items(), key = lambda item: item[1],  reverse = True))
            sorted_lefter = json.dumps(sorted_lefter)
            logger.info(f"Someone has left the home, ID: {det.id} | {sorted_lefter}") #Burada ID yeni oluşmuşsa ve koordinatı eve yakında evden birisi çıkmış sayılıyor ve loglara ekleniyor

    current_ids = np.array(current_ids) #burada eve giren var mı diye bakılıyor, bir önceki frame ile mevcut frame arasında kaybolan ID'lerin konumuna bakılıyor
    #2 tane if olmasının sebebi bir öncekinde bir insan varsa ve sonrakinde ID yoksa boş küme geldiği için mevcut durumunda boş olmasına göre ayrı if var yoksa kod hata veriyor, önemli bişi değil
    if last_ids.size > 0:
        for id in last_ids:
            if current_ids.size == 0:
                if id[1] > 380:
                    home_population["Entered"] += 1
                    sorted_lefter = dict(sorted(home_population.items(), key = lambda item: item[1],  reverse = True))
                    sorted_lefter = json.dumps(sorted_lefter)
                    logger.info(f"Someone has entered the home, ID: {id[0]} | {sorted_lefter}")
            elif (id[0] in current_ids[:, 0]) == False:
                if id[1] > 380:
                    home_population["Entered"] += 1
                    sorted_lefter = dict(sorted(home_population.items(), key = lambda item: item[1],  reverse = True))
                    sorted_lefter = json.dumps(sorted_lefter)
                    logger.info(f"Someone has entered the home, ID: {id[0]} | {sorted_lefter}")

    last_ids = current_ids
        
    for res in residents:
      cv2.rectangle(frame, (int(res[0]), int(res[1])), (int(res[2]), int(res[3])), (255, 255, 255), 3)
      cv2.putText(frame, f"||||THE RESIDENT||||", (int(res[0]), int(res[3])), cv2.FONT_HERSHEY_COMPLEX, 0.4, (0, 0, 255), 1) #Yüzleri çizdiriyor
      break

    sorted_flags = dict(sorted(flags.items(), key = lambda item: item[1],  reverse = True))
    sorted_flags = json.dumps(sorted_flags)
    cv2.putText(frame, f"{sorted_flags}", (10, 400), cv2.FONT_HERSHEY_COMPLEX, 0.4, (205, 21, 125), 1) #TODO DüZELT
    logger.info(sorted_flags)


    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    cv2.imshow("frame", frame)
    cv2.waitKey(1)

    ### BURADAN SONRA SUNUCUYA GÖNDERME KISMI EKLENECEK
    ### frame, home_population ve flags değişkenleri hariç aktarılacak bir şey yok, sadece bunlar üzerinden işlem yapın
    ### home population sayı, flags ise true false veriyor
    ### frame ise BGR, 380 x 640
