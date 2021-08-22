pip install face-detection
pip install tqdm

import time
import math
import os
import numpy as np
import cv2
import face_detection 
from keras.models import load_model
from keras.applications.resnet50 import preprocess_input
import tqdm
import imutils

conf_threshold = 0.1
nms_threshold = 0.1
angle_factor = 0.4
H_zoom_factor = 1.2


def dist(c1, c2):
    return ((c1[0] - c2[0]) ** 2 + (c1[1] - c2[1]) ** 2) ** 0.5

def T2S(T):
    S = abs(T/((1+T**2)**0.5))
    return S

def T2C(T):
    C = abs(1/((1+T**2)**0.5))
    return C

def isclose(p1,p2):

    c_d = dist(p1[2], p2[2])
    if(p1[1]<p2[1]):
        a_w = p1[0]
        a_h = p1[1]
    else:
        a_w = p2[0]
        a_h = p2[1]

    T = 0
    try:
        T=(p2[2][1]-p1[2][1])/(p2[2][0]-p1[2][0])
    except ZeroDivisionError:
        T = 1.633123935319537e+16
    S = T2S(T)
    C = T2C(T)
    d_hor = C*c_d
    d_ver = S*c_d
    vc_calib_hor = a_w*1.3
    vc_calib_ver = a_h*0.4*angle_factor
    c_calib_hor = a_w *1.7
    c_calib_ver = a_h*0.2*angle_factor
    
    if (0<d_hor<vc_calib_hor and 0<d_ver<vc_calib_ver):
        return 1
    elif 0<d_hor<c_calib_hor and 0<d_ver<c_calib_ver:
        return 2
    else:
        return 0



BASE_PATH = ""
FILE_PATH = ""

detector = face_detection.build_detector("DSFDDetector", confidence_threshold=.5, nms_iou_threshold=.3)


mask_classifier = load_model("")

net = cv2.dnn.readNet(BASE_PATH + "YOLO/"+"yolov3-spp.weights", BASE_PATH + "YOLO/" + "yolov3-spp.cfg")

LABELS  = []
with open(BASE_PATH + "YOLO/" + "coco.names", "r") as f:
    LABELS = [line.strip() for line in f.readlines()]


layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
FR=0

vs = cv2.VideoCapture(BASE_PATH + FILE_PATH )
fps = vs.get(cv2.CAP_PROP_FPS)
n_frames = vs.get(cv2.CAP_PROP_FRAME_COUNT)

writer = None
(W, H) = (None, None)

fl = 0
q = 0

for feed in tqdm(range(int(n_frames))):
  
    (grabbed, frame) = vs.read()
    frame = imutils.resize(frame, width=1280,height=720)
    if not grabbed:
        break

    if W is None or H is None:
        (H, W) = frame.shape[:2]
        FW=W
        FR = np.zeros((H+210,FW,3), np.uint8)

        col = (255,255,255)
        FH = H + 210
    FR[:] = col

    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0 , (608, 608), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    start = time.time()
    outs = net.forward(output_layers)
    end = time.time()

    classIDs  = []
    confidences = []
    boxes = []
    
    # Store Detected Objects with Labels, Bounding_Boxes and their Confidences
    for out in outs:
        for detection in out:
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]
            if LABELS[classID] == "person":
                if confidence > conf_threshold:
                
                    # Get Center, Height and Width of the Box
                    box = detection[0:4] * np.array([W, H, W, H])
                    (centerX, centerY, width, height) = box.astype("int")
                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))

                    boxes.append([x, y,int(width), int(height)])
                    confidences.append(float(confidence))
                    classIDs.append(classID)

    idxs = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold,nms_threshold)
    
    if len(idxs) > 0:
        persons = []
        masked_faces = []
        unmasked_faces = []
        status = []
        idf = idxs.flatten()
        close_pair = []
        s_close_pair = []
        center = []
        co_info = []

        for i in idf:
            box = np.array(boxes[i])
            box = np.where(box<0,0,box)
            (x, y, w, h) = box
            cen = [int(x + w / 2), int(y + h / 2)]
            center.append(cen)
            cv2.circle(frame, tuple(cen),1,(0,0,0),1)
            co_info.append([w, h, cen])

            status.append(0)
            persons.append([x,y,w,h])
                    
            person_rgb = frame[y:y+h,x:x+w,::-1]   
            detections = detector.detect(person_rgb)

            if detections.shape[0] > 0:
                detection = np.array(detections[0])
                detection = np.where(detection<0,0,detection)

                x1 = x + int(detection[0])
                x2 = x + int(detection[2])
                y1 = y + int(detection[1])
                y2 = y + int(detection[3])

                try :
                    face_rgb = frame[y1:y2,x1:x2,::-1]  
                    face_arr = cv2.resize(face_rgb, (224, 224), interpolation=cv2.INTER_NEAREST)
                    face_arr = np.expand_dims(face_arr, axis=0)
                    face_arr = preprocess_input(face_arr)

                    score = mask_classifier.predict(face_arr)

                    if score[0][0]<0.10:
                        masked_faces.append([x1,y1,x2,y2])
                    else:
                        unmasked_faces.append([x1,y1,x2,y2])

                except:
                    continue
        masked_face_count = len(masked_faces)
        unmasked_face_count = len(unmasked_faces)
        for i in range(len(center)):
            for j in range(len(center)):
                g = isclose(co_info[i],co_info[j])
                if g == 1:
                    close_pair.append([center[i], center[j]])
                    status[i] = 1
                    status[j] = 1
                elif g == 2:
                    s_close_pair.append([center[i], center[j]])
                    if status[i] != 1:
                        status[i] = 2
                    if status[j] != 1:
                        status[j] = 2
        total_p = len(center)
        low_risk_p = status.count(2)
        high_risk_p = status.count(1)
        safe_p = status.count(0)
        kk = 0                       
        for i in idf:
            cv2.line(FR,(0,H+1),(FW,H+1),(0,0,0),2)
            cv2.putText(FR, "Social Distancing Analyser and Mask Monitoring wrt. COVID-19", (210, H+60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
            cv2.rectangle(FR, (20, H+80), (510, H+180), (100, 100, 100), 2)
            cv2.putText(FR, "Connecting lines shows closeness among people. ", (30, H+100),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 0), 2)
            cv2.putText(FR, "-- YELLOW: CLOSE", (50, H+90+40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 170, 170), 2)
            cv2.putText(FR, "--    RED: VERY CLOSE", (50, H+40+110),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            cv2.rectangle(FR, (535, H+80), (1250, H+140+40), (100, 100, 100), 2)
            cv2.putText(FR, "Bounding box shows the level of risk to the person and Mask Monitoring", (545, H+100),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 0), 2)
            cv2.putText(FR, "-- LIGHT GREEN: SAFE", (565,  H+90+40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.putText(FR, "-- Green: MASKED", (865, H+90+40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 100, 0), 2)
            cv2.putText(FR, "--    DARK RED: HIGH RISK", (565, H+150),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 150), 2)
            cv2.putText(FR, "--   RED: UNMASKED", (865, H+150),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            cv2.putText(FR, "--      ORANGE: LOW RISK", (565, H+170),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 120, 255), 2)
            
            tot_str = "TOTAL COUNT:" + str(total_p)
            high_str = "HIGH RISK COUNT:" + str(high_risk_p)
            low_str = "LOW RISK COUNT:" + str(low_risk_p)
            safe_str = "SAFE COUNT:" + str(safe_p)
            masked_str="MASKED COUNT:" + str(masked_face_count)
            unmasked_str="UNMASKED COUNT:" + str(unmasked_face_count)
            unknown_str="UNKNOWN COUNT:" + str(total_p-masked_face_count-unmasked_face_count)
            cv2.putText(FR, tot_str, (1, H +25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            cv2.putText(FR, safe_str, (160, H +25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(FR, low_str, (310, H +25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 120, 255), 2)
            cv2.putText(FR, high_str, (500, H +25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 150), 2)
            cv2.putText(FR, masked_str, (700, H +25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 100, 0), 2)
            cv2.putText(FR, unmasked_str, (880, H +25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            cv2.putText(FR, unknown_str, (1080, H +25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            if status[kk] == 1:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 150), 2)

            elif status[kk] == 0:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            else:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 120, 255), 2)

            kk += 1
        for h in close_pair:
            cv2.line(frame, tuple(h[0]), tuple(h[1]), (0, 0, 255), 2)
        for b in s_close_pair:
            cv2.line(frame, tuple(b[0]), tuple(b[1]), (0, 255, 255), 2)            

        for f in range(masked_face_count):
            a,b,c,d = masked_faces[f]
            cv2.rectangle(frame, (a, b), (c,d), (0,100,0), 2)

        for f in range(unmasked_face_count):
            a,b,c,d = unmasked_faces[f]
            cv2.rectangle(frame, (a, b), (c,d), (0,0,255), 2)

        FR[0:H, 0:W] = frame  
        frame = FR

        cv2.waitKey(1)

    if writer is None:
        fourcc = cv2.VideoWriter_fourcc('X','V','I','D')
        writer = cv2.VideoWriter(BASE_PATH, fourcc, 30,(frame.shape[1], frame.shape[0]), True)
    writer.write(frame)


writer.release()
vs.release()
cv2.destroyAllWindows()







