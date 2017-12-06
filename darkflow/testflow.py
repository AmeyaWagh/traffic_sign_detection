# import sys
# sys.path.insert(0,'./darkflow')
import os

from darkflow.net.build import TFNet
import cv2
import numpy as np
import csv
import json

print("Initialisation Done")
# cap = cv2.VideoCapture(0)

config = json.load(open('../config.json'))

# options = {"model": "cfg/yolo.cfg", "load": "bin/yolo.weights", "threshold": 0.1,"gpu":0.7}
options = config['yoloConfig']

tfnet = TFNet(options)

# '../dataset/allAnnotations.csv'
with open(os.path.join('../',config["dataset"],'allAnnotations.csv')) as csvfile:
    anotations_list = csvfile.readlines()
    # print(anotations_list)
    for row in anotations_list:
        print(row.split(';'))
        break
print(anotations_list.pop(0))
print('doing')   
print('press q to exit') 
# datadir = '../dataset/vid0/frameAnnotations-vid_cmp2.avi_annotations'
# dataset = os.listdir(datadir)
# dataset = [i for i in dataset if '.png' in i]
predictThresh = 0.4
# for img in dataset:
try:
    for img in anotations_list:
        img = img.split(';')
        # print(img)
        # ret,imgcv = cap.read()
        imgcv = cv2.imread(os.path.join('../',config["dataset"],img[0]))
        result = tfnet.return_predict(imgcv)
        for box in result:
            # print(box)
            x1,y1,x2,y2 = (box['topleft']['x'],box['topleft']['y'],box['bottomright']['x'],box['bottomright']['y'])
            conf = box['confidence']
            # print(conf)
            label = box['label']
            if conf < predictThresh:
                continue
            # print(x1,y1,x2,y2,conf,label)
            cv2.rectangle(imgcv,(x1,y1),(x2,y2),(0,255,0),6)
            # cv2.rectangle(imgcv,(x1,y1),(x1+10,y1+2),(0,255,0),4)
            cv2.putText(imgcv,label,(x1+10,y1+2),cv2.FONT_HERSHEY_COMPLEX,0.5,(255,0,0),2)
        cv2.imshow('detected objects',imgcv)
        if cv2.waitKey(0) == ord('q'):
            break

except KeyboardInterrupt:
    cv2.destroyAllWindows()

print('done')
quit()    