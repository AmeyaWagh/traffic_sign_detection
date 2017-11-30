# import sys
# sys.path.insert(0,'./darkflow')
import os

from darkflow.net.build import TFNet
import cv2
import numpy as np
import csv

print("Initialisation Done")

# script_path = os.path.dirname(os.path.realpath(__file__))
# script_path = os.path.join(script_path,'darkflow')
options = {"model": "cfg/yolo.cfg", "load": "bin/yolo.weights", "threshold": 0.1,"gpu":0.5}

tfnet = TFNet(options)

# imgcv = cv2.imread("sample_img/sample_dog.jpg")
# print(type(imgcv))
# imgcv = np.array(imgcv)
# result = tfnet.return_predict(imgcv)
# print(result)
with open('../dataset/allAnnotations.csv') as csvfile:
    anotations_list = csvfile.readlines()
    # print(anotations_list)
    for row in anotations_list:
        print(row.split(';'))
        break
print(anotations_list.pop(0))
print('doing')    
# datadir = '../dataset/vid0/frameAnnotations-vid_cmp2.avi_annotations'
# dataset = os.listdir(datadir)
# dataset = [i for i in dataset if '.png' in i]
predictThresh = 0.8
# for img in dataset:
for img in anotations_list:
    img = img.split(';')
    # print(img)
    imgcv = cv2.imread(os.path.join('../dataset',img[0]))
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
        cv2.rectangle(imgcv,(x1,y1),(x2,y2),(0,255,0),4)
        cv2.putText(imgcv,label,(x1+10,y1+2),0,0.5,(0,0,0))
    # cv2.destroyAllWindows()
    cv2.imshow('detected objects',imgcv)
    if cv2.waitKey(50) == ord('q'):
        break   
print('done')    