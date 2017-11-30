# import sys
# sys.path.insert(0,'./darkflow')
import os

from darkflow.net.build import TFNet
import cv2
import numpy as np

print("Initialisation Done")

# script_path = os.path.dirname(os.path.realpath(__file__))
# script_path = os.path.join(script_path,'darkflow')
options = {"model": "cfg/yolo.cfg", "load": "bin/yolo.weights", "threshold": 0.1}

tfnet = TFNet(options)

imgcv = cv2.imread("sample_img/sample_dog.jpg")
print(type(imgcv))
# imgcv = np.array(imgcv)
result = tfnet.return_predict(imgcv)
print(result)


print('doing')    
# dataset = os.listdir('./sample_img')
datadir = '../dataset/vid0/frameAnnotations-vid_cmp2.avi_annotations'

dataset = os.listdir(datadir)
print(dataset)
# dataset = [i for i in dataset if '.jpg' in i]
dataset = [i for i in dataset if '.png' in i]
for img in dataset:
    imgcv = cv2.imread(os.path.join(datadir,img))
    result = tfnet.return_predict(imgcv)
    for box in result:
        print(box)
        x1,y1,x2,y2 = (box['topleft']['x'],box['topleft']['y'],box['bottomright']['x'],box['bottomright']['y'])
        conf = box['confidence']
        label = box['label']
        print(x1,y1,x2,y2,conf,label)
        cv2.rectangle(imgcv,(x1,y1),(x2,y2),(0,255,0),4)
        cv2.putText(imgcv,label,(x1+10,y1+2),0,0.5,(0,0,0))
    cv2.imshow('detected objects',imgcv)
    cv2.waitKey()  
    cv2.destroyAllWindows()
print('done')    