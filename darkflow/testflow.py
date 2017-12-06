# import sys
# sys.path.insert(0,'./darkflow')
import os

from darkflow.net.build import TFNet
import cv2
import numpy as np
import csv
import json



class objectDetector():
    def __init__(self,video=False):
        self.config = json.load(open('../config.json'))
        self.video=video
        print(self.config)
        self.options = self.config['yoloConfig']
        self.tfnet = TFNet(self.options)
        self.predictThresh = 0.4
        self.getAnnotations()
        if self.video:
            self.cap = cv2.VideoCapture(0)

    def getAnnotations(self):
        with open(os.path.join('../',self.config["dataset"],'allAnnotations.csv')) as csvfile:
            self.anotations_list = csvfile.readlines()
            # print(anotations_list)
            for row in self.anotations_list:
                print(row.split(';'))
                break
        print(self.anotations_list.pop(0))

    def drawBoundingBox(self,imgcv,result):
        for box in result:
            # print(box)
            x1,y1,x2,y2 = (box['topleft']['x'],box['topleft']['y'],box['bottomright']['x'],box['bottomright']['y'])
            conf = box['confidence']
            # print(conf)
            label = box['label']
            if conf < self.predictThresh:
                continue
            # print(x1,y1,x2,y2,conf,label)
            cv2.rectangle(imgcv,(x1,y1),(x2,y2),(0,255,0),6)
            labelSize=cv2.getTextSize(label,cv2.FONT_HERSHEY_COMPLEX,0.5,2)
            # print('labelSize>>',labelSize)
            _x1 = x1
            _y1 = y1#+int(labelSize[0][1]/2)
            _x2 = _x1+labelSize[0][0]
            _y2 = y1-int(labelSize[0][1])
            cv2.rectangle(imgcv,(_x1,_y1),(_x2,_y2),(0,255,0),cv2.FILLED)
            cv2.putText(imgcv,label,(x1,y1),cv2.FONT_HERSHEY_COMPLEX,0.5,(0,0,0),1)
        return imgcv

    def processFrames(self):
        try:
            for img in self.anotations_list:
                img = img.split(';')
                # print(img)
                # ret,imgcv = cap.read()
                if self.video:
                    ret,imgcv = self.cap.read()
                else:
                    imgcv = cv2.imread(os.path.join('../',self.config["dataset"],img[0]))
                result = self.tfnet.return_predict(imgcv)
                imgcv = self.drawBoundingBox(imgcv,result)        
                cv2.imshow('detected objects',imgcv)
                if cv2.waitKey(10) == ord('q'):
                    print('exitting loop')
                    break
        except KeyboardInterrupt:
            cv2.destroyAllWindows()
            print('exitting program')


if __name__ == '__main__':
    det = objectDetector(video=True)
    det.processFrames()

























# print("Initialisation Done")
# # cap = cv2.VideoCapture(0)

# config = json.load(open('../config.json'))

# options = {"model": "cfg/yolo.cfg", "load": "bin/yolo.weights", "threshold": 0.1,"gpu":0.5}
# # options = config['yoloConfig']

# tfnet = TFNet(options)

# # '../dataset/allAnnotations.csv'
# with open(os.path.join('../',config["dataset"],'allAnnotations.csv')) as csvfile:
#     anotations_list = csvfile.readlines()
#     # print(anotations_list)
#     for row in anotations_list:
#         print(row.split(';'))
#         break
# print(anotations_list.pop(0))
# print('doing')   
# print('press q to exit') 
# # datadir = '../dataset/vid0/frameAnnotations-vid_cmp2.avi_annotations'
# # dataset = os.listdir(datadir)
# # dataset = [i for i in dataset if '.png' in i]
# predictThresh = 0.4
# # for img in dataset:
# try:
#     for img in anotations_list:
#         img = img.split(';')
#         # print(img)
#         # ret,imgcv = cap.read()
#         imgcv = cv2.imread(os.path.join('../',config["dataset"],img[0]))
#         result = tfnet.return_predict(imgcv)
#         for box in result:
#             # print(box)
#             x1,y1,x2,y2 = (box['topleft']['x'],box['topleft']['y'],box['bottomright']['x'],box['bottomright']['y'])
#             conf = box['confidence']
#             # print(conf)
#             label = box['label']
#             if conf < predictThresh:
#                 continue
#             # print(x1,y1,x2,y2,conf,label)
#             cv2.rectangle(imgcv,(x1,y1),(x2,y2),(0,255,0),6)
#             labelSize=cv2.getTextSize(label,cv2.FONT_HERSHEY_COMPLEX,0.5,2)
#             # print('labelSize>>',labelSize)
#             _x1 = x1
#             _y1 = y1#+int(labelSize[0][1]/2)
#             _x2 = _x1+labelSize[0][0]
#             _y2 = y1-int(labelSize[0][1])
#             cv2.rectangle(imgcv,(_x1,_y1),(_x2,_y2),(0,255,0),cv2.FILLED)
#             cv2.putText(imgcv,label,(x1,y1),cv2.FONT_HERSHEY_COMPLEX,0.5,(0,0,0),1)
#         cv2.imshow('detected objects',imgcv)
#         if cv2.waitKey(10) == ord('q'):
#             break

# except KeyboardInterrupt:
#     cv2.destroyAllWindows()

# print('done')
# quit()    