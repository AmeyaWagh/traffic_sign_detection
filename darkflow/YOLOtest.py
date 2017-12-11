import os

from darkflow.net.build import TFNet
import cv2
import numpy as np
import csv
import json
import traceback
from lxml import etree as ET


# fourcc = cv2.VideoWriter_fourcc(*'XVID')

class objectDetector():
    def __init__(self,video=False):
        self.config = json.load(open('../config3.json'))
        self.video=video
        print(self.config)
        self.options = self.config['yoloConfig']
        self.tfnet = TFNet(self.options)
        self.predictThresh = 0.05
        self.getAnnotations()
        print(self.anotations_list)
        if self.video:
            # self.cap = cv2.VideoCapture(0)
            self.cap = cv2.VideoCapture('../../WPI_vdo.mov')
            self.out = cv2.VideoWriter('output.avi',-1, 20.0, (640,480))

    def getlabel(self,xmlPath):
        tree = ET.parse(xmlPath)
        root = tree.getroot()
        print(root[6][0].text) 

    def getAnnotations(self):
        with open('../train/FileSequence.txt') as seqFile:
        # with open('./pittsburgh/FileSequence.txt') as seqFile:
            self.anotations_list = seqFile.readlines()
        self.anotations_list = [seq.split('\n')[0] for seq in self.anotations_list]    

    def drawBoundingBox(self,imgcv,result):
        #finding max val
        self.predictThresh=max([box['confidence'] for box in result])
        for box in result:
            # print(box)
            x1,y1,x2,y2 = (box['topleft']['x'],box['topleft']['y'],box['bottomright']['x'],box['bottomright']['y'])
            conf = box['confidence']
            # print(conf)
            label = box['label']
            print("label",label,"confidence",conf)
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

    def evaluate(self):
        pass

    def processFrames(self):
        if self.video:
            while True:
                try:
                    ret,imgcv = self.cap.read()
                    result = self.tfnet.return_predict(imgcv)
                    imgcv = self.drawBoundingBox(imgcv,result)        
                    self.out.write(imgcv)
                    cv2.imshow('detected objects',imgcv)
                    if cv2.waitKey(10) == ord('q'):
                        self.cap.release()
                        self.out.release()
                        print('exitting loop')
                        break
                except Exception as e:
                    if self.video:
                        self.cap.release()
                        self.out.release()
                    traceback.print_exc()
                    break

        else:        
            for img in self.anotations_list:
                try:
                    img = [img]
                    
                    # if self.video:
                    #     ret,imgcv = self.cap.read()
                    #     # if ret:
                    #         # print("Done")
                    # else:
                    print('\n\nprocessing >> ',img[0],)
                    imgcv = cv2.imread(os.path.join('../train/images',img[0]))
                    # imgcv = cv2.imread(os.path.join('./pittsburgh/images',img[0]))
                    self.getlabel(os.path.join('./train/annotations',img[0]+".xml"))
                    result = self.tfnet.return_predict(imgcv)
                    # print(result[0]["label"],img[0])
                    imgcv = self.drawBoundingBox(imgcv,result)        
                    # self.out.write(imgcv)
                    cv2.imshow('detected objects',imgcv)
                    if cv2.waitKey(10) == ord('q'):
                        print('exitting loop')
                        break
                except Exception as e:
                    traceback.print_exc()
                    if self.video:
                        self.cap.release()
                        # self.out.release()
                    cv2.destroyAllWindows()
                    print('exitting program')

        if self.video:
            self.cap.release()
            # self.out.release()
            cv2.destroyAllWindows()
            print("released and done")

if __name__ == '__main__':
    det = objectDetector(video=False)
    det.processFrames()
#23515
