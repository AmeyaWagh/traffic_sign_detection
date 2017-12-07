import os
# import xml.etree.cElementTree as ET
from lxml import etree as ET
import inspect
import csv
import shutil


class Parser():

    def __init__(self, path=None,percentage=0.7):
        if not os.path.isdir('train'):
            os.mkdir('train')
            os.mkdir('train/images')
            os.mkdir('train/annotations')
        if not os.path.isdir('test'):
            os.mkdir('test')
            os.mkdir('test/images')
            os.mkdir('test/annotations')    
        self.datasetPath = path
        self.imgPath = './train/images'
        self.annotationsPath = './train/annotations'
        self.labels = set()
        # print([x if os.path.isdir(x)==True for x in os.listdir('./dataset')])
        self.LISAdatasetPath = './dataset'
        self.datasetDirs = []
        for x in os.listdir(self.LISAdatasetPath):
            if os.path.isdir(os.path.join(self.LISAdatasetPath,x)) and ("vid" in x):
                print('>>',x)
                self.datasetDirs.append(os.path.join(self.LISAdatasetPath,x))
        sorted(self.datasetDirs)
        print(self.datasetDirs)
        self.trainDataLen = int(percentage*len(self.datasetDirs))
        self.trainDatasetPath = self.datasetDirs[:self.trainDataLen]

        self.testDataLen = len(self.datasetDirs)-self.trainDataLen
        self.testDatasetPath = self.datasetDirs[self.trainDataLen+1:]
        print('train >>',self.trainDataLen)
        print('train >>',self.trainDatasetPath)
        print('test >>',self.testDataLen)
        print('test >>',self.testDatasetPath)
        print('\n\n')

    def generateDataset(self):
        # generate training dataset
        self.FileSequence = []
        self.imgPath = './train/images'
        self.annotationsPath = './train/annotations'
        for subdir in self.trainDatasetPath:
            subdir = os.path.join(subdir,os.listdir(subdir)[0])
            print('trainset >>',subdir)
            self.datasetPath=subdir
            self.generateDatasetFiles()
        self.generateFileSequence('./train')    

        # generate testing dataset
        self.FileSequence = []
        self.imgPath = './test/images'
        self.annotationsPath = './test/annotations'
        for subdir in self.testDatasetPath:
            subdir = os.path.join(subdir,os.listdir(subdir)[0])
            print('testset >>',subdir)
            self.datasetPath=subdir
            self.generateDatasetFiles()    
        self.generateFileSequence('./test')    

        print(self.labels)
        self.generateLabels()

    def generateDatasetFiles(self):
        with open(os.path.join(self.datasetPath,'frameAnnotations.csv')) as csvfile:
            anotations_list = csvfile.readlines()
            # print(anotations_list)
            anotations_list.pop(0)
            for sample in anotations_list:
                sample = sample.split(';')
                # print(sample)
                self.labels.add(sample[1])
                self.generateXML(file=sample[0],
                    label=sample[1],
                    _bndbox={
                        "xmin": sample[2],
                        "ymin": sample[3],
                        "xmax": sample[4],
                        "ymax": sample[5]})
                shutil.copy(
                    os.path.join(self.datasetPath,sample[0]),
                    self.imgPath)
                self.FileSequence.append(sample[0])
        
                # break

    def generateFileSequence(self,filePath):
        with open(os.path.join(filePath,'FileSequence.txt'),'w') as fp:
            for _file in self.FileSequence:
                fp.write(_file+'\n')

    def generateLabels(self):
        with open(os.path.join('./labels.txt'),'w') as fp:
            for label in self.labels:
                fp.write(label+'\n')

    def generateXML(self, folder='VOC2008',
                    file='00002.png',
                    _shape={
                    "width":704,
                    "height":480,
                    "depth":3,
                    },
                    label="person",
                    _bndbox={
                        "xmin": 135,
                        "ymin": 25,
                        "xmax": 236,
                        "ymax": 188}):

        root = ET.Element("annotations")
        ET.SubElement(root, "folder").text = folder
        ET.SubElement(root, "filename").text = file

        source = ET.SubElement(root, "source")
        ET.SubElement(source, "database").text = "The VOC2007 Database"
        ET.SubElement(source, "annotation").text = "PASCAL VOC2007"
        ET.SubElement(source, "image").text = "flickr"
        ET.SubElement(source, "flickrid").text = "341012865"

        owner = ET.SubElement(root, "owner")
        ET.SubElement(owner, "flickrid").text = "341012865"
        ET.SubElement(owner, "name").text = "John Doe"

        size = ET.SubElement(root, "size")
        ET.SubElement(size, "width").text = str(_shape["width"])
        ET.SubElement(size, "height").text = str(_shape["height"])
        ET.SubElement(size, "depth").text = str(_shape["depth"])

        segmented = ET.SubElement(root, "segmented").text = str(0)

        _object = ET.SubElement(root, "object")
        ET.SubElement(_object, "name").text = str(label)
        ET.SubElement(_object, "pose").text = "left"
        ET.SubElement(_object, "truncated").text = "0"
        ET.SubElement(_object, "difficult").text = "0"
        bndbox = ET.SubElement(_object, "bndbox")
        
        ET.SubElement(bndbox, "xmin").text = str(_bndbox["xmin"])
        ET.SubElement(bndbox, "xmax").text = str(_bndbox["xmax"])
        ET.SubElement(bndbox, "ymin").text = str(_bndbox["ymin"])
        ET.SubElement(bndbox, "ymax").text = str(_bndbox["ymax"])

        tree = ET.ElementTree(root)
        # print(inspect.getargspec(tree.write))
        tree.write(
            os.path.join(self.annotationsPath, "{}.xml".format(file)),
            pretty_print=True)


if __name__ == '__main__':
    gen = Parser(path='./dataset/vid0/frameAnnotations-vid_cmp2.avi_annotations/')
    # gen = Parser()
    gen.generateDataset()
    # gen.generateXML()
