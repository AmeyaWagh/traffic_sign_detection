import os
# import xml.etree.cElementTree as ET
from lxml import etree as ET
import inspect


class Parser():

    def __init__(self, path=None):
        if not os.path.isdir('train'):
            os.mkdir('train')
            os.mkdir('train/images')
            os.mkdir('train/annotations')
        self.datasetPath = path
        self.imgPath = './train/images'
        self.annotationsPath = './train/annotations'

    def generateDataset():
        pass

    def generateXML(self, folder='VOC2008',
                    file='00002.png',
                    _shape={
                    "width":10,
                    "height":10,
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
            os.path.join(self.annotationsPath, "filename.xml"),
            pretty_print=True)


if __name__ == '__main__':
    gen = Parser()
    gen.generateXML()
