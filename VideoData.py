import os
import cv2
import numpy as np
import xml.etree.ElementTree as et
#Chargement des données pour prédire une vidéo
class VideoData():
    def __init__(self,path):
        self.file = []
        self.path = path
        for file in os.listdir(path+"/image_02/data/"):
            self.file.append(file)

        self.fileSize = len(self.file)
        self.n = 0


        tree = et.parse(path+"/tracklet_labels.xml")
        root = tree.getroot()
        items=root.find("tracklets")
        self.items=items.findall("item")

        return




    def __iter__(self):
        return self

    # Fonction d'appel d'iterateur et qui retourne un couple (img,dim,orientation,conf)
    def __next__(self):
        self.fileSize
        if self.n < self.fileSize :
            file = self.file[self.n]
            image  = cv2.imread(self.path+"/image_02/data/"+file)
            self.n+=1
            return image,file,self.getTruthCars()

        else:
            raise StopIteration


#On récupère les paramétres des images affichées sur l'image à l'index n
    def getTruthCars(self):
        cars = []
        for it in self.items:
            object_type = it.find("objectType")
            if (object_type.text == "Car"):
                h= float(it.find("h").text)
                w= float(it.find("w").text)
                l= float(it.find("l").text)
                dim = np.array([h,w,l])
                first_frame =  int(it.find("first_frame").text)
                poses = it.find("poses")

                count = int(poses.find("count").text)

                if self.n >= first_frame and (self.n-first_frame) < count:
                    item = poses.findall("item")[self.n-first_frame]

                    rot = -float(item.find("rz").text)+np.pi/2
                    center = np.array([ -float(item.find("ty").text),float(item.find("tz").text),float(item.find("tx").text)])

                    cars.append(np.array([rot,center,dim]))
        return cars
