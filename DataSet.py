import os
import tensorflow as tf
import numpy as np
import cv2
import utilsImage


class DataSet():
#class DataSet():
    def __init__(self,bins,shuffle):
        nbr_max_img = 0

        #On crée les intervalles pour les bins
        self.bins = bins
        self.bins_inter = []
        self.shuffle = shuffle
        for i in range(self.bins):
            self.bins_inter.append((i*2*np.pi/self.bins,(i+1)*2*np.pi/self.bins))


        #Différents dossiers de la base de données
        self.label_path = "/train/label/"
        self.img_path =   "/train/images/"
        self.path = os.path.abspath(os.path.dirname(__file__))

        ## On récupère les noms des fichiers (images ,label , etc )
        self.image_names = []
        for file in os.listdir(self.path+self.img_path):
            self.image_names.append(os.path.splitext(file)[0])

        # self.image_names = self.image_names[:12]

        ## On récupère nos données utiles des fichiers txt (dimensions, position , angles, etc )
        self.data = []
        self.dataSize = 0
        for file_name in self.image_names:
            with open(self.path + self.label_path + file_name+".txt") as file:
                for line_num,line in enumerate(file):
                    line = line[:-1].split(' ')
                    obj_class = line[0]
                    if obj_class == "Car":
                        dimension = np.array([float(line[8]), float(line[9]), float(line[10])], dtype=np.double)
                        posImage = np.array([(int(float(line[4])), int(float(line[5]))), (int(float(line[6])), int(float(line[7])))], dtype=np.int)
                        angle = float(line[3])+np.pi

                        infos = self.getInfos(angle)
                        self.data.append([file_name,line_num,posImage,dimension,infos[0],infos[1],infos[2]])

        self.dataSize = len(self.data)

    ## fonction qui permet de récuperer la confidence et l'orentation
    ## fournir un angle compris entre [0,2pi]
    def getInfos(self,angle):
        orientation = np.zeros((self.bins,2))
        confidence = np.zeros(self.bins)
        for i in range(len(self.bins_inter)):
            if(self.bins_inter[i][0]<=angle and angle<= self.bins_inter[i][1]):
                angle -=  self.bins_inter[i][0]+ np.pi/self.bins
                orientation[i] = np.array([np.cos(angle),np.sin(angle)])
                confidence[i] = 1
                break
        return  (orientation,confidence,angle)



    def __iter__(self):
        self.n = 0
        self.indices = np.arange(0, self.dataSize)
        if(self.shuffle):
            np.random.shuffle(self.indices)

        return self

    # Fonction d'appel d'iterateur et qui retourne un couple (img,dim,orientation,conf)
    def __next__(self):
        if self.n < self.dataSize:
            indice = self.indices[self.n]
            self.n+=1
            image =  cv2.imread(self.path+self.img_path+self.data[indice][0]+".png")
            return (utilsImage.getRezisedImage(image,self.data[indice][2]),self.data[indice][3],self.data[indice][4],self.data[indice ][5],self.data[indice ][6])

        else:
            raise StopIteration

    def __call__(self):
        return self


    ## Fonction qui redimensionne une image vers une taille de 224*224
    # def getImage(self, path ,dim):
    #     image =  cv2.imread(path)
    #     image = image[dim[0][1]:dim[1][1],dim[0][0]:dim[1][0]]
    #     image  = cv2.resize(image, dsize=(224, 224), interpolation=cv2.INTER_CUBIC)
    #
    #     return image
