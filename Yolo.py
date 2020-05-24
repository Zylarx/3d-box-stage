import cv2
import numpy as np
import os


#Class qui gére le modéle yolo et la detection des voitures à partir des images
class Yolo():
    def __init__(self):
        self.CONF = 0.5
        #chargement du modéle
        path = os.path.abspath(os.path.dirname(__file__))
        weightsPath = path+"/yolo/yolov3-spp.weights"
        configPath =path+"/yolo/yolov3-spp.cfg"

        #Etablie les predictions
        self.net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

        self.ln = self.net.getLayerNames()
        self.ln = [self.ln[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]

        return
    def getDetectedCars(self,image):
        H  = image.shape[0]
        W  = image.shape[1]



        blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416),swapRB=True, crop=False)
        self.net.setInput(blob)
        layerOutputs = self.net.forward(self.ln)

        cars = []

        for output in layerOutputs:
            # On regarde tout les objets détéctés
            for detection in output:
                scores = detection[5:]
                classID = np.argmax(scores)
                #Regarde si l'objet detecté est une voiture
                if classID == 2  :
                    confidence = scores[classID]

                    if confidence >= self.CONF:
                        box = detection[0:4] * np.array([W, H, W, H])
                        (centerX, centerY, width, height) = box.astype("int")

                        x = int(centerX - (width / 2))
                        y = int(centerY - (height / 2))
                        if not(x<10 or x+width>W-10):
                            if not(y<10 or y+height>H-10):
            		            #Ajoute la position de la box d'une voiture qui ne se situe pas dans l'un des coins de l'image
                                cars.append([[max(x,0), max(y,0)],[ min(x+int(width),W), min(y+int(height),H)]])
        return cars
