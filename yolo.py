import cv2
import numpy as np
import os

CONF = 0.5


def getDetectedCars(image):
    H  = image.shape[0]
    W  = image.shape[1]
    #chargement du modèle
    path = os.path.abspath(os.path.dirname(__file__))
    weightsPath = path+"/yolo/yolov3-spp.weights"
    configPath =path+"/yolo/yolov3-spp.cfg"

    #Etablie les predictions
    net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

    ln = net.getLayerNames()
    ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]


    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416),swapRB=True, crop=False)
    net.setInput(blob)
    layerOutputs = net.forward(ln)

    cars = []
    # loop over each of the layer outputs
    for output in layerOutputs:
    	# loop over each of the detections
    	for detection in output:
            scores = detection[5:]
            classID = np.argmax(scores)
            #Regarde si l'objet detecté est une voiture
            if classID == 2  :
                confidence = scores[classID]

                if confidence >= CONF:
                    box = detection[0:4] * np.array([W, H, W, H])
                    (centerX, centerY, width, height) = box.astype("int")

                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))
                    if not(x<10 or x+width>W-10):
                        if not(y<10 or y+height>H-10):
        		            #Ajoute la position de la box d'une voiture qui ne se situe pas dans l'un des coins de l'image
                            cars.append([[max(x,0), max(y,0)],[ min(x+int(width),W), min(y+int(height),H)]])
    return cars
