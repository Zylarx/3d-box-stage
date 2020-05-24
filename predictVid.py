from keras.applications.vgg19 import VGG19
import tensorflow as tf
import numpy as np
import os
from Model import Orentiation,Dimension,Confidence
import cv2
import utils
import utilsImage
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import yolo
import VideoData as vd


bins = 2
orientation_model = Orentiation(bins)
confidence_model = Confidence(bins)
dimension_model = Dimension()

# On charge les données des images
path = os.path.abspath(os.path.dirname(__file__))

imagePath = path+"/test/test3"
vidData = vd.VideoData(imagePath)
# matrixCamPath = path+"/train/calib/"+imgId+ ".txt"
# matrixCam = utils.getCamMatrix(matrixCamPath)

matrixCam = np.array([[7.215377e+02, 0.000000e+00 ,6.095593e+02 ,4.485728e+01], [0.000000e+00 ,7.215377e+02, 1.728540e+02 ,2.163791e-01], [0.000000e+00, 0.000000e+00, 1.000000e+00, 2.745884e-03]])

# Chargement des modèles
confidence_model.model.load_weights("model/confidence_epoch_25.h5")
dimension_model.model.load_weights("model/dimension_epoch_25.h5")
orientation_model.model.load_weights("model/orientation_epoch_25.h5")
base_model = VGG19(weights='imagenet',include_top =False)


img_array = []
img_array_fig = []
#figure pour l'affichage 2d
fig = plt.figure()

#On parcourt les images d'un dossier
for img,file,truthCars in vidData:
    print(file)


    ax =  fig.add_subplot()
    ax.set_xlim([-20, 20])
    ax.set_ylim([0, 40])
    #Détection des voitures par YOLO
    size = (img.shape[1],img.shape[0])
    cars = yolo.getDetectedCars(img)

    #On affiche toutes les voitures labélisées sur notre graphe 2d
    for trueCars in truthCars:

        utilsImage.plot2DRectGraph(ax,trueCars[1],trueCars[2],trueCars[0],"blue")



    for car in cars:

        imgResize = utilsImage.getRezisedImage(img, car)
        imgResize = np.array([imgResize])
         # On applique les modèles de prédiction pour la voiture détectée
        dataInput =  base_model.predict(imgResize)
        dataInput = dataInput.reshape((1,7*7*512))
        dim = dimension_model.model.predict(dataInput)[0]
        conf = confidence_model.model.predict(dataInput)
        orient = orientation_model.model.predict(dataInput)[0]

        #On calcule l'angle theta (angle du centre de l'image au centre de la voiture )
        thetaimg  = utilsImage.calc_theta_ray(img,car,matrixCam)

        # On calcule l'angle de la voiture en elle-même
        conf_indice = np.argmax(conf,axis=1)
        cos = orient[2*conf_indice]
        sin  = orient[2*conf_indice+1]
        angle = np.arctan2(sin,cos)+np.pi/bins-np.pi+conf_indice*2*np.pi/bins
        orientGlobal = angle[0]
        orientLocal = (angle+thetaimg)[0]




        center,error = utils.calc3DBox(dim,car,orientLocal,matrixCam,orientGlobal)
        #On dessine les boxs
        # if(error<10000):
        #On considère la detection comme valide
        point = utils.plot3dBoundingBox(dim,center,orientLocal,matrixCam)

        # On affiche le retangle détécté par yolo sur l'image
        # img = cv2.rectangle(img, tuple(car[0]), tuple(car[1]), (255,0,0), 2)
        #On affiche le cube 3d
        utilsImage.drawCube(img,point)


        #On affiche le rectangle sur le graph de la vue 2d
        utilsImage.plot2DRectGraph(ax,center,dim,orientLocal,"orange")


    img_array.append(img)

    #On sauvegarde la représentation 2d
    data = np.array(utilsImage.fig2img(fig))
    img_array_fig.append(data)
    size_fig = (data.shape[1],data.shape[0])
    fig.clf()
    #On sauvegarde le plan 2d


#sauvegarde de la video des boxs
out = cv2.VideoWriter('project.avi',cv2.VideoWriter_fourcc(*'DIVX'), 15, size)

for i in range(len(img_array)):
    out.write(img_array[i])
out.release()

#sauvegarde de la video 2d
out = cv2.VideoWriter('project2d.avi',cv2.VideoWriter_fourcc(*'DIVX'), 15, size_fig)

for i in range(len(img_array_fig)):
    out.write(img_array_fig[i])
out.release()
