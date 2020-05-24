import numpy as np

#Retourne la matrice de passage sauvegardée dans un .txt
def getCamMatrix(path):
    for line in open(path):
        if 'P2:' in line:
            mat = line.strip().split(' ')
            return np.array([[float(mat[1]),float(mat[2]),float(mat[3]),float(mat[4])],[float(mat[5]),float(mat[6]),float(mat[7]),float(mat[8])],[float(mat[9]),float(mat[10]),float(mat[11]),float(mat[12])]])
    return []

#Retourne la matrice de la rotation autour de l'axe y d'angle theta
def rotMatrix3dY(theta):
    return np.array([[np.cos(theta),0,np.sin(theta)],[0,1,0],[-np.sin(theta),0,np.cos(theta)]])

# Prend un angle dans le sens trigonometrique et retourne les vecteurs de contraintes
def getBoundingPoints(orientGlobal, dim):

    h,w,l = dim
    x_corners = np.array([l,l,l,l, 0, 0, 0, 0])
    y_corners =  np.array([ h, 0,  h, 0,  h, 0,  h, 0])
    z_corners =  np.array([0, 0,  w,  w,  w,  w, 0, 0])

    x_corners -= l/2
    y_corners -= h/2
    z_corners -= w/2
    cubeVertex = np.transpose(np.array([x_corners, y_corners, z_corners]))

    constraints = []


    #Disjonction de cas en fonction de l'orientation globale
    if orientGlobal >= 0 and orientGlobal <= np.pi/2:
        xmin = cubeVertex[7]
        xmax = cubeVertex[3]
        ymax = cubeVertex[0]
        ymin = cubeVertex[1]
        constraints.append([xmin,xmax ,ymin, ymax])
    elif orientGlobal >= np.pi/2 and orientGlobal <= np.pi:
        xmin = cubeVertex[1]
        xmax = cubeVertex[5]
        ymax = cubeVertex[2]
        ymin = cubeVertex[3]
        constraints.append([xmin,xmax ,ymin, ymax])
    elif orientGlobal >=np.pi and orientGlobal <=3*np.pi/2:
        xmin = cubeVertex[3]
        xmax = cubeVertex[7]
        ymax = cubeVertex[4]
        ymin = cubeVertex[5]
        constraints.append([xmin,xmax ,ymin, ymax])
    elif orientGlobal >= 3*np.pi/2 and orientGlobal <=2*np.pi:
        xmin = cubeVertex[5]
        xmax = cubeVertex[1]
        ymax = cubeVertex[6]
        ymin = cubeVertex[7]
        constraints.append([xmin,xmax ,ymin, ymax])

    return constraints


#Fonction qui applique le principe de contraintes aux points et calcule leurs regressions lineaires
def get3DBoxCenter(dim, pos , orient,camMatrix,orientGlobal):
    h,w,l  = dim

    xmin_2dBox,ymin_2dBox = pos[0]
    xmax_2dBox,ymax_2dBox = pos[1]
    constraintPoints = getBoundingPoints(orientGlobal,dim)

    box2d = np.array([xmin_2dBox,xmax_2dBox,ymin_2dBox,ymax_2dBox])

    rotMat = rotMatrix3dY(orient)

    pos = 0
    errorMin = -1
    for contraint in constraintPoints:

        indice = [0,0,1,1]
        A = np.zeros((4,3))
        B = np.zeros((4,1))
        #On calcule les 4 équations à résoudre
        for i in range(4):
            vectRot = np.dot(rotMat,contraint[i])

            vectHomo  = np.ones(4)
            vectHomo[:3] = vectRot

            vectHomo= np.dot(camMatrix,vectHomo)

            A[i] =  camMatrix[indice[i],:3] - box2d[i]* camMatrix[2,:3]
            B[i] =  box2d[i]*vectHomo[2]-vectHomo[indice[i]]

           #On applique la régression lineaire
        loc, error, rank, s = np.linalg.lstsq(A[:4], B[:4], rcond=None)


        if(errorMin ==-1 or errorMin > error):
            errorMin = error
            pos = loc


    return (pos,errorMin)



def calc3DBox(dimension, pos, orient, camMatrix,orientGlobal):
    l,h,w = dimension

    if(orient <0):
        orient =2*np.pi +orient
    if(orientGlobal <0):
        orientGlobal =2*np.pi +orientGlobal

    return get3DBoxCenter(dimension, pos, orient, camMatrix,orientGlobal)

#Fonction qui affiche une liste de points sur l'imge à partir de leurs position 3d
def plot3dPoint(point,camMatrix):

    point = np.append(point, 1)
    point = np.dot(matrixCam,point)
    point = point[:2]/point[2]
    plt.scatter(point[0],point[1])
    plt.imshow(img)

#Retourne l'ensemble des coins de la box selon leurs postion dans l'espace avec la camera comme origine
def plot3dBoundingBox(dim, center , orient,camMatrix):
    rotMat = rotMatrix3dY(orient)
    h,w,l = dim
    x_corners = [l,l,l,l, 0, 0, 0, 0,l/2]
    y_corners = [ h, 0,  h, 0,  h, 0,  h, 0,h/2]
    z_corners = [0, 0,  w,  w,  w,  w, 0, 0,0]

    x_corners -=  l / 2
    y_corners -= h/ 2
    z_corners -= w / 2

    corners_3d = np.array([x_corners, y_corners, z_corners])

    corners_3d = np.dot(rotMat,corners_3d)

    for i in range(9):
        corners_3d[0][i] += center[0]
        corners_3d[1][i] += center[1]
        corners_3d[2][i] += center[2]
    corners_3d = np.array([corners_3d[0], corners_3d[1], corners_3d[2],np.ones(9)])
    corners_3d = np.dot(camMatrix,corners_3d)
    for i in range(9):
        corners_3d[0][i] /= corners_3d[2][i]
        corners_3d[1][i] /= corners_3d[2][i]


    return corners_3d[:2]
