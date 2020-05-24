import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from PIL import Image

#Redimensionner une image en une image de 224x224
def getRezisedImage(image,dim):
    image = image[dim[0][1]:dim[1][1],dim[0][0]:dim[1][0]]
    image  = cv2.resize(image, dsize=(224, 224), interpolation=cv2.INTER_CUBIC)

    return image


def calc_theta_ray( img, box_2d, proj_matrix):
    width = img.shape[1]
    center = (box_2d[1][0] + box_2d[0][0]) / 2
    dx = center - (width / 2)
    angle = np.arctan(  abs(dx) / proj_matrix[0][0] )*np.sign(dx)

    return angle


def drawCube(img,vertex):
    vertex_idx = [[0,1],[0,2],[0,6],[7,6],[7,1],[7,5],[6,4],[1,3],[2,3],[2,4],[5,4],[5,4],[5,3]]
    for idx in vertex_idx:
        img = cv2.line(img,(int(vertex[0][idx[0]]),int(vertex[1][idx[0]])),(int(vertex[0][idx[1]]),int(vertex[1][idx[1]])), (0  , 191, 255),2)
    # img = cv2.circle(img, (int(vertex[0][7]),int(vertex[1][7])), radius=3, color=(0, 0, 255), thickness=-1)
    return img

def plot2DRectGraph(ax,center,dim,angle,color):
    points = np.array([[-dim[2]/2, -dim[0]/2],
                      [-dim[2]/2, dim[0]/2],
                      [dim[2]/2, dim[0]/2],
                      [dim[2]/2, -dim[0]/2]])
    P = np.array([[np.cos(angle),np.sin(angle)],[-np.sin(angle),np.cos(angle)]])
    Z = np.zeros((4,2))
    C = np.array([center[0],center[2]])
    for i in range(4): Z[i,:] =  np.dot(P,points[i,:]) + C.T

    # ax.scatter(Z[:, 0], Z[:, 1])
    rect = Polygon(Z,linewidth=1,color = color)
    ax.add_patch(rect)


#Fonctions qui permettent la sauvegarde d'un canvas matplot vers une image RGB
def fig2data ( fig ):
    fig.canvas.draw ( )

    # Get the RGBA buffer from the figure
    w,h = fig.canvas.get_width_height()
    buf = np.fromstring ( fig.canvas.tostring_argb(), dtype=np.uint8 )
    buf.shape = ( w, h,4 )

    # canvas.tostring_argb give pixmap in ARGB mode. Roll the ALPHA channel to have it in RGBA mode
    buf = np.roll ( buf, 3, axis = 2 )
    return buf
def fig2img ( fig ):
    buf = fig2data ( fig )
    w, h, d = buf.shape
    return Image.frombytes( "RGBA", ( w ,h ), buf.tostring( ) ).convert('RGB')
