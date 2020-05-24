import tensorflow as tf
import tensorflow.keras.layers as layer
from tensorflow.keras.models import Sequential
import numpy as np

class Orentiation():

    def __init__(self,bins):
        self.model = Sequential([
        # layer.Dense(256, input_shape=(1,3), activation=tf.nn.relu),
        layer.Dense(256, input_shape=(7*7*512,), activation=tf.nn.relu),
        layer.Dropout(0.5),
        layer.Dense(256, activation=tf.nn.relu),
        layer.Dropout(0.5),
        layer.Dense(2*bins)
        ])



        #Fonction qui reproduit la fonction de coût comme définit dans le rapport
    def loss_func(self,orientationCalBatch ,angleTruthBatch, confidenceTruthBatch ):
        size = confidenceTruthBatch.shape[0]
        self.colum_1_tensor = tf.keras.backend.constant(np.full((size,2),[1,0]))
        self.colum_2_tensor = tf.keras.backend.constant(np.full((size,2),[0,1]))

        ##On récupère les couples (cos(theta i),sin(theta i)) qui sont les bons c'est à dire à la postion du 1 dans confidenceTruthBatch
        orientationCalcReshape = tf.keras.backend.reshape(orientationCalBatch,(size,2,2))
        orientationCalcReshape = tf.keras.backend.switch(confidenceTruthBatch,orientationCalcReshape,0*orientationCalcReshape)
        orientationCalcReshape = tf.keras.backend.sum(orientationCalcReshape,axis = 1)

        #On scinde les matrices en deux en récuperant la premiere et la seconde colonne
        orientCalcColum1 = tf.keras.backend.switch(self.colum_1_tensor,orientationCalcReshape,0*orientationCalcReshape)
        orientCalcColum2 = tf.keras.backend.switch(self.colum_2_tensor,orientationCalcReshape,0*orientationCalcReshape)
        orientCalcColum1 = tf.keras.backend.sum(orientCalcColum1,axis = 1)
        orientCalcColum2 = tf.keras.backend.sum(orientCalcColum2,axis = 1)


        angleCal = tf.math.atan2(orientCalcColum2,orientCalcColum1)

        cos =tf.keras.backend.cos(angleTruthBatch - angleCal)


        return 1-tf.keras.backend.mean(tf.keras.backend.cos(angleTruthBatch - angleCal))

    def grad(self, input ,angleTruthBatch, confidenceTruthBatch):

        with tf.GradientTape() as tape:
            orientationCalBatch = self.model(input,training = True)
            loss_value = self.loss_func(orientationCalBatch,angleTruthBatch,confidenceTruthBatch)

        return  loss_value,tape.gradient(loss_value, self.model.trainable_variables)

class Dimension():

    def __init__(self):
        self.model = Sequential([
        layer.Dense(256, input_shape=(7*7*512,),dtype = tf.float32, activation=tf.nn.relu),
        layer.Dropout(0.5),
        layer.Dense(256, activation=tf.nn.relu),
        layer.Dropout(0.5),
        layer.Dense(3)])
        self.loss_obj = tf.keras.losses.MeanAbsoluteError()


    # def loss_func(self,input, y_true,training):
    #     HUBER_DELTA = 0.5
    #     y_pred = self.model(input,training =  training)
    #     tf.keras.backend.print_tensor(y_pred)
    #     tf.keras.backend.print_tensor(y_true)
    #     x   = tf.keras.backend.abs(y_true - y_pred)
    #     x   = tf.keras.backend.switch(x < HUBER_DELTA, 0.5 * x ** 2, HUBER_DELTA * (x - 0.5 * HUBER_DELTA))
    #     return  tf.keras.backend.sum(x)
    def loss_func(self,input, y_true,training):
        HUBER_DELTA = 0.5
        y_pred = self.model(input,training =  training)
        tf.keras.backend.print_tensor(y_pred)
        tf.keras.backend.print_tensor(y_true)
        return  self.loss_obj (y_true,y_pred)

    def grad(self, input, y_true):
        with tf.GradientTape() as tape:
            loss_value = self.loss_func(input,y_true,True)
        return  loss_value,tape.gradient(loss_value, self.model.trainable_variables)





class Confidence():

    def __init__(self,bins):
        self.model = Sequential([
        layer.Dense(256, input_shape=(7*7*512,), activation=tf.nn.relu),
        layer.Dropout(0.5),
        layer.Dense(256, activation=tf.nn.relu),
        layer.Dropout(0.5),
        layer.Dense(bins, activation=tf.nn.sigmoid)])

        self.loss_obj = tf.keras.losses.BinaryCrossentropy()


    def loss_func(self,input, y_true,training):
        y_ = self.model(input,training =  training)

        return self.loss_obj(y_true , y_)


    def grad(self, input, y_true):
        with tf.GradientTape() as tape:
            loss_value = self.loss_func(input, y_true, True)
        return  loss_value,tape.gradient(loss_value, self.model.trainable_variables)
