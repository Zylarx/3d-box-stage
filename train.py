from keras.applications.vgg19 import VGG19
from DataSet import DataSet
import tensorflow as tf
import numpy as np
from Model import Orentiation,Dimension,Confidence

batch_size = 6
epoch = 30
learning_rate = 0.0001
learning_rate_dim = 0.001
bins = 2
start_id = 0
# data_loss = open("data_loss.txt","w")

# Générateur de données
dataSet_obj = DataSet(bins,True)
dataSet_gen = tf.data.Dataset.from_generator(dataSet_obj, output_types=(tf.float32, tf.float32, tf.float32, tf.float32, tf.float32),output_shapes=(tf.TensorShape([224, 224,3]), tf.TensorShape([3]),tf.TensorShape([2,2]),tf.TensorShape([2]),tf.TensorShape([])))
dataset_batch_gen = dataSet_gen.batch(batch_size)

base_model = VGG19(weights='imagenet',include_top =False)
orientation_model = Orentiation(bins)
confidence_model = Confidence(bins)
dimension_model = Dimension()


optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate,momentum = 0.8)

# data_loss.write("dimension_loss;orientation_loss;confidence_loss")

# Chargement des modèles s'ils existent
if start_id !=0:
    confidence_model.model.load_weights("model/final/confidence_epoch_"+str(start_id)+".h5")
    dimension_model.model.load_weights("model/final/dimension_epoch_"+str(start_id)+".h5")
    orientation_model.model.load_weights("model/final/orientation_epoch_"+str(start_id)+".h5")


nbr_batch = int(dataSet_obj.dataSize/6)

#On entraine les modèles
for ep in range(start_id+1,epoch):
    epoch_loss_avg = tf.keras.metrics.Mean()
    epoch_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
    i=0
    for data in dataset_batch_gen:

        print("-------------- inter --------", ep)
        print(i," / ",nbr_batch )
        i+=1
        dataInput =  base_model.predict(data[0],steps = 1)
        dataInput = dataInput.reshape((data[0].shape[0],7*7*512))
        #
        loss_value_dim , grads_dim = dimension_model.grad(dataInput,data[1])
        optimizer.apply_gradients(zip(grads_dim, dimension_model.model.trainable_variables))
        tf.keras.backend.print_tensor(loss_value_dim)

        loss_value_conf , grads_conf = confidence_model.grad(dataInput,data[3])
        optimizer.apply_gradients(zip(grads_conf, confidence_model.model.trainable_variables))
        tf.keras.backend.print_tensor(loss_value_conf)
        #
        loss_value_orient , grads_orient = orientation_model.grad(dataInput,data[4],data[3])
        optimizer.apply_gradients(zip(grads_orient, orientation_model.model.trainable_variables))
        tf.keras.backend.print_tensor(loss_value_orient)
        # data_loss.write( str(tf.keras.backend.get_value(loss_value_dim))+";"+  str(tf.keras.backend.get_value(loss_value_orient))+";"+ str(tf.keras.backend.get_value(loss_value_conf)) +"\n")
    # On sauvergarde les modèles
    confidence_model.model.save("model/confidence"+"_epoch_"+str(ep)+".h5")
    orientation_model.model.save("model/orientation"+"_epoch_"+str(ep)+".h5")
    dimension_model.model.save("model/dimension"+"_epoch_"+str(ep)+".h5")
# data_loss.close()
