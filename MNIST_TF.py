#necessary imports
import tensorflow as tf
import numdifftools as nd
import numpy as np
import pandas as pd
import tqdm
import pickle
import matplotlib.pyplot as plt
#np.version.version
from scipy.optimize import minimize
from random import shuffle
from sklearn.decomposition import PCA

(x_train, y_train), (x_test, y_test)=tf.keras.datasets.mnist.load_data(path='mnist.npz')

x_train=(x_train.reshape((x_train.shape[0],x_train.shape[1]*x_train.shape[2])))
x_test=(x_test.reshape((x_test.shape[0],x_test.shape[1]*x_test.shape[2])))



class_name=9
indx_0_train=[i for i, x in enumerate(y_train) if x == class_name]
#indx_3=np.random.choice(indx_3, size=5000, replace=False, p=None)
indx_1_train=[i for i, x in enumerate(y_train) if x != class_name]
#indx_5=np.random.choice(indx_5, size=1000, replace=False, p=None)
shuffle(indx_0_train)
shuffle(indx_1_train)

#taking equal number of samples to train
x_train_0 = (x_train[indx_0_train])/255.0
x_train_1 = (x_train[indx_1_train[0:len(indx_0_train)]])/255.0
print(x_train_0.shape,x_train_1.shape)

indx_0_test=[i for i, x in enumerate(y_test) if x == class_name]
#indx_3=np.random.choice(indx_3, size=5000, replace=False, p=None)
indx_1_test=[i for i, x in enumerate(y_test) if x != class_name]
#indx_5=np.random.choice(indx_5, size=1000, replace=False, p=None)
shuffle(indx_0_test)
shuffle(indx_1_test)

#taking equal number of samples to test
x_test_0 = x_test[indx_0_test]/255.0
x_test_1 = x_test[indx_1_test[0:len(indx_0_test)]]/255.0
#x_test_0=x_test_0[0:100]
#x_test_1=x_test_1[0:100]
print(x_test_0.shape,x_test_1.shape)

# this coce is for making the input work on the tensorflow code
y_train_0=np.full((len(x_train_0)),0)
y_train_1=np.full((len(x_train_1)),1)
y_train_tf=np.concatenate((y_train_0, y_train_1), axis=0)
x_train_tf=np.concatenate((x_train_0, x_train_1), axis=0)
y_test_0=np.full((len(x_test_0)),0)
y_test_1=np.full((len(x_test_1)),1)
y_test_tf=np.concatenate((y_test_0, y_test_1), axis=0)
x_test_tf=np.concatenate((x_test_0, x_test_1), axis=0)

index=np.random.permutation(len(x_train_tf))

x_train_tf=x_train_tf[index]
y_train_tf=y_train_tf[index]

index_t=np.random.permutation(len(x_test_tf))
x_test_tf=x_train_tf[index_t]
y_test_tf=y_train_tf[index_t]

from keras.backend.tensorflow_backend import set_session

config = tf.ConfigProto(
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
    # device_count = {'GPU': 1}
)
print("-------------Starting training for "+str(class_name)+"vs all----------------")
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
set_session(session)
from keras.models import Sequential
model = Sequential()
from keras.layers import Dense
model.add(Dense(1, activation='sigmoid', input_shape=(784,)))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
# with session.as_default():
#     with session.graph.as_default():
# callbacks=model.fit(x_train_tf, y_train_tf, epochs=25, batch_size=128)
model.fit(x_train_tf, y_train_tf, epochs=100, batch_size=128)
# print(callbacks.history['loss'][-1])
# print(callbacks.history['acc'][-1])

loss_acc = model.evaluate(x_test_tf, y_test_tf, batch_size=128)
print(loss_acc)

