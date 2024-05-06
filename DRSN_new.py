# -*- coding: utf-8 -*-
"""
Created on Wed Aug  3 15:24:11 2022

@author: abaoluo
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 28 23:24:05 2019

Implemented using TensorFlow 1.0.1 and Keras 2.2.1
 
M. Zhao, S. Zhong, X. Fu, et al., Deep Residual Shrinkage Networks for Fault Diagnosis, 
IEEE Transactions on Industrial Informatics, 2019, DOI: 10.1109/TII.2019.2943898

There might be some problems in the Keras code. The weights in custom layers of models created using the Keras functional API may not be optimized.
https://www.reddit.com/r/MachineLearning/comments/hrawam/d_theres_a_flawbug_in_tensorflow_thats_preventing/

TensorFlow被曝存在严重bug，搭配Keras可能丢失权重
https://cloud.tencent.com/developer/news/661458

The TFLearn code is recommended for usage.
https://github.com/zhao62/Deep-Residual-Shrinkage-Networks/blob/master/DRSN_TFLearn.py

@author: super_9527
"""
import keras
import numpy as np
import pandas
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.layers import Dense, Conv2D, BatchNormalization, Activation, Dropout
from keras.layers import AveragePooling2D, Input, GlobalAveragePooling2D
from keras.optimizers import Adam, SGD
from keras.regularizers import l1, l2
from keras import backend as K
from keras.models import Model
from keras.layers.core import Lambda
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from keras.utils import to_categorical
from keras.utils.vis_utils import plot_model
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.metrics import recall_score, precision_score, f1_score
'''
class Metrics(keras.callbacks.Callback):
    def __init__(self, valid_data):
        super(Metrics, self).__init__()
        self.validation_data = valid_data
 
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        val_predict = np.argmax(self.model.predict(self.validation_data[0]), -1)
        val_targ = self.validation_data[1]
        if len(val_targ.shape) == 2 and val_targ.shape[1] != 1:
            val_targ = np.argmax(val_targ, -1)
 
        _val_f1 = f1_score(val_targ, val_predict, average='macro')
        _val_recall = recall_score(val_targ, val_predict, average='macro')
        _val_precision = precision_score(val_targ, val_predict, average='macro')
 
        logs['val_f1'] = _val_f1
        logs['val_recall'] = _val_recall
        logs['val_precision'] = _val_precision
        print(" — val_f1: %f — val_precision: %f — val_recall: %f" % (_val_f1, _val_precision, _val_recall))
        return 
'''

K.set_learning_phase(1)
np.random.seed(42)
tf.random.set_seed(42)
'''
df_tr = pandas.read_excel("datatrain4.xlsx", header = None)
#df_te = pandas.read_excel("data_te4.xlsx", header = None)
std = StandardScaler()
x_train = np.float_(df_tr.iloc[1:, 1:9])

y_train = np.float_(df_tr.iloc[1:, 10])
x_test = np.float_(df_te.iloc[1:, 1:9])

y_test = np.float_(df_te.iloc[1:, 10])
input_shape=(12,)
'''

data = pandas.read_excel("essay_zy1.xlsx")
values = data.values

scaler = MinMaxScaler(feature_range=(0,1))
#XY = scaler.fit_transform(values)
XY = values
X = XY[:, [1,2,4,6,7]]
X = scaler.fit_transform(X)
Y = XY[:, 9]
'''
std = StandardScaler()
X = np.array(values[:,0]).reshape(-1,1)
Y = np.array(values[:,1]).reshape(-1,1)
X = std.fit_transform(X)

'''
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.30, random_state = 0)
'''
x_train = X[:220, :]
y_train = Y[:220]
x_test = X[221:, :]
y_test = Y[221:]
'''

#lables = [0,1]
#num_classes = 2
# Input image dimensions
img_rows, img_cols = 1, 5

# The data, split between train and test sets
#(x_train, y_train), (x_test, y_test) = mnist.load_data()

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)
'''
# Noised data
x_train = x_train.astype('float32') / 255. + 0.5*np.random.random([x_train.shape[0], img_rows, img_cols, 1])
x_test = x_test.astype('float32') / 255. + 0.5*np.random.random([x_test.shape[0], img_rows, img_cols, 1])
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')
'''

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, 8)
y_test = keras.utils.to_categorical(y_test, 8)

def abs_backend(inputs):
    return K.abs(inputs)

def expand_dim_backend(inputs):
    return K.expand_dims(K.expand_dims(inputs,1),1)

def sign_backend(inputs):
    return K.sign(inputs)

def pad_backend(inputs, in_channels, out_channels):
    pad_dim = (out_channels - in_channels)//2
    inputs = K.expand_dims(inputs,-1)
    inputs = K.spatial_3d_padding(inputs, ((0,0),(0,0),(0,0)), 'channels_last')
    return K.squeeze(inputs, -1)

def scorer_f(estimator, X):   #your own scorer
      return np.mean(estimator.score_samples(X))

# Residual Shrinakge Block
def residual_shrinkage_block(incoming, nb_blocks, out_channels, downsample=False,
                             downsample_strides=2):
    
    residual = incoming
    in_channels = incoming.get_shape().as_list()[-1]
    
    for i in range(nb_blocks):
        
        identity = residual
       
        if not downsample:
            downsample_strides = 1
        
        residual = BatchNormalization()(residual)
        residual = Activation('relu')(residual)
        residual = Conv2D(out_channels, kernel_size=(1,1), strides=(downsample_strides, downsample_strides), 
                          padding='same', kernel_initializer='he_normal',
                          kernel_regularizer=l2(1e-4))(residual)
        '''
        residual = BatchNormalization()(residual)
        residual = Activation('relu')(residual)
        residual = Conv2D(out_channels, kernel_size=(3,3),
                          padding='same', kernel_initializer='he_normal')(residual)
        '''
        residual = BatchNormalization()(residual)
        residual = Activation('relu')(residual)
        residual = Conv2D(out_channels, kernel_size=(3,3),
                          padding='same', kernel_initializer='he_normal',
                          kernel_regularizer=l2(1e-4))(residual)
        # Calculate global means
        residual_abs = Lambda(abs_backend)(residual)
        abs_mean = GlobalAveragePooling2D()(residual_abs)
        
        # Calculate scaling coefficients
        scales = Dense(14, activation='tanh', kernel_initializer='he_normal', 
                       kernel_regularizer=l2(1e-4))(abs_mean)
        scales = BatchNormalization()(scales)
        scales = Activation('relu')(scales)
        scales = Dense(out_channels, activation='sigmoid',kernel_regularizer=l2(1e-4))(scales)
        scales = Lambda(expand_dim_backend)(scales)
        
        # Calculate thresholds
        thres = keras.layers.multiply([abs_mean, scales])
       
        # Soft thresholding
        sub = keras.layers.subtract([residual_abs, thres])
        zeros = keras.layers.subtract([sub, sub])
        n_sub = keras.layers.maximum([sub, zeros])
        residual = keras.layers.multiply([Lambda(sign_backend)(residual), n_sub])
        
        # Downsampling using the pooL-size of (1, 1)
        if downsample_strides > 1:
            identity = AveragePooling2D(pool_size=(1,1), strides=(2,2))(identity)
            
        # Zero_padding to match channels
        if in_channels != out_channels:
            identity = Lambda(pad_backend, arguments={'in_channels':in_channels,'out_channels':out_channels})(identity)
        
       
        residual = keras.layers.add([residual, identity])
    
    return residual


# define and train a model
inputs = Input(shape=input_shape)
net = Conv2D(128, 3, padding='same', kernel_initializer='he_normal',kernel_regularizer=l2(1e-4))(inputs)
net = BatchNormalization()(net)
#net = Dropout(0.3)(net)
net = Conv2D(64, 1, padding='same', kernel_initializer='he_normal',kernel_regularizer=l2(1e-4))(net)
net = residual_shrinkage_block(net, 1, 64, downsample=True)
net = BatchNormalization()(net)
net = Dropout(0.5)(net)
#net = residual_shrinkage_block(net, 1, 4)
#net = BatchNormalization()(net)
net = Activation('relu')(net)

net = GlobalAveragePooling2D()(net)
net = BatchNormalization()(net)
outputs = Dense(8,activation='softmax', kernel_initializer='he_normal',kernel_regularizer=l2(1e-4))(net)

METRICS = [
      #keras.metrics.TruePositives(name='tp'),
      #keras.metrics.FalsePositives(name='fp'),
      #keras.metrics.TrueNegatives(name='tn'),
      #keras.metrics.FalseNegatives(name='fn'), 
      keras.metrics.CategoricalAccuracy(name='accuracy'),
      keras.metrics.Precision(name='precision'),
      keras.metrics.Recall(name='recall'),
      #keras.metrics.AUC(name='auc'),
]

model = Model(inputs=inputs, outputs=outputs)
adam = keras.optimizers.Adam(lr=0.02)
model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=METRICS)

'''
#参数优化
model = KerasClassifier(build_fn=model, verbose=0)
bc = [10, 20, 40, 60, 80, 100]
epochs = [200, 500, 1000]
param_grid = {'batch_size': list((10,20,40)), 'epochs': list((50,100,200))}
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1)
grid_result = grid.fit(x_train, y_train)
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
for params, mean_score, scores in grid_result.grid_scores_:
    print("%f (%f) with: %r" % (scores.mean(), scores.std(), params))
'''

bc = 128
#m = Metrics(valid_data = (x_test, y_test))
#model.fit(x_train, y_train, batch_size=100, epochs=5, verbose=1, validation_data=(x_test, y_test))
history = model.fit(x_train, y_train,epochs=200, verbose=0, batch_size=bc,  validation_data=(x_test, y_test))

#绘制
precision = history.history['precision']
val_precision = history.history['val_precision']
epochs = range(1, len(precision) + 1)

plt.figure(1)
#plt.clf()
plt.plot(epochs, precision, 'y', label='Dropout')
plt.plot(epochs, val_precision, 'y', ls='--', label='Dropout')
plt.ylim((0, 2))
plt.title('Training and validation precision', fontsize=20)
plt.xlabel('Epochs', fontsize=18)
plt.ylabel('Loss', fontsize=18)
plt.legend(fontsize=18, markerscale=2., scatterpoints=1)
#plt.legend()
plt.show()


plt.figure(2)
plt.clf()
recall = history.history['recall']
val_recall = history.history['val_recall']
plt.plot(epochs, recall, 'b', label='Training recall')
plt.plot(epochs, val_recall, 'r', label='Validation recall')
plt.ylim((0, 1))
plt.title('Training and validation recall', fontsize=20)
plt.xlabel('Epochs', fontsize=18)
plt.ylabel('Accuracy', fontsize=18)
plt.legend(fontsize=18, markerscale=2., scatterpoints=1)
#plt.legend()
plt.show()

'''
plt.figure(3)
plt.clf()
accuracy = history.history['accuracy']
val_accuracy = history.history['val_accuracy']
plt.plot(epochs, accuracy, 'b', label='Training acc')
plt.plot(epochs, val_accuracy, 'r', label='Validation acc')
plt.ylim((0, 1))
plt.title('Training and validation accuracy', fontsize=20)
plt.xlabel('Epochs', fontsize=18)
plt.ylabel('Accuracy', fontsize=18)
plt.legend(fontsize=18, markerscale=2., scatterpoints=1)
#plt.legend()
plt.show()
'''
plt.figure(4)
#plt.clf()
loss = history.history['loss']
val_loss = history.history['val_loss']
plt.plot(epochs, loss, 'y', label='Dropout')
plt.plot(epochs, val_loss, 'y', ls='--', label='Dropout')
plt.ylim((0, 2))
plt.title('Training and validation loss', fontsize=20)
plt.xlabel('Epochs', fontsize=18)
plt.ylabel('Loss', fontsize=18)
plt.legend(fontsize=18, markerscale=2., scatterpoints=1)
#plt.legend()
plt.show()
# get results
K.set_learning_phase(0)
DRSN_train_score = model.evaluate(x_train, y_train, batch_size=bc, verbose=0)
print('Train loss:', DRSN_train_score[0])
print('Train accuracy:', DRSN_train_score[1])
DRSN_test_score = model.evaluate(x_test, y_test, batch_size=bc, verbose=0)
x_test = np.array(x_test)

DRSN_test = model.predict(x_test)
#result_i = (np.abs(DRSN_test - y_test)/y_test).mean()
predicts = np.argmax(DRSN_test, axis=1)
y_t = np.argmax(y_test, axis=1)
print('Test loss:', DRSN_test_score[0])
print('Test accuracy:', DRSN_test_score[1])
#model.save('my_model55.h5')