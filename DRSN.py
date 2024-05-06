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


K.set_learning_phase(1)
np.random.seed(42)

data = pandas.read_excel("essay_input2.xlsx")
values = data.values


XY = values
X = XY[:, ["GR", "RD", "RS", "CNC", "AC", "Derivative of AC"]]

scaler = MinMaxScaler(feature_range=(0,1))
X = scaler.fit_transform(X)

Y = XY[:, "FACIES"]

#The data, split between train and val sets
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.25, random_state = 0)

img_rows, img_cols = 1, 6

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, 7)
y_test = keras.utils.to_categorical(y_test, 7)

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
net = Conv2D(64, 1, padding='same', kernel_initializer='he_normal',kernel_regularizer=l2(1e-4))(net)
net = residual_shrinkage_block(net, 1, 64, downsample=True)
net = BatchNormalization()(net)
net = Activation('relu')(net)

net = GlobalAveragePooling2D()(net)
net = BatchNormalization()(net)
outputs = Dense(7,activation='softmax', kernel_initializer='he_normal',kernel_regularizer=l2(1e-4))(net)

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

bc = 32
history = model.fit(x_train, y_train,epochs=200, verbose=0, batch_size=bc,  validation_data=(x_test, y_test))

# get results
K.set_learning_phase(0)
DRSN_train_score = model.evaluate(x_train, y_train, batch_size=bc, verbose=0)
print('Train loss:', DRSN_train_score[0])
print('Train accuracy:', DRSN_train_score[1])
DRSN_test_score = model.evaluate(x_test, y_test, batch_size=bc, verbose=0)
print('Test loss:', DRSN_test_score[0])
print('Test accuracy:', DRSN_test_score[1])
