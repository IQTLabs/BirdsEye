##############################################################
#  Radio Fingerprinting in RFML Environment                 #
#        Neural Network for Direction Finding Data 2020     #
#         Author: Debashri Roy                              #
#############################################################
############ IMPORTING NECESSARY PACKAGES ################
import itertools
import os
import random
import sys  # Package to get command line arguments
import time  # Package is for computing execution time
from array import array

import keras
import keras.models as models
import matplotlib
import matplotlib.pyplot as plt
import numpy as np  # Package for numerical computation
import pandas as pd
import scipy
import tensorflow as tf
from keras import backend as K
from keras.callbacks import ModelCheckpoint
from keras.callbacks import ReduceLROnPlateau
from keras.layers import Embedding
from keras.layers import GRU
from keras.layers import GRUCell
from keras.layers import LSTM
from keras.layers import LSTMCell
from keras.layers import RNN
from keras.layers import SimpleRNN
from keras.layers import SimpleRNNCell
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.advanced_activations import PReLU
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import Convolution1D
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.convolutional import ZeroPadding2D
from keras.layers.convolutional_recurrent import ConvLSTM2D
from keras.layers.core import Activation
from keras.layers.core import Dense
from keras.layers.core import Dropout
from keras.layers.core import Flatten
from keras.layers.core import Reshape
from keras.layers.noise import GaussianNoise
from keras.models import Sequential
from keras.regularizers import *
from keras.utils import np_utils
from keras.utils.np_utils import to_categorical
from mpl_toolkits.mplot3d import Axes3D
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.optimizers import SGD
np.set_printoptions(threshold=np.inf)  # To print each elements

#   by setting env variables before Keras import you can set up which backend
#os.environ["KERAS_BACKEND"] = "theano"
os.environ['KERAS_BACKEND'] = 'tensorflow'
os.environ['THEANO_FLAGS'] = 'device=cuda0, dnn.enabled=False'
#import theano
#theano.config.mode = ""


#import theano as th
#import theano.tensor as T
#from keras.optimizers import adam_v2, Nadam, Adadelta
#from keras.optimizers import Adam, RMSprop, Adagrad
#from keras.regularizers import l2, activity_l2
# import BatchNormalization


# matplotlib.use('TkAgg')
matplotlib.use('Agg')
#import seaborn as sns


########## FUNCTIONS TO CALCULATE F SCORE OF THE MODEL ###############


def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))
######################################################################


data_file_loc1 = '/media/oem/data/data/test-cut-1.raw'
data_file_loc2 = '/media/oem/data/data/test-cut-2.raw'
data_file_loc3 = '/media/oem/data/data/test-cut-3.raw'
data_file_loc4 = '/media/oem/data/data/test-cut-4.raw'

#data_file_loc2 = '/media/oem/data/data/set-4/birds-eye-dji-p3-5745MHz-lp-500m-90gain-0deg.raw'
#data_file_loc3 = '/media/oem/data/data/set-4/birds-eye-dji-p3-5745MHz-lp-500m-90gain-0deg.raw'
#data_file_loc4 = '/media/oem/data/data/set-4/birds-eye-dji-p3-5745MHz-lp-500m-90gain-0deg.raw'

#data_file_loc1 = '/media/oem/data/birds-eye-test-2-462562500hz-omni-omni-0.5w-150m.raw'
#data_file_loc2 = '/media/oem/data/birds-eye-test-2-462562500hz-omni-omni-0.5w-25m.raw'
#data_file_loc3 = '/media/oem/data/birds-eye-test-2-462562500hz-omni-omni-0.5w-50m.raw'
#data_file_loc4 = '/media/oem/data/birds-eye-test-2-462562500hz-omni-omni-0.5w-100m.raw'

# dtype_all= np.dtype([('raw-iq0', 'c16')]) # gr_complex is '32fc' --> make any sense?

# gr_complex is '32fc' --> make any sense?
dtype_all = scipy.dtype([('raw-iq', scipy.complex64)])

print('Total number of i/q samples for 0 deg:',
      scipy.fromfile(open(data_file_loc1), dtype=dtype_all).shape[0])
#
print('Total number of i/q samples for 45 deg:',
      scipy.fromfile(open(data_file_loc2), dtype=dtype_all).shape[0])
#
print('Total number of i/q samples for 90:',
      scipy.fromfile(open(data_file_loc3), dtype=dtype_all).shape[0])
#
print('Total number of i/q samples for 135:',
      scipy.fromfile(open(data_file_loc4), dtype=dtype_all).shape[0])

sample_size = 512  # CHANGE
no_of_samples = 4000  # CHANGE
iqdata_loc1 = scipy.fromfile(
    open(data_file_loc1), dtype=dtype_all, count=sample_size * no_of_samples)
iqdata_loc2 = scipy.fromfile(
    open(data_file_loc2), dtype=dtype_all, count=sample_size * no_of_samples)
iqdata_loc3 = scipy.fromfile(
    open(data_file_loc3), dtype=dtype_all, count=sample_size * no_of_samples)
iqdata_loc4 = scipy.fromfile(
    open(data_file_loc4), dtype=dtype_all, count=sample_size * no_of_samples)

# iqdata_loc1 = scipy.fromfile(open(data_file_loc1), dtype=dtype_all) # DATA COLLECTED at UCF
# iqdata_loc2 = scipy.fromfile(open(data_file_loc2), dtype=dtype_all) # DATA COLLECTED at UCF
# iqdata_loc3 = scipy.fromfile(open(data_file_loc3), dtype=dtype_all) # DATA COLLECTED at UCF
# iqdata_loc4 = scipy.fromfile(open(data_file_loc4), dtype=dtype_all) # DATA COLLECTED at UCF


start_time = time.time()  # Taking start time to calculate overall execution time
no_of_loc1 = iqdata_loc1.shape[0]
no_of_loc2 = iqdata_loc2.shape[0]
no_of_loc3 = iqdata_loc3.shape[0]
no_of_loc4 = iqdata_loc4.shape[0]

print('no_of_loc1:', no_of_loc1)

# USING ONLY LAST N SAMPLES OF DATA
number_of_data_to_read = sample_size * no_of_samples
extra_rows_loc1 = no_of_loc1 - number_of_data_to_read
extra_rows_loc2 = no_of_loc2 - number_of_data_to_read
extra_rows_loc3 = no_of_loc3 - number_of_data_to_read
extra_rows_loc4 = no_of_loc4 - number_of_data_to_read

print('number_of_data_to_read:', number_of_data_to_read)
print('extra_rows_loc1:', extra_rows_loc1)

xdata_loc1 = np.delete(iqdata_loc1, np.s_[:extra_rows_loc1], 0)
xdata_loc2 = np.delete(iqdata_loc2, np.s_[:extra_rows_loc2], 0)
xdata_loc3 = np.delete(iqdata_loc3, np.s_[:extra_rows_loc3], 0)
xdata_loc4 = np.delete(iqdata_loc4, np.s_[:extra_rows_loc4], 0)

# PREPARING THE DATA WITHOUT TIME INFORMATION
no_of_data_loc1 = iqdata_loc1.shape[0]
no_of_data_loc2 = iqdata_loc2.shape[0]
no_of_data_loc3 = iqdata_loc3.shape[0]
no_of_data_loc4 = iqdata_loc4.shape[0]


##################### CHANNELING REAL AND IMAGINARY PART OF XDATA ###########################

# xdata1 = np.dstack((xydata['raw-iq0'].real.reshape(no_of_data, 1), xydata['raw-iq0'].imag.reshape(no_of_data, 1)))
# for k in range(1, 1024):
#     st = "raw-iq" + str(k)
#     xdata_temp = np.dstack((xydata[st].real.reshape(no_of_data, 1), xydata[st].imag.reshape(no_of_data, 1)))
#     xdata1 = np.concatenate([xdata1, xdata_temp], axis=1)
# ydata1 = xydata['trans-id']
#
# xdata = xdata1.astype(np.float)
# ydata = ydata1.astype(np.int).flatten()
#
# print("UNTIL XDATA CHANNELING")

# THE WEIGHT MATRIX #################3
W = np.matrix([[np.cos(1*(np.pi/8)), np.sin(1*(np.pi/8))],
               [np.cos(2*(np.pi/8)), np.sin(2*(np.pi/8))],
               [np.cos(3*(np.pi/8)), np.sin(3*(np.pi/8))],
               [0, np.sin(4*(np.pi/8))],
               [np.cos(5*(np.pi/8)), np.sin(5*(np.pi/8))],
               [np.cos(6*(np.pi/8)), np.sin(6*(np.pi/8))],
               [np.cos(7*(np.pi/8)), np.sin(7*(np.pi/8))],
               [np.cos(8*(np.pi/8)), 0]]
              )

# W = np.matrix([[np.cos(4*(np.pi/8)), np.sin(4*(np.pi/8))],
# [np.cos(4*(np.pi/8)), np.sin(4*(np.pi/8))],
# [np.cos(4*(np.pi/8)), np.sin(4*(np.pi/8))],
# [np.cos(4*(np.pi/8)), np.sin(4*(np.pi/8))],
# [np.cos(0*(np.pi/8)), np.sin(0*(np.pi/8))],
# [np.cos(0*(np.pi/8)), np.sin(0*(np.pi/8))],
# [np.cos(0*(np.pi/8)), np.sin(0*(np.pi/8))],
# [np.cos(0*(np.pi/8)), np.sin(0*(np.pi/8))]]
# )

print(W)
print('Shape:', W.shape)

#######################################################################################

xdata_loc1 = np.concatenate([iqdata_loc1['raw-iq'].real.reshape(number_of_data_to_read, 1),
                            iqdata_loc1['raw-iq'].imag.reshape(number_of_data_to_read, 1)], axis=1)
print('Shape 0 Deg directional data after concatination:', xdata_loc1.shape)
xdata_loc1 = np.matmul(xdata_loc1, np.transpose(W))
print('Shape 0 Deg directional data after multiplication:', xdata_loc1.shape)

xdata_loc2 = np.concatenate([iqdata_loc2['raw-iq'].real.reshape(number_of_data_to_read, 1),
                            iqdata_loc2['raw-iq'].imag.reshape(number_of_data_to_read, 1)], axis=1)
print('Shape 45 Deg directional data  after concatination:', xdata_loc2.shape)
xdata_loc2 = np.matmul(xdata_loc2, np.transpose(W))
print('Shape 45 Deg directional data after multiplication:', xdata_loc2.shape)

xdata_loc3 = np.concatenate([iqdata_loc3['raw-iq'].real.reshape(number_of_data_to_read, 1),
                            iqdata_loc3['raw-iq'].imag.reshape(number_of_data_to_read, 1)], axis=1)
print('Shape 90 Deg directional data   after concatination:', xdata_loc3.shape)
xdata_loc3 = np.matmul(xdata_loc3, np.transpose(W))
print('Shape 90 Deg directional data after multiplication:', xdata_loc3.shape)

xdata_loc4 = np.concatenate([iqdata_loc4['raw-iq'].real.reshape(number_of_data_to_read, 1),
                            iqdata_loc4['raw-iq'].imag.reshape(number_of_data_to_read, 1)], axis=1)
print('Shape 135 Deg directional data   after concatination:', xdata_loc4.shape)
xdata_loc4 = np.matmul(xdata_loc4, np.transpose(W))
print('Shape 135 Deg directional data after multiplication:', xdata_loc4.shape)

# DOES TRANSPOSE MAKE SENSE????
print('Shape Before:::', xdata_loc1.shape)
xdata_loc1 = xdata_loc1.T.reshape(
    no_of_data_loc1//(sample_size), sample_size*8)  # CHNAGED FROM 2
print('Shape After', xdata_loc1.shape)

xdata_loc2 = xdata_loc2.T.reshape(
    no_of_data_loc2//(sample_size), sample_size*8)
xdata_loc3 = xdata_loc3.T.reshape(
    no_of_data_loc3//(sample_size), sample_size*8)
xdata_loc4 = xdata_loc4.T.reshape(
    no_of_data_loc4//(sample_size), sample_size*8)


xdata = np.concatenate(
    [xdata_loc1, xdata_loc2, xdata_loc3, xdata_loc4], axis=0)


# CREATING LABEL FOR THE DATASETS
ydata_loc1 = np.full(xdata_loc1.shape[0], 0, dtype=int)
ydata_loc2 = np.full(xdata_loc2.shape[0], 1, dtype=int)
ydata_loc3 = np.full(xdata_loc3.shape[0], 2, dtype=int)
ydata_loc4 = np.full(xdata_loc4.shape[0], 3, dtype=int)


ydata = np.concatenate(
    [ydata_loc1, ydata_loc2, ydata_loc3, ydata_loc4], axis=0)


# PREPROCESSING X AND Y DATA
xdata = xdata.astype(np.float)

ydata = ydata.astype(np.int).flatten()

# REMOVING THE NANS
xdata = np.nan_to_num(xdata)


# ############## RANDOMLY SHUFFLING THE DATA ###################
#
xydata = np.concatenate([xdata.reshape(
    xdata.shape[0], xdata.shape[1]), ydata.reshape(ydata.shape[0], 1)], axis=1)

np.random.shuffle(xydata)

print('Shape of XYDATA', xydata.shape)

# xdata, ydata = xydata[:,0:sample_size*2+2], xydata[:,((sample_size*2+2))]  # ADDED 2 FOR LAT LONG

# multiplied by 8 because we augmented with weight matrix
xdata, ydata = xydata[:, 0:sample_size*8], xydata[:, ((sample_size*8))]


#################### NORMALIZE THE X DATA #######################


# Normalize the data with zero mean and unit variance for each column
standard = preprocessing.StandardScaler().fit(xdata)
xdata = standard.transform(xdata)


############### SEPARATING TRAIN AND TEST DATA #######################

xtrain, xtest, ytrain, ytest = train_test_split(
    xdata, ydata, test_size=0.2, random_state=42)  # 90/20 is train/test size
print('XTRAIN AND XTEST SHAPE:', xtrain.shape, xtest.shape)
print('YTRAIN AND YTEST SHAPE:', ytrain.shape, ytest.shape)

# reshape to be [samples][width][height][channels]
xtrain = xtrain.reshape((xtrain.shape[0], 8, sample_size, 1)).astype('float32')
xtest = xtest.reshape((xtest.shape[0], 8, sample_size, 1)).astype('float32')


num_classes = 4  # TOTAL NUMBER OF Data

# Convert labels to categorical one-hot encoding
# DEFINE THE NUMBER OF TOTAL CLASSES IN LABEL
ytrain_one_hot = to_categorical(ytrain, num_classes=num_classes)
ytest_one_hot = to_categorical(ytest, num_classes=num_classes)


print('XTRAIN AND XTEST SHAPE:', xtrain.shape, xtest.shape)
print('YTRAIN AND YTEST SHAPE:', ytrain_one_hot.shape, ytest_one_hot.shape)

############################################################
#                                                          #
########    Building a Convolutional Neural Network #################
#							   #
############################################################

dr = 0.6  # dropout rate (%)
batch_size = 64  # Mini batch size
nb_epoch = 150  # Number of Epoch (Give a higher number to get better accuracy)
# classes = array("i", [0, 1]) # CHANGE: LABEL
# classes = ["T1","T2"]
classes = ['0', '45', '90', '135']  # CHANGE LABEL
in_shp = list(xtrain.shape[1:])  # Input Dimension
print(in_shp)
# model = models.Sequential()
timesteps = 1
data_dim = xtrain.shape[1]

###############################
# NEXT CHANGE: MINIMIZE THE KERNEL SIZE AND STRIDES
# THEN: CHANGE THE ACTIVATIONS OF THE LAYERS
##############################

############################################################
#                                                          #
########  Building a 2D Convolutional Neural Network   #####
#						                            	   #
############################################################

# xtrain = xtrain.reshape(xtrain.shape[0], 1, xtrain.shape[1])
# xtest = xtest.reshape(xtest.shape[0], 1, xtest.shape[1])


# print ("AFTER RESHAPE")
ytrain_one_hot = np.reshape(
    ytrain_one_hot, (ytrain_one_hot.shape[0], num_classes))  # Used in training
ytest_one_hot = np.reshape(
    ytest_one_hot, (ytest_one_hot.shape[0], num_classes))  # Used in training

# Modeling the CNN
model = Sequential()

model.add(Conv2D(64, (2, 2), input_shape=(8, sample_size, 1),
          activation='relu'))  # CHANGE # Stride (1, 1)
model.add(MaxPooling2D())  # Pool size: (2, 2) and stride (2, 2)
model.add(Dropout(0.2))
model.add(Conv2D(64, (2, 2), activation='relu'))
model.add(MaxPooling2D())
model.add(Dropout(dr))
model.add(Flatten())
#model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

# Compile model
# For a multi-class classification problem
sgd = SGD(lr=0.0001, decay=1e-6, momentum=0.9, nesterov=True)
#adam = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-8)
# model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy']) # Multiclass classification with rmsprop

model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=[
              'acc', f1_m, precision_m, recall_m])  # Multiclass classification with rms adam optimizer # CHANGE

#model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc', f1_m, precision_m, recall_m])

model.summary()
filepath = '/home/oem/birds-eye2/DeePOE-mod/direction_data_4_direction_2D_CNN_Mapping_Outdoor.wts.h5'
print('The dropout rate was: ')
print(dr)


# Fit the model
# history= model.fit(xtrain, ytrain_one_hot, epochs=nb_epoch, batch_size=batch_size, validation_data = (xtest, ytest_one_hot), callbacks = [keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=2, save_best_only=True, mode='auto'), keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, verbose=2, mode='auto')])
history = model.fit(xtrain, ytrain_one_hot, epochs=nb_epoch, batch_size=batch_size, validation_split=0.1, callbacks=[
    keras.callbacks.ModelCheckpoint(
        filepath, monitor='val_loss', verbose=2, save_best_only=True, mode='auto'),
    keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, verbose=2, mode='auto')])

# Evaluate the model
loss, accuracy, f1_score, precision, recall = model.evaluate(
    xtest, ytest_one_hot, batch_size=batch_size)  # CHANGE
print('\nTest Loss: %s: %.2f%%' %
      (model.metrics_names[0], loss * 100))  # CHANGE
print('\nTest Accuracy: %s: %.2f%%' %
      (model.metrics_names[1], accuracy * 100))  # CHANGE
print('\nTest F1 Score: %s: %.2f%%' %
      (model.metrics_names[2], f1_score))  # CHANGE
print('\nTest Precision: %s: %.2f%%' %
      (model.metrics_names[3], precision * 100))  # CHANGE
print('\nTest Recall: %s: %.2f%%' %
      (model.metrics_names[4], recall * 100))  # CHANGE

# Calculating total execution time
end_time = time.time()  # Taking end time to calculate overall execution time
print('\n Total Execution Time (Minutes): ')
print(((end_time - start_time) / 60))

#### SET PLOTTING PARAMETERS #########
params = {'legend.fontsize': 'xx-large',
          'axes.labelsize': 'xx-large',
          'axes.titlesize': 'xx-large',
          'xtick.labelsize': 'xx-large',
          'ytick.labelsize': 'xx-large'}
plt.rcParams.update(params)


# Show Accuracy Curves
fig = plt.figure()
# plt.title('Training Performance')
plt.plot(history.epoch, history.history['acc'],
         label='Training Accuracy', linewidth=2.0, c='b')
plt.plot(history.epoch, history.history['val_acc'],
         label='Validation Accuracy', linewidth=2.0, c='r')
plt.ylabel('Accuracy(%)')
plt.xlabel('Epoch')
plt.legend()
# fig = plt.subplots(nrows=1, ncols=1)  # create figure & 1 axis
# ax.plot([0, 1, 2], [1UCF0, 20, 3])
plt.tight_layout()
# save the figure to file
fig.savefig(
    '/home/oem/birds-eye2/DeePOE-mod/direction_4_acc_2D_CNN_Mapping_Outdoor.png')
plt.close(fig)


# plt.show()


def plot_confusion_matrix(cm, title='Confusion Matrix', cmap=plt.cm.YlGnBu, labels=[], normalize=False):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    # print("Normalized confusion matrix")
    else:
        cm = cm.astype('int')
    # print('Confusion matrix, without normalization')
    plt.rcParams.update(params)  # ADDED
    fig = plt.figure(figsize=(12, 12))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    # plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=45)
    plt.yticks(tick_marks, labels)
    thresh = cm.max() / 2
    fmt = '.2f' if normalize else 'd'
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        # plt.text(j, i,"{:,}".format(cm[i, j]),
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment='center', fontsize='xx-large',
                 color='white' if cm[i, j] > thresh else 'black')

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    # fig, ax = plt.subplots(nrows=1, ncols=1)  # create figure & 1 axis
    # ax.plot([0, 1, 2], [10, 20, 3])
    plt.tight_layout()
    # save the figure to file
    fig.savefig(
        '/home/oem/birds-eye2/DeePOE-mod/direction_4_conf_mat_2D_CNN_Mapping_Outdoor.png')
    plt.close(fig)


# plt.show()


# Plot confusion matrix
test_Y_hat = model.predict(xtest, batch_size=batch_size)
conf = np.zeros([len(classes), len(classes)])
confnorm = np.zeros([len(classes), len(classes)])
for i in range(0, xtest.shape[0]):
    j = list(ytest_one_hot[i, :]).index(1)
    k = int(np.argmax(test_Y_hat[i, :]))
    conf[j, k] = conf[j, k] + 1
plot_confusion_matrix(conf, labels=classes, normalize=False)
for i in range(0, len(classes)):
    confnorm[i, :] = conf[i, :] / np.sum(conf[i, :])
# plot_confusion_matrix(confnorm, labels=classes)

end_time = time.time()  # Taking end time to calculate overall execution time
print('\n Total Execution Time (Minutes): ')
print(((end_time-start_time)/60))
