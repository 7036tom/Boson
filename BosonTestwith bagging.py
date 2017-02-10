# Create first network with Keras
from keras.models import Sequential
from keras.layers import Dense, Reshape, Activation, Dropout,Layer, LocallyConnected1D, LocallyConnected2D, Convolution1D, GlobalMaxPooling1D, Flatten, MaxPooling1D, MaxPooling2D
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import SGD
from keras.optimizers import RMSprop, Adamax
import math
from math import log

import csv

from keras import backend as K
from theano import tensor as T

from sklearn.model_selection import StratifiedKFold

import numpy as np
import pandas
from keras.regularizers import l1, activity_l1, l1l2
# fix random seed for reproducibility
seed = 7
np.random.seed(seed)

from keras.utils import np_utils
from keras.layers.convolutional import Convolution2D
K.set_image_dim_ordering('th')

class WinnerTakeAll1D_GaborMellis(Layer):

    def __init__(self, spatial=1, OneOnX = 3,**kwargs):
        self.spatial = spatial
        self.OneOnX = OneOnX
        self.uses_learning_phase = True
        super(WinnerTakeAll1D_GaborMellis, self).__init__(**kwargs)

    def call(self, x, mask=None):
        R = T.reshape(x,(T.shape(x)[0],T.shape(x)[1]/self.OneOnX,self.OneOnX))
        M = K.max(R, axis=(2), keepdims=True)
        R = K.switch(K.equal(R, M), R, 0.)
        R = T.reshape(R,(T.shape(x)[0],T.shape(x)[1]))
        return R

    def get_output_shape_for(self, input_shape):
        shape = list(input_shape)
        return tuple(shape)
    
L=WinnerTakeAll1D_GaborMellis(spatial=1, OneOnX=3)

# load dataset
dataframe = pandas.read_csv("training.csv", header=None)
dataset = dataframe.values

dataframe2 = pandas.read_csv("test.csv", header=None)
dataset2 = dataframe2.values

# Let's move all the azymuth angle features at the end of X (12, 16, 19, 21, 26, 29)
# 12 <-> 28 PhiCentrality
dataset[:,[12,28]]
#dataset2[:,[12,28]]
# 16 <-> 27 Phitau
dataset[:,[16,26]]
#dataset2[:,[16,26]]
# 19 <-> 25 Philep
dataset[:,[19,25]]
#dataset2[:,[19,25]]
# 21 <-> 24 Phimet
dataset[:,[21,24]]
#dataset2[:,[21,24]]
# 26 Phijetleading
# 29 Phijetsubleading


# split into input (X) and output (Y) variables
W = dataset[0:10000:,31:32]
X = dataset[0:10000:,1:31].astype(float)
Y = dataset[0:10000:,32].astype(float)
X2 = dataset[0:230000:,1:31].astype(float)
Y2 = dataset[0:230000:,32].astype(float)

Z2 = dataset2[0:550000,1:31].astype(float)

nbr_signals = 0;
nbr_backgrounds =0;
for i in range(230000):
	if (Y2[i] == 1):
		nbr_signals = nbr_signals + 1
	else:
		nbr_backgrounds = nbr_backgrounds + 1






# Implementation of advanced features in x and x2
# Notes : minv(tau,lep) (3) | 

"""
for i in range(9999):
	# Implementation of Phi derived features
	X[i,24] = min(dataset[i,26]-dataset[i,25],dataset[i,26]-dataset[i,24],dataset[i,25]-dataset[i,24])
	X[i,25] = min(dataset[i,26]-dataset[i,24],dataset[i,25]-dataset[i,24])
	X[i,26] = min(dataset[i,26]-dataset[i,25],dataset[i,26]-dataset[i,24])
	X[i,27] = dataset[i,25]-dataset[i,24] # Erreur ?
	# Implementation of mass based features
	#print(dataset[1, 3])
	#x[i,4] = log(1+dataset[i, 3])
	


for i in range(220000):
	# Implementation of Phi derived features
	X2[i,24] = min(dataset[i,26]-dataset[i,25],dataset[i,26]-dataset[i,24],dataset[i,25]-dataset[i,24])
	X2[i,25] = min(dataset[i,26]-dataset[i,24],dataset[i,25]-dataset[i,24])
	X2[i,26] = min(dataset[i,26]-dataset[i,25],dataset[i,26]-dataset[i,24])
	X2[i,27] = dataset[i,25]-dataset[i,24] # Erreur ?

	# Implementation of mass based features
	#x2[i,4] = log(1)#+dataset[i, 3])

for i in range(550000):
	# Implementation of Phi derived features
	Z2[i,24] = min(dataset2[i,26]-dataset2[i,25],dataset2[i,26]-dataset2[i,24],dataset2[i,25]-dataset2[i,24])
	Z2[i,25] = min(dataset2[i,26]-dataset2[i,24],dataset2[i,25]-dataset2[i,24])
	Z2[i,26] = min(dataset2[i,26]-dataset2[i,25],dataset2[i,26]-dataset2[i,24])
	Z2[i,27] = dataset2[i,25]-dataset2[i,24] # Erreur ?

	# Implementation of mass based features
	#x2[i,4] = log(1)#+dataset[i, 3])
"""


X -= np.mean(X, axis = 0) # center
X2 -= np.mean(X2, axis = 0) # center
Z2 -= np.mean(Z2, axis = 0) # center


#Normalisation of inputs.

X /= np.std(X, axis = 0) # normalize
X2 /= np.std(X2, axis = 0) # normalize
Z2 /= np.std(Z2, axis = 0) # normalize

"""
#In case of locallyConnected.
X = X.reshape(X.shape[0], 1, 7, 4).astype('float32')
X2 = X2.reshape(X2.shape[0],1, 7, 4).astype('float32')
Z2 = Z2.reshape(Z2.shape[0],1, 7, 4).astype('float32')
"""


Y = np_utils.to_categorical(Y, 2) # convert class vectors to binary class matrices
#Y2 = np_utils.to_categorical(Y2, 2)


#input_shape=(28,1)
model = [Sequential(), Sequential(),Sequential(), Sequential(), Sequential()]

#rms = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0001)
admax = Adamax(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0001)

# Weights more adapted to the class imbalanced of the issue.
class_weight = {0 : 3,
    1: 1}


# define 10-fold cross validation test harness
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
cvscores = []

j = 0

for i in range(1):
	for train, test in kfold.split(X2, Y2):
		print(j) 
		print("ieme fold")
		# create model
		model[j].add(Dense(120, input_dim=30, init='normal', activation='relu' ,W_regularizer=l1l2(l1=0.000005, l2=0.00005))) #W_regularizer=l1(0.000001), activity_regularizer=activity_l1(0.000001)))
		model[j].add(L)
		model[j].add(Dense(120, activation ='relu'))
		model[j].add(L)
		model[j].add(Dense(120, activation ='relu'))
		model[j].add(L)
		model[j].add(Dense(2))
		model[j].add(Activation('softmax'))

		model[j].compile(optimizer=admax, loss='binary_crossentropy', metrics=['sparse_categorical_accuracy']) # Gradient descent
		#model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy']) # Gradient descent
		#model[j].compile(optimizer='adam', loss='categorical_crossentropy', metrics=[''sparse_categorical_accuracy'']) # Gradient descent

		print (model[j].summary()) # Affiche les details du reseau !

		Y3 = np_utils.to_categorical(Y2, 2) # convert class vectors to binary class matrices
		W2 =  np_utils.to_categorical(Y2[test], 2)
	
		
		# Early stopping.
		
		callbacks = [
    	EarlyStopping(monitor='val_loss', patience=10, verbose=0),
    	ModelCheckpoint("/home/tom/Documents/Projets python/bosongit/weigh.hdf", monitor='val_loss', save_best_only=True, verbose=0),
		]
		
		
		# Fit the model		
		model[j].fit(X2[train], Y3[train],validation_data=(X2[test], Y3[test]), nb_epoch=40, batch_size=96, class_weight=class_weight, shuffle=True, verbose=1, callbacks=callbacks)#, class_weight=class_weight)

		#model.fit(X2[train], Y3[train], nb_epoch=200, batch_size=96)

		# evaluate the model
		scores = model[0].evaluate(X2[test], Y3[test])
		print("%s: %.2f%%" % (model[0].metrics_names[1], scores[1]*100))
		cvscores.append(scores[1] * 100)
		j = j +1;

print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))


"""
model.fit(X2, Y2, validation_data=(X, Y), nb_epoch=200, batch_size=96)
# Final evaluation of the model
scores = model.evaluate(X, Y, verbose=0)
print("Baseline Error: %.2f%%" % (100-scores[1]*100))
"""


#print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

#test

#scores = model.evaluate(X, Y)
#print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

"""
Z = model.predict(X, batch_size=32, verbose=0)

s = 0
b = 0

for i in range(0,9999):
	if (Z[i][1]>Z[i][0]):
		if (Y[i][1]>Y[i][0]):
			s = s + W[i][0]
		if (Y[i][1]<=Y[i][0]):
			b = b + W[i][0]

br = 10.0
radicand = 2 *( (s+b+br) * math.log(1.0 + s/(b+br)) - s)
AMS = math.sqrt(radicand)


print("AMS : ")
print(AMS)
"""

Z0_restult = model[0].predict(Z2, batch_size=32, verbose=0)
Z1_restult = model[1].predict(Z2, batch_size=32, verbose=0)
Z2_restult = model[2].predict(Z2, batch_size=32, verbose=0)
Z3_restult = model[3].predict(Z2, batch_size=32, verbose=0)
Z4_restult = model[4].predict(Z2, batch_size=32, verbose=0)


c = csv.writer(open("Submission.csv", "wb"))
c.writerow(["EventId","RankOrder","Class"])
for i in range(550000):
	if (Z0_restult[i][1]+Z1_restult[i][1]+Z2_restult[i][1]+Z3_restult[i][1]+Z4_restult[i][1]> Z0_restult[i][0]+Z1_restult[i][0]+Z2_restult[i][0]+Z3_restult[i][0]+Z4_restult[i][0]):
		c.writerow([350000+i,i+1,'s' ])
	if (Z0_restult[i][1]+Z1_restult[i][1]+Z2_restult[i][1]+Z3_restult[i][1]+Z4_restult[i][1]<= Z0_restult[i][0]+Z1_restult[i][0]+Z2_restult[i][0]+Z3_restult[i][0]+Z4_restult[i][0]):
		c.writerow([i+350000,i+1,'b' ])

#----------------------------------------------------------------------------
# TO DO
# CV bagging with 10 repetitions DONE
# Momentum DONE
# Learning rate DONE
# L1 penalty DONE
# L2 penalty DONE
# AMS metric DONE
# Advanced features 4/10 (mail)
# Normalization DONE
# Elimination of Azymuth angles features DONE
# Winner takes all activation DONE 
# Constrain neurons in first layer
#
#
#
#----------------------------------------------------------------------------
