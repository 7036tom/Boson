# Create first network with Keras
from keras.models import Sequential
from keras.layers import Dense, Reshape, Activation, Dropout,Layer
from keras.optimizers import Adamax
from keras import backend as K
from theano import tensor as T
import numpy as np
import pandas
import math
from keras.regularizers import l1l2
from keras.utils import np_utils
from sklearn.metrics import cohen_kappa_score, make_scorer


# Use scikit-learn to grid search the batch size and epochs
import numpy
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier


# WTAA
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


def AMS(estimator, y_true, y_probs):
	
	Z = estimator.predict(y_true, batch_size=32, verbose=0)
	
	Y = y_probs

	s = 0
	b = 0
	
	print(X[0][0])
	print(X[1][0])
	print(y_true[0][0])
	print(y_true[1][0])
	print(X[250000-166666][0])
	print(X[250000-166666+1][0])


	
	if (y_true[0][0]==X[0][0]):
		Wp=W[:len(y_true)]	
		
	elif(y_true[0][0]==X[250000-166666][0]):
		Wp=W[250000-166666:]	
		
	else:
		Wp=W[250000-len(y_true):]
		
		
	for i in range(0,len(y_true)):
		if (Z[i]==1):
			if (Y[i][0] <= Y[i][1]):
				s = s + Wp[i][0]
			if (Y[i][0] > Y[i][1]):
				b = b + Wp[i][0]
	
	br = 10.0
	radicand = 2 *( (s+b+br) * math.log(1.0 + s/(b+br)) - s)

	AMS = math.sqrt(radicand)
	print("ams : "+str(AMS))
	return AMS


# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)

# Function to create model, required for KerasClassifier

def create_model(decay):
    # create model
    #L=WinnerTakeAll1D_GaborMellis(spatial=1, OneOnX=WTAX)
    model = Sequential()
    


    model.add(Dense(90, input_dim=30, init='normal', activation='relu' ,W_regularizer=l1l2(l1=9E-7, l2=5e-07))) #W_regularizer=l1(0.000001), activity_regularizer=activity_l1(0.000001)))
    #model.add(L)
    model.add(Dense(130, activation ='relu'))
    #model.add(L)
    model.add(Dense(85, activation ='relu'))
    #model.add(L)
    model.add(Dense(2))
    model.add(Activation('softmax'))
    admax = Adamax(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=decay)
    model.compile(optimizer=admax, loss='binary_crossentropy', metrics=['accuracy']) # Gradient descent
    return model


# Weights more adapted to the imbalances of the classes.
class_weight = {0 : 3,
    1: 1}


# load dataset
dataframe = pandas.read_csv("training.csv", header=None)
dataset = dataframe.values

X = dataset[0:250000:,1:31].astype('float32')
Y = dataset[0:250000:,32].astype('float32')
W = dataset[0:250000:,31:32]

for i in range(250000):
	if (X[i][0] - 0.162894 < 0.0001):
		print(i)
		break

X -= np.mean(X, axis = 0) # center
X /= np.std(X, axis = 0) # normalize

Y = np_utils.to_categorical(Y, 2) # convert class vectors to binary class matrices

# create model
model = KerasClassifier(build_fn=create_model, nb_epoch=10, batch_size=80, verbose=1)


kappa_scorer = make_scorer(cohen_kappa_score)
# define the grid search parameters
neurons1 = [200,220,240]
neurons2 = [200, 275, 350]
neurons3 = [275, 250, 300]
batch_size = [80,83,86,89,92,95,98, 100]
epochs = [100, 120, 140, 160, 180, 200]
WTAX=[3,4,5]
l1_value = [0.0000005, 0.0000003, 0.0000007, 0.0000009, 0.0000001]
l2_value = [0.0000005, 0.0000003, 0.0000007, 0.0000009, 0.0000001]
l_rate = [0.001, 0.002, 0.003 ,0.004]
decay = [0.01, 0.001, 0.0001, 0.00001, 0.000001, 0]

param_grid = dict(decay=decay)
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, scoring=AMS, cv =None)#, verbose=1)

grid_result = grid.fit(X, Y)
# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))
