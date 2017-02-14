# Create first network with Keras
from keras.models import Sequential
from keras.layers import Dense, Reshape, Activation, Dropout,Layer, LocallyConnected1D, LocallyConnected2D, Convolution1D, GlobalMaxPooling1D, Flatten, MaxPooling1D, MaxPooling2D
from keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler, ReduceLROnPlateau
from keras.optimizers import SGD
from keras.optimizers import RMSprop, Adamax
from keras import backend as K
from theano import tensor as T
import numpy as np
import pandas
from keras.regularizers import l1l2
from keras.utils import np_utils
K.set_image_dim_ordering('th')



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
    
L=WinnerTakeAll1D_GaborMellis(spatial=1, OneOnX=3)


# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)

# Function to create model, required for KerasClassifier
def create_model(neurons = 50):
	# create model
	model = Sequential()
	model.add(Dense(neurons, input_dim=30, init='normal', activation='relu' ,W_regularizer=l1l2(l1=0.000005, l2=0.00005))) #W_regularizer=l1(0.000001), activity_regularizer=activity_l1(0.000001)))
	model.add(L)
	model.add(Dense(120, activation ='relu'))
	model.add(L)
	model.add(Dense(120, activation ='relu'))
	model.add(L)
	model.add(Dense(2))
	model.add(Activation('softmax'))
	model.compile(optimizer='Adamax', loss='binary_crossentropy', metrics=['sparse_categorical_accuracy']) # Gradient descent

	return model

# load dataset
dataframe = pandas.read_csv("training.csv", header=None)
dataset = dataframe.values

X = dataset[0:230000:,1:31].astype(float)
Y = dataset[0:230000:,32].astype(float)

X -= np.mean(X, axis = 0) # center
X /= np.std(X, axis = 0) # normalize

Y = np_utils.to_categorical(Y, 2) # convert class vectors to binary class matrices

# create model
model = KerasClassifier(build_fn=create_model, nb_epoch=40, batch_size=96, verbose=0)

# define the grid search parameters
neurons = [50, 70, 90, 120, 150, 200, 250, 300]
param_grid = dict(neurons=neurons)
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=1)
grid_result = grid.fit(X, Y)
# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))