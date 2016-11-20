# adapted from keras examples
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Reshape
from keras.layers import Convolution2D, MaxPooling2D, AveragePooling2D
from keras.optimizers import SGD
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2, activity_l2
from keras.layers.core import Activation

MAPS_COUNT_PARAM=128 # a higher value like 96 ends up with a Nan loss on GPU
#LAMBDA_PARAM=0.0002 
# init='he_normal'
DROPOUT_VALUE=0.5

def make_model(batch_size,input_size):

	''' define the model'''
	model = Sequential()

	model.add(Convolution2D(MAPS_COUNT_PARAM, 3, 3, border_mode='same', \
		      batch_input_shape=(batch_size, 3, input_size, input_size),\
		      init='he_normal'))
	model.add(Dropout(DROPOUT_VALUE))

	model.add(Activation('relu'))
	model.add(Convolution2D(MAPS_COUNT_PARAM  , 3, 3, border_mode='same', init='he_normal',subsample=(4,4)))
	model.add(Dropout(DROPOUT_VALUE))
	model.add(Activation('relu'))
	model.add(Convolution2D(MAPS_COUNT_PARAM, 3, 3, border_mode='same', init='he_normal'))
	model.add(Dropout(DROPOUT_VALUE))
	model.add(Activation('relu'))
	#model.add(BatchNormalization(mode=1))

	model.add(Convolution2D(MAPS_COUNT_PARAM*2, 3, 3, border_mode='same', init='he_normal',subsample=(2,2)))
	model.add(Dropout(DROPOUT_VALUE))
	model.add(Activation('relu'))
	model.add(Convolution2D(MAPS_COUNT_PARAM*2, 3, 3, border_mode='same', init='he_normal'))
	model.add(Dropout(DROPOUT_VALUE))
	model.add(Activation('relu'))
	model.add(Convolution2D(MAPS_COUNT_PARAM*2, 3, 3, border_mode='same', init='he_normal',subsample=(2,2)))
	model.add(Dropout(DROPOUT_VALUE))
	model.add(Activation('relu'))
	#model.add(BatchNormalization(mode=1))

	model.add(Convolution2D(MAPS_COUNT_PARAM*3, 3, 3, border_mode='same', init='he_normal'))
	model.add(Dropout(DROPOUT_VALUE))
	model.add(Activation('relu'))
	model.add(Convolution2D(MAPS_COUNT_PARAM*3, 3, 3, border_mode='same', init='he_normal'))
	model.add(Dropout(DROPOUT_VALUE))
	model.add(Activation('relu'))
	model.add(Convolution2D(MAPS_COUNT_PARAM*3, 3, 3, border_mode='same', init='he_normal',subsample=(2,2)))
	model.add(Dropout(DROPOUT_VALUE))
	model.add(Activation('relu'))
#	model.add(BatchNormalization(mode=0,axis=1))

	model.add(Convolution2D(MAPS_COUNT_PARAM*4, 3, 3, border_mode='same', init='he_normal'))
	model.add(Dropout(DROPOUT_VALUE))
	model.add(Activation('relu'))
	model.add(Convolution2D(MAPS_COUNT_PARAM*4, 3, 3, border_mode='same', init='he_normal'))
	model.add(Dropout(DROPOUT_VALUE))
	model.add(Activation('relu'))
	model.add(Convolution2D(MAPS_COUNT_PARAM*4, 3, 3, border_mode='same', init='he_normal',subsample=(2,2)))
	model.add(Dropout(DROPOUT_VALUE))
	model.add(Activation('relu'))
	#model.add(BatchNormalization(mode=1))	

	model.add(Convolution2D(MAPS_COUNT_PARAM*4, 3, 3, border_mode='same', init='he_normal'))
	model.add(Dropout(DROPOUT_VALUE))
	model.add(Activation('relu'))
	model.add(Convolution2D(MAPS_COUNT_PARAM*4, 3, 3, border_mode='same', init='he_normal'))
	model.add(Dropout(DROPOUT_VALUE))
	model.add(Activation('relu'))
	model.add(AveragePooling2D(pool_size=(4, 4)))
	#model.add(BatchNormalization(mode=1))	
	#print('model characteristics:',model.summary())

	model.add(Flatten())

	model.add(Dense(1000))
	model.add(Activation('softmax'))

	#print('model characteristics:',model.summary())

	return model
