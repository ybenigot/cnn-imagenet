from keras.optimizers import SGD, Nadam
from keras.datasets import cifar10
from keras.utils.np_utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l2, activity_l2
from keras.callbacks import EarlyStopping,ModelCheckpoint

from trace import TraceWeights

import numpy as np
import sys
import imageutils as im
import time
import random as rn
import datetime as dt
import os
import plot as pl
import modelallcnn as mo
import socket

from sklearn.utils import shuffle # used for shuffling batch of samples
from scipy import ndimage as nd # used for image rotation

from imagenet_read import Reader

############### Parameters ################
IMAGE_SIZE=256
IMAGE_BORDER=0#32
INPUT_SIZE=IMAGE_SIZE+2*IMAGE_BORDER
CATEGORIES_COUNT=1000
BATCH_SIZE=48#16
LEARN_RATE=0.0005
#DECAY=0.0001
IMAGES_FOR_MEAN=200
TRAINING_SET_RATIO=0.9

############## Data load parameters #########
WEIGHTS_DIR='../../Documents/record.keras.weights'

if socket.gethostname()[0:7] == 'yvesMBP':  # only for testing
	#Mac HD test :
	ISMAC=True
	ROOT_FILE_NAME_TRAIN='/Users/yves/Documents/datasets/imagenet/test/train/image'
	ROOT_FILE_NAME_VAL='/Users/yves/Documents/datasets/imagenet/test/val/image'
	ROOT_FILE_NAME_TEST='/Users/yves/Documents/datasets/imagenet/test/test/image'
else:	#Ubuntu :
	ISMAC=False

	if socket.gethostname()[0:7] == 'yves-ml':  # only for testing
		ROOT_FILE_NAME_TRAIN = '/home/yves/Dropbox/imagenet/n01532829/image'
		ROOT_FILE_NAME_VAL   = '/home/yves/Dropbox/imagenet/n01532829/image'
		ROOT_FILE_NAME_TEST  = '/home/yves/Dropbox/imagenet/n01532829/image'

	else:  # the real dataset
		ROOT_FILE_NAME_TRAIN='/media/yves/sandisk3/datasets/imagenet/train/image'
		ROOT_FILE_NAME_VAL='/media/yves/sandisk3/datasets/imagenet/val/image'
		ROOT_FILE_NAME_TEST='/media/yves/sandisk3/datasets/imagenet/test/image'

FILE_COUNT_TRAIN=80
FILE_COUNT_VAL=100
FILE_COUNT_TEST=20
TRAIN_SAMPLES_COUNT=1025176
VAL_SAMPLES_COUNT=50000
TEST_SAMPLES_COUNT=255991

DISPLAY_IMAGES=True

class PreProcessor:

	def set_readers(self, reader, reader_val, reader_test):
		self.reader = reader
		self.reader_val = reader_val
		self.reader_test = reader_test

	def get_readers(self):
		return self.reader,self.reader_val,self.reader_test

	def compute_stats(self):
		''' compute mean per color channel for all data '''

		print("compute image statistics...")
		gen = self.reader.read_zipfile(ROOT_FILE_NAME_TRAIN,FILE_COUNT_TRAIN,IMAGE_SIZE, IMAGES_FOR_MEAN,None,None)
		data,y = next(gen)
		self.reader.stop_read()
		print("...done")

		m = np.zeros(data.shape[1])
		s = np.zeros(data.shape[1])
		for j in range(0,data.shape[1]):
			m[j] = np.mean(data[:,j,:])
			s[j] = np.std(data[:,j,:])
		self.avg=m
		self.std=s	
		return m,s

	def process_data_batch(self,X,y):
		''' preprocess a batch of data in memory, same algorithm for train, validation, and test data '''
		''' note : the same batch of data is "re augmented" again for each pass '''
		#processing is done in reader
		y_cat = to_categorical(y,CATEGORIES_COUNT)
		return X,y_cat,y

class Engine:

	X_batch_current=0
	y_batch_current=0
	y_batch_current_id=0

	def __init__(self,preprocessor):
		self.preprocessor = preprocessor

	def dataGenerator(self,reader,root_file_name,file_count,batch_size,avg,std):
		''' a python 3 generator for producing batches of data '''
		print("new generator for batches of size %s \n" % (batch_size,) )
		g = reader.read_zipfile(root_file_name, file_count, IMAGE_SIZE, batch_size,avg,std)
		count = 0
		while(True):
			X_batch, y_batch = next(g)
			if X_batch is None:  # restart reader if all samples already read
				g = reader.read_zipfile(root_file_name, file_count, IMAGE_SIZE, batch_size, avg, std)
				X_batch, y_batch = next(g)
			self.X_batch_current,self.y_batch_current,self.y_batch_current_id = self.preprocessor.process_data_batch(X_batch, y_batch)
			count += 1
			if count % 30000 == 0 and DISPLAY_IMAGES:
				im.display_image_set(self.X_batch_current,self.y_batch_current_id,\
								  reader,INPUT_SIZE,self.preprocessor.avg,self.preprocessor.std,ISMAC)
			yield self.X_batch_current,self.y_batch_current

	def fit(self,model , epochs,avg,std):
		''' train the model '''
		earlyStopping=EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='auto')
		checkpointer = ModelCheckpoint(filepath=WEIGHTS_DIR+"/weights.{epoch:02d}-{val_loss:.2f}.hdf", verbose=1, save_best_only=True)
		traceWeightsTrain=TraceWeights(1,self,100)
		reader,reader_val,reader_test = self.preprocessor.get_readers()
		g_train=self.dataGenerator(reader,   ROOT_FILE_NAME_TRAIN, FILE_COUNT_TRAIN, BATCH_SIZE, avg, std)
		g_valid=self.dataGenerator(reader_val, ROOT_FILE_NAME_VAL, FILE_COUNT_VAL,   BATCH_SIZE, avg, std)
		history=model.fit_generator(g_train,callbacks=[checkpointer,\
			#earlyStopping,\
			traceWeightsTrain,\
			],\
			samples_per_epoch=TRAIN_SAMPLES_COUNT,\
			nb_epoch=epochs,verbose=1,\
			validation_data=g_valid,\
			nb_val_samples=VAL_SAMPLES_COUNT)
		return history

	@staticmethod
	def predict(self,model):
		''' predict Y given X using model '''
		reader,reader_val,reader_test = self.preprocessor.get_readers()
		g_test=self.dataGenerator(reader_test, ROOT_FILE_NAME_TEST, FILE_COUNT_TEST, BATCH_SIZE, avg, std)
		pred = predict_generator(self, g_test, val_samples)
		return pred

	@staticmethod
	def compute_accuracy(pred,Y):
		'''compute prediction accuracy by matching pred and Y'''
		comparison = np.argmax(pred,1)==np.argmax(Y,1)
		accuracy = sum(comparison)/pred.shape[0]
		return accuracy

def show_images(archives_root,archive_count,s,pre):
	print("prepare initial display images for : "+s)
	reader= Reader(IMAGE_SIZE,IMAGE_BORDER)
	gen = reader.read_zipfile(archives_root,archive_count,IMAGE_SIZE, 16, pre.avg, pre.std)
	X_batch,y_batch = next(gen)
	reader.stop_read()
	print("...done(2)")
	# display some input data
	if DISPLAY_IMAGES:
		X1,y1,y_id = pre.process_data_batch(X_batch,y_batch)
		#print("now displaying images...",X1[0:2,0:2,0:2,0:2])
		im.display_image_set(X1,y_id,reader,INPUT_SIZE,pre.avg,pre.std,ISMAC)

def show_results(pred,X,Y):
	classification=np.argmax(pred,1)	
	for i in rn.sample(range(X.shape[0]), 1):
		im.display_normalized_image(X[i,:],INPUT_SIZE)
		#print('prediction:',cl.labels[classification[i]],'actual value:',cl.labels[np.argmax(Y[i])])
		time.sleep(5)

def main():

	epochs=int(sys.argv[1])
	print(epochs,' epochs')

	try:
		reload_model=sys.argv[3]
	except:
		reload_model="NO"

	reader= Reader(IMAGE_SIZE,IMAGE_BORDER)
	preprocessor = PreProcessor()
	preprocessor.set_readers(reader,None,None)
	engine = Engine(preprocessor)

	# prepare the model
	model = mo.make_model(BATCH_SIZE,INPUT_SIZE)

	#opt = SGD(lr=LEARN_RATE, decay=decay_param, momentum=0.9, nesterov=True)
	opt = Nadam(lr=LEARN_RATE)#,clipvalue=100)

	model.compile(loss='categorical_crossentropy', optimizer=opt,metrics=["accuracy"])

	avg, std=preprocessor.compute_stats()

	print("image estimated statistics avg=",avg," std dev=",std)

	show_images(ROOT_FILE_NAME_TRAIN,FILE_COUNT_TRAIN,"TRAIN",preprocessor)
	show_images(ROOT_FILE_NAME_VAL,FILE_COUNT_VAL,"VAL",preprocessor)
	show_images(ROOT_FILE_NAME_TEST,FILE_COUNT_TEST,"TEST",preprocessor)
	
	if reload_model != "NO":
		print('load model weights:',reload_model)
		model.load_weights(reload_model)

	reader = Reader(IMAGE_SIZE,IMAGE_BORDER)
	reader_val = Reader(IMAGE_SIZE,IMAGE_BORDER)
	preprocessor.set_readers(reader, reader_val,None)

	print('model parameters:',model.count_params())
	print('model characteristics:',model.summary())
	print('----------------------------------------------------------------------------------------')

	hist=engine.fit(model, epochs, avg, std)
	print(hist.history)

	reader.stop_read()
	reader_val.stop_read()

	# save learned weights
	f="%d-%m-%y"
	filename=WEIGHTS_DIR + '/weights-'+dt.date.today().strftime(f)
	model.save_weights(filename,overwrite=True)

	# test the model
	reader_test = Reader(IMAGE_SIZE,IMAGE_BORDER)
	preprocessor.set_readers(None,None,reader_test)
	pred = engine.predict(model)

	# accuracy=engine.compute_accuracy(pred,y_test)
	# print('accuracy on test data: ',accuracy*100, '%')
	# show_results(pred,X_test,y_test)

	pl.plot(hist.history,len(hist.history['acc']))
	os.system('./plot.sh')

if __name__ == "__main__":
    main()


