import tarfile
from PIL import Image
import numpy as np
import os
from zipfile import ZipFile
from io import BytesIO
import random as rn
from random import shuffle
from multiprocessing import Process, Queue, Manager
import logging
import sys
#from logging.config import fileConfig

TRANSLATION_AUGMENTATION=0.05
FLIP_AUGMENTATION=False
ROTATION_ANGLE=0
AUGMENTATION=False

NB_PROCESS=8

def log(s):
	logging.debug(s)

class Reader:

	def __init__(self,image_size,image_border):
		''' read table for converting file names to words, in memory '''
		#logging.basicConfig(filename='main.log',level=logging.DEBUG,\
		#	  format='%(asctime)s -- %(name)s -- %(levelname)s -- %(message)s')
		#log("====== init reader ======")
		rn.seed(a=1, version=2)
		self.queue = Manager().Queue(maxsize=4)
		self.image_size = image_size
		self.image_border = image_border
		self.input_size = self.image_size + 2*self.image_border
		with open('words1000.txt', 'r') as f:
			words=list(f)
			words.sort()  # sort on synset key
			i = 0
			self.synset_table = np.ndarray(shape=(len(words)), dtype=object)
			self.labels =       np.ndarray(shape=(len(words)), dtype=object)
			for word in words:
				code,label = str.split(word, '\t')
				self.synset_table[i] = code  # the classification index of a given synset, here the code value, will be its line number in the file
				self.labels[i]       = label
				i += 1

	@staticmethod
	def read_tarfile(file_name):
		''' read a jpeg image tar file in memory as a list of numpy 2D arrays '''
		images=[]
		print("file name : ",file_name)
		tar = tarfile.open(file_name, 'r|*')
		for tarinfo in tar:
			print(tarinfo.name, "is", tarinfo.size, "bytes in size and is", end="")
			if tarinfo.isreg():
				buf = tar.extractfile(tarinfo)
				im = Image.open(buf)
				print("image size",im.size)
				arr = np.array(im)
				images.append(arr)
		tar.close()
		return images

	def index_of(self,synset):
		''' find synset classification index, so convert an n9999999 format to a number from 0 to 999 '''
		index = np.searchsorted(self.synset_table,synset)
		return index # index of synset in words.txt

	def label_of(self,index):
		''' find label of a given classification index '''
		return self.labels[int(index)] 

	def check_y(self,y):
		i = 0
		for value in y:
			if value>=1000:
				log ("#"+str(i)+" invalid value : "+str(value))
			i += 1	

	def read_zipfile(self,root_file_name, file_count, image_size, max_to_read, avg, std):
		''' read a jpeg image zip file in memory as a list of numpy 2D arrays
			root_file_name : 	start of the zip archive name, not including the '_'
			file_count : 		number of files, which are numbered from 0 to this number
			data :				an ndarray, shape (number of images, X,Y,color)
			max_to_read :		maximum number of images to read
		'''
		try:
			#log("root file name: "+root_file_name+" max to read : "+str(max_to_read))
			index_in_data = 0
			data = np.zeros((max_to_read,3,image_size,image_size),dtype=np.float32)
			y = np.zeros((max_to_read),dtype=int)
			while True:  # if the end of the dataset is reached, then restart from the beginning
				for file_number in range(0,file_count):  # enumerate all the dataset zipfiles
					archive_name=root_file_name+'_'+str(file_number)+'.zip'
					with ZipFile(archive_name, 'r') as zip:
						zipinfos = zip.infolist()
						image_count = len(zipinfos)
					shuffle(zipinfos) # randomize file order in archive to avoid short range repetition
					#log("read archive "+archive_name+" rank : "+str(file_number)+" of "+str(file_count,)+" archives")
					self.processes=list()
					for rank in range(0,NB_PROCESS):
						p=Process(target=self.image_reader,args=(avg, std, NB_PROCESS, rank, archive_name, image_size, zipinfos, self.queue,))
						p.start()
						self.processes.append(p)
					#log("--- processes Started ---")
					#log("will get from queue : "+str(image_count))
					for i in range (0,image_count):  # loop on images inside one archive
						#log("before get queue "+str(i)+" ...")
						try :
							filename, arr = self.queue.get(block=True)
							#log("... after get queue "+str(i)+" file: "+filename)
						except Exception as e:	
							#log("queue error : {}".format(e))
							except_type, except_class, tb = sys.exc_info()
							#log("queue error info : "+str(except_type)+" , "+str(except_class))
						y[index_in_data] = self.index_of(filename[0:9])  # the filename first characters are the synset
						try:
							data[index_in_data,0,:,:] = arr[:,:,0]
							data[index_in_data,1,:,:] = arr[:,:,1]
							data[index_in_data,2,:,:] = arr[:,:,2]
						except:
							except_type, except_class, tb = sys.exc_info()
							#log("data copy error: "+str(except_type)+" , "+str(except_class))
						index_in_data += 1
						if index_in_data >= max_to_read:
							self.check_y(y)
							#log("yield data batch")
							yield data, y
							index_in_data = 0  # reset index to start of batch
		except Exception as e:
			#log("parent process error {}".format(e))
			self.stop_read()

	def stop_read(self):  # shall be called manually if the generator is partially used
		for process in self.processes:
			process.terminate()
		self.processes=()  # in case stop is called again
		#log("--- processes ended ---")
		
	def reshape_image(self,X,image_size):
		if len(X.shape)<3:
			X2 = np.zeros((image_size,image_size,3))
			X2[:,:,0] = X
			X2[:,:,1] = X
			X2[:,:,2] = X
			return X2
		else:
			return X

	def image_reader(self, avg, std, nb_process, process_rank, archive_name, image_size, shuffled_infos, queue):
		# logging.basicConfig(filename='image_reader'+str(process_rank)+'.log',level=logging.DEBUG,\
		# 	  format='%(asctime)s -- %(name)s -- process:'+str(process_rank)+'-- %(levelname)s -- %(message)s')
		try:
			#log("process started")
			entry_rank=0
			with ZipFile(archive_name, 'r') as zip:
				# we use shuffled info instead of zipinfos = zip.infolist()
				for zipinfo in shuffled_infos:  # in each zipfile enumerate component files
					if entry_rank % nb_process == process_rank:
						#log("image : "+zipinfo.filename)
						image_data = zip.read(zipinfo.filename)
						input = BytesIO(image_data)
						input.seek(0)  # to avoid OSError
						im = Image.open(input)
						arr = np.array(im,dtype=np.float32)
						arr = self.reshape_image(arr,image_size)  # account for 1 channel B&W images
						if avg is None:  # do not scale when avg is not defined
							#log("no scaling")
							X = arr
						else: 
							#log("scaling")
							X = self.scale_data(arr,avg,std)
						if AUGMENTATION:
							X = self.augment(X)
						#log("send to queue...")
						queue.put((zipinfo.filename,X),block=True)	
						#log("...sent to queue")
					entry_rank += 1
		except Exception as e:
			#log("child process error {}".format(e))
			pass
			
	def scale_data(self,data,avg,std):
		''' scale the image pixel values using average and standard deviation per channel '''
		scale=128
		data = data.astype('float32')
		#data = data.reshape((3,self.image_size,self.image_size))
		data2 = np.zeros((self.input_size,self.input_size,3),dtype=np.float32)
		if len(data.shape)==3:
			for j in range(0,3):
				#substract mean and divide per std deviation, per sample and per color channel 
				data2[self.image_border:self.image_size+self.image_border,self.image_border:self.image_size+self.image_border,j] =\
					(data[:,:,j] - avg[j]) / std[j]
		else:  # in case the image has only one channel
			for j in range(0,3):
				#substract mean and divide per std deviation, per sample and per color channel 
				data2[self.image_border:self.image_size+self.image_border,self.image_border:self.image_size+self.image_border,j] =\
					(data[:,:] - avg[j]) / std[j]
		return data2

	def augment(self,X):
		''' compute pseudo-random translation, flip etc of X data to augment the dataset inputs '''
		''' pseudo-random augmentation means that multiple augmenation on the same data will yield the same result '''
		X2=np.zeros(X.shape)
		x_max=X.shape[1]
		y_max=X.shape[2]
		x_range=range(0,x_max)
		y_range=range(0,y_max)
		max_translation=int(x_max*TRANSLATION_AUGMENTATION)
		# loop on sample, channel, x coord, y coord
		flip=bool(rn.randrange(0,1,1))
		x_translation=rn.randrange(0, max_translation,1)
		y_translation=rn.randrange(0, max_translation,1)
		for k in x_range:    # enumerate image pixels on k,l
			for l in y_range:
				if ROTATION_ANGLE>0 and k==0 and l==0: # just one at the start of an image translation per pixel
					random_angle=rn.randrange(-ROTATION_ANGLE, ROTATION_ANGLE,1)
					for j in range(0,3):
						X[j,:,:] = nd.rotate(X[j,:,:], random_angle, reshape=False)							
				if k+x_translation in x_range and l+y_translation in y_range:
					if flip:
						for j in range(0,3): # same augmentation translation/flip for all channels
							X2[j,x_max-k,y_max-l]=image=X[j,k+x_translation,l+y_translation]
					else :
						for j in range(0,3):
							X2[j,k,l]=image=X[j,k+x_translation,l+y_translation]
		return X2


