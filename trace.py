from keras.callbacks import Callback
from keras import backend as K
import numpy as np

class TraceWeights(Callback):

	def __init__(self,mode,engine,trace_after):
		''' mode should be 0 for training, 1 for testing , engine is used to get current batch data'''
		self.mode = mode
		self.engine = engine
		self.trace_after = trace_after
		self.counter = 0

	def on_train_begin(self, logs={}):
		print('train begin')

	def fmt(self,n):
		return str.rjust(str(n),12)

	def print_ndarray_stats(self, s, i, X):
		''' i layer number,
			s data name,
			X data array '''
		# data is sampled to avoid using too much computation, and statistics are printed then
		# sample values of X into X1 
		X1=np.take(X,rn.sample(range(0,X.size),1000))
		print("L:",i, ":",s, ":", X.shape,":", self.fmt(np.amin(X1)),":",self.fmt(np.amax(X1)),":", self.fmt(np.mean(X1)),":", \
			   self.fmt(np.count_nonzero(np.isnan(X1))), ":", self.fmt(np.count_nonzero(~np.isnan(X1))) )

	def on_batch_begin(self, batch, logs={}):
		''' on batch begin we display the statistics of the weights and the outputs to see how NaN propagate '''
		self.counter = self.counter + 1
		if self.counter < 1400*self.trace_after :
			return # //////// trace nothing until approx 100 epochs
		print("\n")
		number_of_layers= len(self.model.layers)
		for i in range(1,number_of_layers):
			weights=self.model.layers[i].get_weights()
			if len(weights)>0:
				self.print_ndarray_stats("W", i,abs(weights[0]))	# trace gradient scale
			get_layer_output = K.function([self.model.layers[0].input,K.learning_phase()],[self.model.layers[i].output])
			X = self.engine.X_batch_current
			layer_output = get_layer_output([X,self.mode])[0]
			self.print_ndarray_stats("Y", i,abs(layer_output))		# trace activation scale
