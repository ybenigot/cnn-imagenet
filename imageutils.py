from PIL import Image,ImageDraw,ImageFont
import numpy as np

def unpickle(file):
    import pickle
    fo = open(file, 'rb')
    dict = pickle.load(fo, encoding='bytes')
    fo.close()
    return dict

def display_image(im1,image_size,imageTitle):
	im1=im1.reshape((3,image_size,image_size)).transpose(1,2,0)
	img = Image.fromarray(im1, 'RGB')
	print("displaying image : ",imageTitle)
	draw = ImageDraw.Draw(img) # write title text on the image top
	font = ImageFont.truetype("Keyboard.ttf", 12)
	draw.text((10, 10),imageTitle,(255,255,255),font=font)
	img.show(title=imageTitle) # may show the image title, however it is already written in the image

def display_normalized_image(image,imageTitle,image_size,avg,std):
	im1=image
	for i in range(0,3):
		im1[i,:,:] = image[i,:,:] * std[i] + avg[i]
	display_image(im1.astype('uint8'),image_size,imageTitle)

def display_images(X,y,N,reader,INPUT_SIZE,avg,std):
	for i in range(0,N):
		display_normalized_image(X[i,:,:,:],reader.label_of(y[i]),INPUT_SIZE,avg,std)

def display_image_set(X,y,reader,INPUT_SIZE,avg,std,isMac):  # display a set of 16 (4x4) images
	im1 = np.ndarray(shape=(4*INPUT_SIZE,4*INPUT_SIZE,3), dtype=np.uint8)
	for a in range(0,4):
		for b in range(0,4):
			for i in range(0,3):
				im1[INPUT_SIZE*a:INPUT_SIZE*(a+1),INPUT_SIZE*b:INPUT_SIZE*(b+1),i] = X[a*4+b,i,:,:] * std[i] + avg[i]
	img = Image.fromarray(im1, 'RGB')
	draw = ImageDraw.Draw(img) # write title text on the image top
	if isMac:
		font = ImageFont.truetype("Keyboard.ttf", 12)
	else:	
		font = ImageFont.truetype("Ubuntu-L.ttf", 12)
	for a in range(0,4):
		for b in range(0,4):
			label = reader.label_of(y[a*4+b]).replace(",","\n")  # replace commas for image display
			draw.text((10+b*INPUT_SIZE, 10+a*INPUT_SIZE),label,(255,255,0),font=font)
			print("at : ",a,b," label: ",label)
	img.show() 

def load_dataset():
	dict={}
	for i in range(1,6):
		dict1=unpickle('/Users/yves/.keras/datasets/cifar-10-batches-py/data_batch_'+str(i))
		dict.update(dict1)

	Y_train=dict[b'labels']
	X_train=dict[b'data']

	print (X_train.shape)
	#for k in range(0,X_train.shape[0]):
	#	X_train[k] = reshape(X_train[k])

	display_image(X_train[0])
	display_image(X_train[1])
	display_image(X_train[2])

	return (X_train,Y_train)

def normalize(data):
	m=np.mean(data)
	s=np.std(data)
	return (data-m)/s

def mean1(data):
	''' substract mean per image sample and per color channel'''
	for i in range(0,data.shape[0]):
		for j in range(0,data.shape[1]):
			m = np.mean(data[i,j,:])
			data[i,j,:,:] = data[i,j,:,:]-m
	return data



def mean2(data1,data2,data3):
	''' substract mean per color channel for training set data1 from all datasets'''
	for j in range(0,data1.shape[1]):
		m = np.mean(data1[:,j,:])
		data1[:,j,:,:] -= m
		data2[:,j,:,:] -= m
		data3[:,j,:,:] -= m
	return data1, data2, data3

def whiten(data,epsilon):
	''' ZCA whiten per channel '''
	n=data.shape[0]
	p=data.shape[2] # width of an image ; here we assume square images
	for j in range(0,data.shape[1]): #enumerate color channels
		x = data[:,j,:,:].reshape(n,p*p) 								# x(imagePixels),sample#)
		print('before sigma',x.shape)
		sigma = x.dot(x.T) 
		print('after sigma\n')
		sigma  /=n
		u,s,v = np.linalg.svd(sigma)
		xWhite = np.diag(1./np.sqrt(s + epsilon)).dot(u.T).dot(x)		# compute PCA
		xWhite = u.dot(xWhite) 											# compute ZCA
		data[:,j,:,:]=xWhite.reshape(n,p,p)
	return data



