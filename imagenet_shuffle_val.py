import tarfile
from PIL import Image
import numpy as np
import os
import random as rn
import zipfile as z
import re
import sys

#   read imagenet 2012 validation files and produce random zip archives
#	use validation label file to rename the filenames in the zip files 
#
#   this program is derived from imagenet_shuffle_train.py and has been changed to take into account
#   the specifics of the validation data file naming scheme and labels files structure

ARCHIVE_COUNT=100
MAX_FILES=100000
GROUND_TRUTH_FILE='ILSVRC2012_validation_ground_truth.txt'
WORDS='wordsval.txt'
dataset_dir='.'
ARCHIVE_DIR='/home/yves/datasets/imagenet/val'

file_table = np.ndarray(shape=(MAX_FILES), dtype=object)

# we use a dict because the file count (60k) is not enough that it will be a performance bottleneck
labels_dict={}

def read_synsets():
	''' create a dict linking the image file number to the synset of the image '''
	global labels_dict
	synset_dict={}
	# read a file with one line per synset
	with open(os.path.join(dataset_dir, WORDS), 'r') as f:
		words=list(f)
		for word in words:
			synset,rank,label = str.split(word, ' ')
			print('synset : ',synset,' rank: ',rank,' label: ',label)
			synset_dict[rank]=synset
	# read a file with one line per image, the line rank being the synset index in field 2 (rank) of former file
	with open(os.path.join(dataset_dir, GROUND_TRUTH_FILE), 'r') as f:
		ranks=list(f)
		file_no = 1
		for rank in ranks:
			rank = str.split(rank, '\n')[0]
			labels_dict[file_no] = synset_dict[rank]
			print('file : ',file_no,' synset : ',labels_dict[file_no])			
			file_no += 1

def get_synset(index):
	''' given a file number, return the synset which is associated to the image in the file '''
	global labels_dict
	return labels_dict[index]

def load_file_table(directory):
	''' load a file table for all imagenet validation files '''
	global file_table
	count=0
	for dirpath, directories, files in os.walk(directory):  # there should be just one directory with plenty of files
		j = 0
		for file in files:
			filepath = os.path.join(dirpath, file)
			if re.match(".*JPEG", filepath):
				file_no = int(file[15:23])
				print("filepath : ", filepath, " for ", file, " number " , file_no)
				file_table[j]=filepath
				j += 1
				count += 1
	return count

def write_randomized_archive(count):
	''' write ARCHIVE_COUNT zip archives, filling them with randomly selected files '''
	zip_table = np.ndarray(shape=(ARCHIVE_COUNT), dtype=object)
	
	for i in range(0,ARCHIVE_COUNT):  # create all zipfiles at once and open them
		zip_table[i]=z.ZipFile(os.path.join(ARCHIVE_DIR,'image_'+str(i)+'.zip'),'w')

	for i in range(0,MAX_FILES):			  # put each file in a randomly selected archive
		if (file_table[i] !=None):
			print(file_table[i])
			archive_number = rn.randrange(0,ARCHIVE_COUNT)
			simple_file_name = re.sub('.*/','',file_table[i])
			file_no = int(simple_file_name[15:23])
			synset = get_synset(file_no) 
			simple_file_name = synset + "_" + str(file_no) + ".JPEG"
			print("file name in archive ",simple_file_name)
			zip_table[archive_number].write(file_table[i],simple_file_name)

	for i in range(0,ARCHIVE_COUNT):  # close all zipfiles at once 
		zip_table[i].close()

def main():

	global dataset_dir

	dataset_dir=sys.argv[1]  # name of directory holding the synsets of Imagenet

	print("start")
	read_synsets()
	count = load_file_table(dataset_dir)
	write_randomized_archive(count)
	print ("end")

if __name__ == "__main__":
    main()
