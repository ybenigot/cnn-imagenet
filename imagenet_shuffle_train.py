import tarfile
from PIL import Image
import numpy as np
import os
import random as rn
import zipfile as z
import re
import sys


ARCHIVE_COUNT=100
MAX_FILES=3000
MAX_FILES2=10000

file_table = np.ndarray(shape=(MAX_FILES,MAX_FILES2), dtype=object)

def load_file_table(directory):
	''' load a file table for all imagenet files '''
	global file_table
	i = 0
	count=0
	for dirpath, directories, files in os.walk(directory):
		j = 0
		for file in files:
			filepath = os.path.join(dirpath, file)
#			print("filepath : ", filepath)
			if re.match(".*JPEG", filepath):
				file_table[i,j]=filepath
			count += 1
			j += 1
		i += 1
	return count


def write_randomized_archive(count):
	''' write ARCHIVE_COUNT zip archives, filling them with randomly selected files '''
	zip_table = np.ndarray(shape=(ARCHIVE_COUNT), dtype=object)
	
	for i in range(0,ARCHIVE_COUNT):  # create all zipfiles at once and open them
		zip_table[i]=z.ZipFile('image_'+str(i)+'.zip','w')

	for i in range(0,MAX_FILES):			  # put each file in a randomly selected archive
		for j in range(0,MAX_FILES2):
			if (file_table[i,j] !=None):
				#print(file_table[i,j])
				archive_number = rn.randrange(0,ARCHIVE_COUNT)
				simple_file_name = re.sub('.*/','',file_table[i,j])
				zip_table[archive_number].write(file_table[i,j],simple_file_name)

	for i in range(0,ARCHIVE_COUNT):  # close all zipfiles at once 
		zip_table[i].close()

def main():

	dataset_dir=sys.argv[1]  # name of directory holding the synsets of Imagenet

	print("start")
	count = load_file_table(dataset_dir)
	write_randomized_archive(count)
	print ("end")

if __name__ == "__main__":
    main()
