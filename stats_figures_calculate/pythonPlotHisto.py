import numpy as np
import matplotlib.pyplot as plt
import csv
import sys

allVariants =[('accuracyFile_all_images_all_ur.txt','UR+All','ur_all.jpg'),\
('accuracyFile_all_images_all_iur.txt','IUR+All','iur_all.jpg'),\
('accuracyFile_all_images_all_gur.txt','GUR+All','gur_all.jpg'),\
('accuracyFile_all_images_merge_ur.txt','UR+RR','ur_rr.jpg'),\
('accuracyFile_all_images_merge_iur.txt','IUR+RR','iur_rr.jpg'),\
('accuracyFile_all_images_merge_gur.txt','GUR+RR','gur_rr.jpg'),\
('accuracyFile_all_images_clarifai_ur.txt','UR+VB','ur_vb.jpg'),\
('accuracyFile_all_images_clarifai_iur.txt','IUR+VB','iur_vb.jpg'),\
('accuracyFile_all_images_clarifai_gur.txt','GUR+VB','gur_vb.jpg')]

i=0;
for variant in allVariants:
	data=[];
	with open(variant[0],'rb') as file:
		reader = csv.reader(file, delimiter='\t');
		for line in reader:
			data.append(float(line[2]));

	plt.figure(i);
	i=i+1;
	plt.gcf().subplots_adjust(bottom=0.15)
	plt.gcf().subplots_adjust(left=0.15)
	plt.title(variant[1],fontsize=24);
	plt.ylabel('Frequency',fontsize=24)
	plt.xlabel('Accuracy',fontsize=24)
	plt.yticks(fontsize=20);
	plt.xticks(fontsize=20);
	plt.hist(data, facecolor='w');
	#plt.show()
	plt.savefig(variant[2]);