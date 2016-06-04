from __future__ import print_function
from assoc_space import AssocSpace
import sys
#import numpy as np

def normalizeAndMapToFive(value, maxV, minV):
	return ((value-minV)*5.0)/(maxV-minV);
	
def loadConcreteNessRatings(concretenessRatingFile):
	concreteNessDictionary = {};
	with open(concretenessRatingFile, "r") as f:
		i=0;
		for line in f:
			if i ==0:
				i=i+1;
				continue;
			tokens=line.split("\t");
			## 0-word, 1- bigram, 2-concreteness mean,3-concreteness SD
			word = tokens[0].strip().replace(" ","_");
			concreteNessDictionary[tokens[0].strip()] = float(tokens[2].strip());
			i=i+1;
	return concreteNessDictionary;

if len(sys.argv) < 2:
	print("python ",sys.argv[0]," <targetfile> <AssocSpaceDirectory>(optional)")
	sys.exit();

assocDir="/windows/drive2/For PhD/KR Lab/UMD_vision_integration/Image_Riddle/conceptnet5/data/assoc/assoc-space-5.4";
if len(sys.argv) == 3:
	assocDir = sys.argv[2];
assocSpace = AssocSpace.load_dir(assocDir);
names = assocSpace.labels
#print "argmax of sigma", np.argmax(assocSpace.sigma);

## TODO: load concreteness data
concretenessRatingFile = "/windows/drive2/For PhD/KR Lab/UMD_vision_integration/Image_Riddle/Concreteness_ratings_Brysbaert_et_al_BRM.txt";
concreteNessDictionary = loadConcreteNessRatings(concretenessRatingFile);
	
with open(sys.argv[1], "r") as f:
	i=0;
	for line in f:
		if line.startswith("##"):
			print(line,end="");
			continue;
		words=line.split("\t");
		word =words[0].strip();
		word1="/c/en/"+word;
		indexOfWord = names.index(word1);
		# u is numpy matrix rows are labels and columns are eigen vectors.
		# first vector supposed to be the largest one
		string = str(assocSpace.u[indexOfWord,0])+"\t"+line;
		if word in concreteNessDictionary.keys():
			string = str(concreteNessDictionary[word])+"\t"+string;
		else:
			string = str(normalizeAndMapToFive(assocSpace.u[indexOfWord,0],0.00324597,-0.00188222))+"\t"+string;
		print(string,end="");

