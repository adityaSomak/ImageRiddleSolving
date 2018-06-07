from __future__ import print_function

import conceptnet_util
import numpy as np
import util
import sys
import random

def calculateRelativeAccuracy(expectedWord, detections, limitSuggestions=20):
	if "-" in expectedWord:
		expectedWord=expectedWord[:expectedWord.index("-")];
	maxSimilarity = 0;
	similarWord = None;
	for word in detections:
		i=0;
		if i==limitSuggestions:
			break;
		similarity = conceptnet_util.getWord2VecSimilarity(expectedWord,word.strip(),True);
		if similarity > maxSimilarity:
			maxSimilarity = similarity;
			similarWord = word.strip();
		i=i+1;
	return [maxSimilarity,similarWord];
	
def processClarifaiJsonFile(fileName):
	with open(fileName, 'r') as myfile:
		line = myfile.read();
	line = line.replace("u\'", "");
	line = line.replace("\'", "");
	line = line.replace(":", "\n");
	lines = line.split("\n");
	return lines;
print("python ",sys.argv[0]," <detectionsFolder> <number-of-puzzles> ")

numberOfPuzzles = int(sys.argv[2]);
detectionFolder = sys.argv[1]+"Detection/";

sortedFilePrefixList_file = sys.argv[1]+"filelist.txt";

puzzleAccuracyFile = open('baselineaccuracyFile_all_puzzles.txt','w');
	
sumIndividualAccuracy = 0;
sumPuzzleAccuracy=0;
	
with open(sortedFilePrefixList_file, 'r') as myfile:
	i=0;
	for prefix in myfile:
		prefix = prefix.replace("\n","");

		if i == numberOfPuzzles:
			break;
		if conceptnet_util.getWord2VecKeyFoundCode(prefix)==conceptnet_util.TOO_RARE_WORD_CODE:
			i=i+1;
			continue;
		try:
			print('Iteration for prefix:\t%s\n' % (prefix));
			sumAcc = 0;
			simWords = "";
			for part in range(1,5):
				trainingImageFileName = detectionFolder+prefix+"_"+str(part)+".txt";
				lines = processClarifaiJsonFile(trainingImageFileName);
				detections = (lines[15][2:lines[15].index("]")]).split(",");
				weights = (lines[16][2:lines[16].index("]")]).split(",");
				print(detections);
			
				[acc,simWord]= calculateRelativeAccuracy(prefix, detections, 20);	
				print('\taccuracy %g' % acc);
				sumIndividualAccuracy = sumIndividualAccuracy+acc;
				
				sumAcc = sumAcc+acc;
				simWords = simWords+","+str(simWord);
				i=i+1;		
			
			sumPuzzleAccuracy= sumPuzzleAccuracy+sumAcc/4.0;
			string= prefix+"\t"+simWords+"\t"+str(sumAcc/4.0)+"\n";
			puzzleAccuracyFile.write(string);
				
		except Exception as e:
			raise
			
		#if i%25==0:
		#	string = str(sumIndividualAccuracy)+"\n";
		#	accuracyFile.write(string);
		#	accuracyFile.flush();
		if i%12==0:
			puzzleAccuracyFile.flush();
			
#string = str(sumIndividualAccuracy)+"\n";
#accuracyFile.write(string);
	
string = str(sumPuzzleAccuracy)+"\n";
puzzleAccuracyFile.write(string);
puzzleAccuracyFile.close();

#accuracyFile.close();
