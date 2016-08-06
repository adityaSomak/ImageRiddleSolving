from __future__ import print_function

import conceptnet_util
import numpy as np
import util
import sys
import os
import random
import time

import mergeTargets
import clusterTargets
import WordWeightsOptimization2
import pslModelOneNewOptimization_v2 as pslOne
import pslModelTwoNewOptimization as pslTwo

from joblib import Parallel, delayed 

def calculateRelativeAccuracy(expectedWord, finalReorderedTargetsFileName, limitSuggestions=50):
	if "-" in expectedWord:
		expectedWord=expectedWord[:expectedWord.index("-")];
	maxSimilarity = 0;
	similarWord = None;
	with open(finalReorderedTargetsFileName, 'r') as f:
		i=0;
		for line in f:
			tokens =line.split("\t");
			if i==limitSuggestions:
				break;
			similarity = conceptnet_util.getWord2VecSimilarity(expectedWord,tokens[0].strip(),True);
			if similarity > maxSimilarity:
				maxSimilarity = similarity;
				similarWord = tokens[0].strip();
			i=i+1;
	return [maxSimilarity,similarWord];

# TODO: Implement Method
def orderMergedTargetsAccordingToCentroid(mergeStageDSTuples):
	raise NotImplementedError;

# TODO: Implement Method
def orderWordsAccordingToCentroid(centroids, reweightedSeedsFiles):
	raise NotImplementedError;

# TODO: Implement Method
def calculateWord2vecCentroidAndAcc(reweightedSeedsFileName):
	raise NotImplementedError;

def solveIndividualRiddles(detectionFolder,prefix,allSeedsDictionary,inferenceFolder,seedsCentralityFile,
	pipelineStage, imageNum):
	sumIndividualAccuracy = 0;
	trainingImage = detectionFolder+prefix+"_"+str(imageNum)+".txt";
	WordWeightsOptimization2.VERBOSE = False;
	reorderedSeedsFiles = WordWeightsOptimization2.reorderWeightsBasedOnPopularity(allSeedsDictionary,\
	detectionFolder,prefix,int(imageNum),int(imageNum),inferenceFolder);
	reweightedSeedsFileName = reorderedSeedsFiles[0];
	print("\treweighting seeds completed..");
	if pipelineStage == "clarifai":
		## Note: We will not do parallel processing for this
		[acc,centroid] = calculateWord2vecCentroidAndAcc(reweightedSeedsFileName);
		return [acc,centroid,reweightedSeedsFileName];

	#### Step 1: Merge targets from different seeds.
	mergeTargets.VERBOSE= False;
	[sortedScoreAndIndexList, targetWordsList, targetWordsDictonary,seedsDetected_weights,\
	orderedSeedWordsList,allSeedsDictionary] = mergeTargets.mergeTargetsFromDetectedSeeds(\
	reweightedSeedsFileName, seedsCentralityFile,1500);
	print("\tmerging targets completed..");
	if pipelineStage == "merge":
		## Note: We will not do parallel processing for this
		return [sortedScoreAndIndexList, targetWordsList, targetWordsDictonary];

	#### Step 2: cluster the merged set of targets
	[sortedScoreAndIndexList,pairwiseDistancesTargetWords] = clusterTargets.returnClustersFast(\
	sortedScoreAndIndexList, targetWordsList, targetWordsDictonary, orderedSeedWordsList, 2500);
	print(pairwiseDistancesTargetWords.shape);
	print("\tclustering targets completed..");

	#### Step 3: create 1-word and 2-word model
	pslOne.VERBOSE = False;
	finalReorderedTargetsFileName = pslOne.optimizeAndInferConceptsModelOneNew(\
	allSeedsDictionary,seedsDetected_weights,\
	orderedSeedWordsList,reweightedSeedsFileName,sortedScoreAndIndexList, targetWordsList, targetWordsDictonary, \
	pairwiseDistancesTargetWords);

	[acc,simWord]= calculateRelativeAccuracy(prefix, finalReorderedTargetsFileName, 20);	
	print('\taccuracy %g' % acc);
	sumIndividualAccuracy = sumIndividualAccuracy+acc;
	return sumIndividualAccuracy;
	
#(0.8,1,0.8,2,1,4),\ #done
choice = (0.9,1,0.4,2,1,4);#\
#(0.8,1,0.4,1,1,4),\
#(0.9,1,0.4,1,1,4),\
#(0.9,2,0.3,1,3,4),\
#(0.9,2,0.3,3,3,4)];

if len(sys.argv) < 4:
	print("python ",sys.argv[0]," <seedsCentralityfile> <detectionsFolder> <number-of-puzzles> \
		<inferenceFolder> <stage> <from>,<to> parallel");
	print("Stage options are: clarifai/merge/all.")
	sys.exit();

seedsCentralityFile = sys.argv[1];
allSeedsDictionary = util.populateModifiedSeedsAndConcreteScoreDictionary(seedsCentralityFile);

detectionFolder = sys.argv[2]+"Detection/";
numberOfPuzzles = int(sys.argv[3]);
inferenceFolder = sys.argv[4];
pipelineStage = sys.argv[5];
startPuzzle =-1;
endPuzzle = -1;
if len(sys.argv) > 6:
	tok = sys.argv[6].split(",");
	startPuzzle = int(tok[0]);
	endPuzzle = int(tok[1]);
	numberOfPuzzles = endPuzzle - startPuzzle + 1;
if len(sys.argv) > 7:
	if sys.argv[6] == "parallel" and pipelineStage != "all":
		print("Not Permitted!!! Use parallel only with all stages");
		sys.exit();

sortedFilePrefixList_file = sys.argv[2]+"filelist.txt";

if startPuzzle != -1:
	accuracyFile = open('accuracyResults/IUR/accuracyFile_all_puzzles_'+str(startPuzzle)+'_iur.txt','w');
else:
	accuracyFile = open('accuracyResults/IUR/accuracyFile_all_puzzles_iur.txt','w');

startTime = time.time();
#for choice in parameterSpace:
if startPuzzle != -1:
	puzzleAccuracyFile = open('accuracyResults/IUR/accuracyFile_all_puzzles_'+str(pipelineStage)+'_'+\
		str(startPuzzle)+'_'+str(endPuzzle)+'_iur.txt','w');
else:
	puzzleAccuracyFile = open('accuracyResults/IUR/accuracyFile_all_puzzles_'+str(pipelineStage)+'_iur.txt','w');
util.setParameters(choice[0],choice[1],choice[2],choice[3],choice[4],choice[5]);

sumIndividualAccuracy = 0;
sumPuzzleAccuracy=0;
string = str(choice[0])+","+str(choice[1])+","+str(choice[2])+","+\
str(choice[3])+","+str(choice[4])+","+str(choice[5])+"\t"+str(sumIndividualAccuracy)+"\n";
accuracyFile.write(string);
accuracyFile.flush();

string = str(choice[0])+","+str(choice[1])+","+str(choice[2])+","+\
str(choice[3])+","+str(choice[4])+","+str(choice[5])+"\t"+str(sumPuzzleAccuracy)+"\n";
puzzleAccuracyFile.write(string);
puzzleAccuracyFile.flush();

with open(sortedFilePrefixList_file, 'r') as myfile:
	i=0;
	for prefix in myfile:
		prefix = prefix.replace("\n","");
		finalOutputFileName = inferenceFolder+"opt_"+prefix+"_inf_all.txt";
		if os.path.isfile(finalOutputFileName):
			i=i+1;
			continue;
		if startPuzzle != -1 and i < startPuzzle:
			continue;
		if endPuzzle != -1 and i == endPuzzle:
			break;
		if i == numberOfPuzzles:
			break;
				
		prefixMinus = prefix;
		if "-" in prefix:
			prefixMinus=prefix[:prefix.index("-")];		
		if conceptnet_util.getWord2VecKeyFoundCode(prefixMinus)==conceptnet_util.TOO_RARE_WORD_CODE:
			i=i+1;
			continue;

		try:
			print('Iteration for prefix:\t%s\n' % (prefix));
			
			sumAccuracy=0;
			if len(sys.argv) > 6 and sys.argv[7] == "parallel":
				sumAccuracy = Parallel(n_jobs=4)(delayed(solveIndividualRiddles)(detectionFolder, prefix,\
					allSeedsDictionary, inferenceFolder, seedsCentralityFile,imageNum) for imageNum in range(1,5));
			else:
				centroids =[];
				reweightedSeedsFiles =[];
				mergeStageDSTuples =[];
				for imageNum in range(1,5):
					if pipelineStage == "all":
						sumAccuracy = sumAccuracy+ solveIndividualRiddles(detectionFolder,prefix,\
							allSeedsDictionary, inferenceFolder, seedsCentralityFile, pipelineStage,\
							imageNum);
					elif pipelineStage == "clarifai":
						[acc, centroid, reweightedSeedsFileName] = solveIndividualRiddles(detectionFolder,prefix,\
							allSeedsDictionary, inferenceFolder, seedsCentralityFile, pipelineStage, imageNum);
						sumAccuracy = sumAccuracy+acc;
						centroids.append(centroid);
						reweightedSeedsFiles.append(reweightedSeedsFileName);
					elif pipelineStage == "merge":
						[sortedScoreAndIndexList, targetWordsList, targetWordsDictonary] = solveIndividualRiddles(\
							detectionFolder,prefix,allSeedsDictionary, inferenceFolder, \
							seedsCentralityFile, pipelineStage, imageNum);
						mergeStageDSTuples.append((sortedScoreAndIndexList, targetWordsList, targetWordsDictonary));
			
			for acc in sumAccuracy:
				sumIndividualAccuracy=acc+sumAccuracy;
				
			if pipelineStage == "all":
				pslTwo.VERBOSE= False;
				finalTargetsFileName = pslTwo.callPSLModelTwo(allSeedsDictionary,inferenceFolder,prefix);
			elif pipelineStage == "clarifai":
				finalTargetsFileName = orderWordsAccordingToCentroid(centroids, reweightedSeedsFiles);
			elif pipelineStage == "merge":
				finalTargetsFileName = orderMergedTargetsAccordingToCentroid(mergeStageDSTuples);

			[acc,simWord] = calculateRelativeAccuracy(prefix, finalTargetsFileName, 20);
			if simWord != None:
				sumPuzzleAccuracy= sumPuzzleAccuracy+acc;
				string= prefix+"\t"+simWord+"\t"+str(acc)+"\n";
				puzzleAccuracyFile.write(string);

			i=i+1;
		except Exception as e:
			raise
		
		if i%50==0:
			string = str(choice[0])+","+str(choice[1])+","+str(choice[2])+","+\
			str(choice[3])+","+str(choice[4])+","+str(choice[5])+"\t"+str(sumIndividualAccuracy)+"\n";
			accuracyFile.write(string);
			accuracyFile.flush();
			puzzleAccuracyFile.flush();
		
tries= tries+1;
string = str(choice[0])+","+str(choice[1])+","+str(choice[2])+","+\
str(choice[3])+","+str(choice[4])+","+str(choice[5])+"\t"+str(sumIndividualAccuracy)+"\n";
accuracyFile.write(string);

string = str(choice[0])+","+str(choice[1])+","+str(choice[2])+","+\
str(choice[3])+","+str(choice[4])+","+str(choice[5])+"\t"+str(sumPuzzleAccuracy)+"\n";
puzzleAccuracyFile.write(string);
puzzleAccuracyFile.close();

accuracyFile.close();

endTime = time.time();
print('\t Elapsed Time in seconds: %g' % (endTime-startTime)); 
