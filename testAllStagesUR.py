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

# Normalize seed names, normalize the weights as in other Re-weighting.
# Write the re-weighted file.
def normalizeSeedsAndWeights(allSeedsDictionary,detectionFolder,prefix,\
	imageNum,inferenceFolder,apiUsed="clarifai"):
	if apiUsed == "clarifai":
		[detections,weights] = util.processClarifaiJsonFile(detectionFolder+prefix+"_"+str(imageNum)+".txt");
	else:
		[detections,weights] = util.getDetectionsFromTSVFile(detectionFolder+prefix+"_"+str(imageNum)+".txt");
		[detections, weights] = conceptnet_util.chooseSingleRepresentativeDetection(allSeedsDictionary, detections, weights);

	if len(detections) == 0:
		return None;
	print(detections);
	sumTotal = 0;
	seeds = [];
	newWeights =[];
	for index_i in range(len(detections)):
		detection = detections[index_i].strip();
		newWeight = float(weights[index_i]);
		if detection not in allSeedsDictionary.keys():
			continue;
		sumTotal = sumTotal+newWeight;
		newWeights.append(newWeight);
		seeds.append(allSeedsDictionary[detection][1]);
	outputFile = open(inferenceFolder+"opt_"+prefix+"_"+str(imageNum)+"_c.txt","w");
	for index_i in range(len(seeds)):
		seed_i = seeds[index_i];
		normalizedWeight = newWeights[index_i]/sumTotal;
		print(index_i,"\t",seed_i,"\t",allSeedsDictionary[seed_i][0],"\t",normalizedWeight,file=outputFile);
	return outputFile.name;

def solveIndividualRiddles(detectionFolder,prefix,allSeedsDictionary,inferenceFolder,seedsCentralityFile,
	pipelineStage, imageNum, API_USED):
	sumIndividualAccuracy = 0;
	trainingImage = detectionFolder+prefix+"_"+str(imageNum)+".txt";
	WordWeightsOptimization2.VERBOSE = False;
	reweightedSeedsFileName = normalizeSeedsAndWeights(allSeedsDictionary,\
	detectionFolder,prefix,int(imageNum),inferenceFolder,API_USED);
	print("\treweighting seeds completed..");
	if reweightedSeedsFileName == None:
		if pipelineStage == "clarifai":
			return [None, None];
		elif pipelineStage == "merge":
			return [None,None, None];
		else:
			return None;
	if pipelineStage == "clarifai":
		## Note: We will not do parallel processing for this
		centroid = conceptnet_util.calculateWord2vecCentroidAndHighestAcc(allSeedsDictionary,reweightedSeedsFileName);
		return [centroid,reweightedSeedsFileName];


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
	

#if len(sys.argv) < 4:
#	print("python ",sys.argv[0]," <seedsCentralityfile> <detectionsFolder> <number-of-puzzles> \
#		<inferenceFolder> <stage> <from>,<to> parallel");
#	print("Stage options are: clarifai/merge/all.")
#	sys.exit();

[seedsCentralityFile,allSeedsDictionary,detectionFolder,numberOfPuzzles,inferenceFolder,\
	pipelineStage,API_USED,startPuzzle,endPuzzle,sortedFilePrefixList_file,argsdict] = \
	util.processAllArgumentsReturnVariables(sys.argv[1:]);	

accuracyResultsFolderprefix = "accuracyResults/UR/";
if API_USED != "clarifai":
	accuracyResultsFolderprefix = "accuracyResults/resnet/UR/";
	choice = util.R_CHOICE;
else:
	choice = util.CHOICE;

if startPuzzle != -1:
	accuracyFile = open(accuracyResultsFolderprefix+'accuracyFile_all_puzzles_'+str(pipelineStage)+'_'+\
		str(startPuzzle)+'_ur.txt','w');
else:
	accuracyFile = open(accuracyResultsFolderprefix+'accuracyFile_all_puzzles_'+str(pipelineStage)+'_ur.txt','w');

startTime = time.time();
#for choice in parameterSpace:
if startPuzzle != -1:
	puzzleAccuracyFile = open(accuracyResultsFolderprefix+'accuracyFile_all_images_'+str(pipelineStage)+'_'+\
		str(startPuzzle)+'_'+str(endPuzzle)+'_ur.txt','w');
else:
	puzzleAccuracyFile = open(accuracyResultsFolderprefix+'accuracyFile_all_images_'+str(pipelineStage)+'_ur.txt','w');
util.setParameters(choice[0],choice[1],choice[2],choice[3],choice[4],choice[5]);

sumIndividualAccuracy = 0;
sumPuzzleAccuracy=0;
choiceString = ','.join(str(x) for x in choice);
string = choiceString+"\t"+str(sumIndividualAccuracy)+"\n";
accuracyFile.write(string);
accuracyFile.flush();

string = choiceString+"\t"+str(sumPuzzleAccuracy)+"\n";
puzzleAccuracyFile.write(string);
puzzleAccuracyFile.flush();

with open(sortedFilePrefixList_file, 'r') as myfile:
	i=0;
	for prefix in myfile:
		if startPuzzle != -1 and i < startPuzzle:
			i=i+1;
			continue;
		if endPuzzle != -1 and i == endPuzzle:
			break;
		if i == numberOfPuzzles:
			break;
		prefix = prefix.replace("\n","");
		finalOutputFileName = inferenceFolder+"opt_"+prefix+"_inf_all.txt";
		if os.path.isfile(finalOutputFileName):
			i=i+1;
			continue;
				
		prefixMinus = prefix;
		if "-" in prefix:
			prefixMinus=prefix[:prefix.index("-")];		
		if conceptnet_util.getWord2VecKeyFoundCode(prefixMinus)==conceptnet_util.TOO_RARE_WORD_CODE:
			i=i+1;
			continue;

		#try:
		print('Iteration for prefix:\t%s\n' % (prefix));
		
		sumAccuracy=0;
		if argsdict["par"] == "parallel":
			sumAccuracy = Parallel(n_jobs=4)(delayed(solveIndividualRiddles)(detectionFolder, prefix,\
				allSeedsDictionary, inferenceFolder, seedsCentralityFile,pipelineStage, imageNum) for imageNum in range(1,5));
			for acc in sumAccuracy:
				sumIndividualAccuracy=acc+sumIndividualAccuracy;
		else:
			centroids =[];
			reweightedSeedsFiles =[];
			mergeStageDSTuples =[];
			incomplete = False;
			for imageNum in range(1,5):
				if pipelineStage == "all":
					indivAcc = solveIndividualRiddles(detectionFolder,prefix,\
						allSeedsDictionary, inferenceFolder, seedsCentralityFile, pipelineStage,\
						imageNum,API_USED);
					if indivAcc == None:
						incomplete = True;
						break;
					sumAccuracy = sumAccuracy + indivAcc;
				elif pipelineStage == "clarifai":
					[centroid, reweightedSeedsFileName] = solveIndividualRiddles(detectionFolder,prefix,\
						allSeedsDictionary, inferenceFolder, seedsCentralityFile, pipelineStage, imageNum,\
						API_USED);
					if reweightedSeedsFileName == None:
						incomplete = True;
						break;
					#sumAccuracy = sumAccuracy+acc;
					centroids.append(centroid);
					reweightedSeedsFiles.append(reweightedSeedsFileName);
				elif pipelineStage == "merge":
					[sortedScoreAndIndexList, targetWordsList, targetWordsDictonary] = solveIndividualRiddles(\
						detectionFolder,prefix,allSeedsDictionary, inferenceFolder, \
						seedsCentralityFile, pipelineStage, imageNum,API_USED);
					if sortedScoreAndIndexList == None:
						incomplete = True;
						break;
					mergeStageDSTuples.append((sortedScoreAndIndexList, targetWordsList, targetWordsDictonary));
				sumIndividualAccuracy= sumAccuracy+sumIndividualAccuracy;
			
		if incomplete:
			i=i+1;
			continue;

		if pipelineStage == "all":
			pslTwo.VERBOSE= False;
			finalTargetsFileName = pslTwo.callPSLModelTwo(allSeedsDictionary,inferenceFolder,prefix);
		elif pipelineStage == "clarifai":
			finalTargetsFileName = conceptnet_util.orderWordsAccordingToCentroid(centroids, reweightedSeedsFiles, \
				allSeedsDictionary, inferenceFolder, prefix);
		elif pipelineStage == "merge":
			finalTargetsFileName = conceptnet_util.orderMergedTargetsAccordingToCentroid(mergeStageDSTuples, \
				allSeedsDictionary, inferenceFolder, prefix);

		[acc,simWord] = calculateRelativeAccuracy(prefix, finalTargetsFileName, 20);
		if simWord != None:
			sumPuzzleAccuracy= sumPuzzleAccuracy+acc;
			string= prefix+"\t"+simWord+"\t"+str(acc)+"\n";
			puzzleAccuracyFile.write(string);

		i=i+1;
		#except Exception as e:
		#	raise e
		
		if i%50==0:
			string = choiceString+"\t"+str(sumIndividualAccuracy)+"\n";
			accuracyFile.write(string);
			accuracyFile.flush();
			puzzleAccuracyFile.flush();
		
string = choiceString+"\t"+str(sumIndividualAccuracy)+"\n";
accuracyFile.write(string);

string = choiceString+"\t"+str(sumPuzzleAccuracy)+"\n";
puzzleAccuracyFile.write(string);
puzzleAccuracyFile.close();

accuracyFile.close();

endTime = time.time();
print('\t Elapsed Time in seconds: %g' % (endTime-startTime)); 
