from __future__ import print_function

import os
import sys
import time

from joblib import Parallel, delayed

import WordWeightsOptimization2
import conceptnet_util
import pslModelOneNewOptimization_v2 as pslOne
import pslModelTwoNewOptimization as pslTwo
import util
from preprocess import mergeTargets, clusterTargets


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

def solveIndividualRiddles(detectionFolder,prefix,allSeedsDictionary,inferenceFolder,seedsCentralityFile,part):
	sumIndividualAccuracy = 0;
	#for part in range(1,5):
	trainingImage = detectionFolder+prefix+"_"+str(part)+".txt";
	WordWeightsOptimization2.VERBOSE = False;
	reorderedSeedsFiles = WordWeightsOptimization2.reorderWeightsBasedOnPopularity(allSeedsDictionary,\
	detectionFolder,prefix,int(part),int(part),inferenceFolder);
	reweightedSeedsFileName = reorderedSeedsFiles[0];
	print("\tmerging targets completed..");

	#### Step 1: Merge targets from different seeds.
	mergeTargets.VERBOSE= False;
	[sortedScoreAndIndexList, targetWordsList, targetWordsDictonary,seedsDetected_weights,\
	orderedSeedWordsList,allSeedsDictionary] = mergeTargets.mergeTargetsFromDetectedSeeds(\
	reweightedSeedsFileName, seedsCentralityFile,1500);
	print("\tmerging targets completed..");

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
parameterSpace = [(0.9,1,0.4,2,1,4)];#\
#(0.8,1,0.4,1,1,4),\
#(0.9,1,0.4,1,1,4),\
#(0.9,2,0.3,1,3,4),\
#(0.9,2,0.3,3,3,4)];

if len(sys.argv) < 4:
	print("python ",sys.argv[0]," <seedsCentralityfile> <detectionsFolder> <number-of-puzzles> <inferenceFolder> <from>,<to> parallel")
	sys.exit();

seedsCentralityFile = sys.argv[1];
allSeedsDictionary = util.populateModifiedSeedsAndConcreteScoreDictionary(seedsCentralityFile);

numberOfPuzzles = int(sys.argv[3]);
detectionFolder = sys.argv[2]+"Detection/";
inferenceFolder = sys.argv[4];
startPuzzle =-1;
endPuzzle = -1;
if len(sys.argv) > 5:
	tok = sys.argv[5].split(",");
	startPuzzle = int(tok[0]);
	endPuzzle = int(tok[1]);
	numberOfPuzzles = endPuzzle - startPuzzle + 1;

sortedFilePrefixList_file = sys.argv[2]+"filelist.txt";

if startPuzzle != -1:
	accuracyFile = open('accuracyResults/accuracyFile_all_puzzles_'+str(startPuzzle)+'_iur.txt','w');
else:
	accuracyFile = open('accuracyResults/accuracyFile_all_puzzles_iur.txt','w');

startTime = time.time();
tries =0;
for choice in parameterSpace:
	if startPuzzle != -1:
		puzzleAccuracyFile = open('accuracyResults/accuracyFile_all_puzzles_'+str(tries)+'_'+str(startPuzzle)+'_'+str(endPuzzle)+'_iur.txt','w');
	else:
		puzzleAccuracyFile = open('accuracyResults/accuracyFile_all_puzzles_'+str(tries)+'_iur.txt','w');
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
					
			#if partNumber != -1 and i%numParts != partNumber:
			#	i=i+1;
			#	continue;
			prefixMinus = prefix;
			if "-" in prefix:
				prefixMinus=prefix[:prefix.index("-")];		
			if conceptnet_util.getWord2VecKeyFoundCode(prefixMinus)==conceptnet_util.TOO_RARE_WORD_CODE:
				i=i+1;
				continue;

			try:
				print('Iteration for prefix:\t%s\n' % (prefix));
				#TODO: parallelize this for loop
				#sumAccuracy = solveIndividualRiddles(detectionFolder,prefix,allSeedsDictionary,inferenceFolder,seedsCentralityFile);
				#arguments = [detectionFolder,prefix,allSeedsDictionary,inferenceFolder,seedsCentralityFile];
				sumAccuracy = Parallel(n_jobs=4)(delayed(solveIndividualRiddles)(detectionFolder,prefix,allSeedsDictionary,\
					inferenceFolder,seedsCentralityFile,part) for part in range(1,5));
				for acc in sumAccuracy:
					sumIndividualAccuracy=acc+sumAccuracy;
					
				pslTwo.VERBOSE= False;
				finalTargetsFileName = pslTwo.callPSLModelTwo(allSeedsDictionary,inferenceFolder,prefix);
				[acc,simWord] = calculateRelativeAccuracy(prefix, finalTargetsFileName, 20);
				if simWord != None:
					sumPuzzleAccuracy= sumPuzzleAccuracy+acc;
					string= prefix+"\t"+simWord+"\t"+str(acc)+"\n";
					puzzleAccuracyFile.write(string);
				i=i+1;
			except Exception as e:
				raise
			
			if i%25==0:
				string = str(choice[0])+","+str(choice[1])+","+str(choice[2])+","+\
				str(choice[3])+","+str(choice[4])+","+str(choice[5])+"\t"+str(sumIndividualAccuracy)+"\n";
				accuracyFile.write(string);
				accuracyFile.flush();
			if i%12==0:
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
