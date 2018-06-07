from __future__ import print_function

import sys

import conceptnet_util
import pslModelOneNewOptimization_v2 as pslOne
import pslModelTwoNewOptimization as pslTwo
import util
from misc import WordWeightsOptimization3
from preprocess import mergeTargets, clusterTargets


def calculateRelativeAccuracy(expectedWord, finalReorderedTargetsFileName, limitSuggestions=100):
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

	
#(0.8,1,0.8,2,1,4),\
#(0.8,1,0.4,1,1,4),\ (68.09, worse than previous)
parameterSpace = [(0.9,1,0.4,2,1,4),\
(0.9,1,0.4,1,1,4),\
(0.9,2,0.3,1,3,4),\
(0.9,2,0.3,3,3,4)];

idealParatmeters =[(0.8,1,0.8,2,1,4)];

print("python ",sys.argv[0]," <seedsCentralityfile> <detectionsFolder> <number-of-puzzles> <inferenceFolder> <partnumber>,<parts> <chooseIdeal>")

seedsCentralityFile = sys.argv[1];
allSeedsDictionary = util.populateModifiedSeedsAndConcreteScoreDictionary(seedsCentralityFile);

numberOfPuzzles = int(sys.argv[3]);
detectionFolder = sys.argv[2]+"Detection/";
inferenceFolder = sys.argv[4];

sortedFilePrefixList_file = sys.argv[2]+"filelist.txt";

partNumber =-1;
numParts = -1;
if len(sys.argv) > 5:
	tok = sys.argv[5].split(",");
	partNumber = int(tok[0]);
	numParts = int(tok[1]);
	

if partNumber != -1:
	accuracyFile = open('accuracyResults/accuracyFile_all_puzzles_'+str(partNumber)+'_gur.txt','w');
else:
	accuracyFile = open('accuracyResults/accuracyFile_all_puzzles_gur.txt','w');
	
tries =0;
if sys.argv[6]=="True":
	parameterSpace = idealParatmeters;
		
for choice in parameterSpace:
	if partNumber != -1:
		puzzleAccuracyFile = open('accuracyResults/accuracyFile_all_puzzles_'+str(tries)+'_'+str(partNumber)+'_gur.txt','w');
	else:
		puzzleAccuracyFile = open('accuracyResults/accuracyFile_all_puzzles'+str(tries)+'_gur.txt','w');
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
			#if tries==0:
			#	i=i+1;
			#	continue;
			if i == numberOfPuzzles:
				break;
			prefixMinus = prefix;
			if "-" in prefix:
				prefixMinus=prefix[:prefix.index("-")];
			
			if partNumber != -1 and i%numParts != partNumber:
				i=i+1;
				continue;	
				
			if conceptnet_util.getWord2VecKeyFoundCode(prefixMinus)==conceptnet_util.TOO_RARE_WORD_CODE:
				i=i+1;
				continue;
			try:
				print('Iteration for prefix:\t%s\n' % (prefix));
				WordWeightsOptimization3.VERBOSE = False;
				reorderedSeedsFileNames = WordWeightsOptimization3.reorderWeightsBasedOnCloseness(allSeedsDictionary, \
																								  detectionFolder, prefix, inferenceFolder);
				
				for part in range(1,5):
					trainingImage = detectionFolder+prefix+"_"+str(part)+".txt";
					
					reweightedSeedsFileName = reorderedSeedsFileNames[part-1];
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
					
				pslTwo.VERBOSE= False;
				finalTargetsFileName = pslTwo.callPSLModelTwo(allSeedsDictionary,inferenceFolder,prefix);
				[acc,simWord] = calculateRelativeAccuracy(prefix, finalTargetsFileName, 10);
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
