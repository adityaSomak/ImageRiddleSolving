from __future__ import print_function

import conceptnet_util
import numpy as np
import util
import sys
import random

import mergeTargets
import clusterTargets
import WordWeightsOptimization2
import pslModelOneNewOptimization_v2

def calculateRelativeAccuracy(expectedWord, finalReorderedTargetsFileName, limitSuggestions=100):
	maxSimilarity = 0;
	with open(finalReorderedTargetsFileName, 'r') as f:
		i=0;
		for line in f:
			tokens =line.split("\t");
			if i==limitSuggestions:
				break;
			similarity = conceptnet_util.getWord2VecSimilarity(expectedWord,tokens[0].strip(),True);
			if similarity > maxSimilarity:
				maxSimilarity = similarity;
			i=i+1;
	return maxSimilarity;

	
	
sim_threshold_space = [0.7,0.8,0.9];#np.arange(0.3,0.9,0.1);
top_k_sim_targets_space = np.arange(1,4,1);
sim_threshold_onewordrule_space = np.arange(0.2,0.8,0.1);
sum_confidence_limit_space = np.arange(1,5,1);
conceptnet_sim_wt_space = np.arange(1,5,1);
word2vec_sim_wt_space = np.arange(1,5,1);

list1=[(x,y,z) for x in sim_threshold_space for y in top_k_sim_targets_space for z in sim_threshold_onewordrule_space];
list3=[(y,z) for y in conceptnet_sim_wt_space for z in word2vec_sim_wt_space];
print("python ",sys.argv[0]," <seedsCentralityfile> <detectionsFolder> <number-of-images> <partsList>")

seedsCentralityFile = sys.argv[1];
allSeedsDictionary = util.populateModifiedSeedsAndConcreteScoreDictionary(seedsCentralityFile);

numberOfTrainingImages = int(sys.argv[3]);
detectionFolder = sys.argv[2]+"Detection/";

sortedFilePrefixList_file = sys.argv[2]+"filelist.txt";
partsList = sys.argv[4].split(";");

choicesTriedAlready = set();
with open('accuracyFile_paramspace_old.txt','r') as f:
	for line in f:
		tokens=line.split("\t");
		choicesTriedAlready.add(tokens[0]);

for choice in choicesTriedAlready:
	print(choice);

accuracyFile = open('accuracyFile_paramspace.txt','w');

parameterSearchChoices=np.arange(1,10000,1);
for i in range(10001,len(list1)*len(sum_confidence_limit_space)*len(list3)):
	j = random.randint(0,i);
	if j < 9999:
		parameterSearchChoices[j]=i;

parameterSearchTries=0;
for choice in parameterSearchChoices:
	u = choice%len(sim_threshold_space);
	rest = choice/len(sim_threshold_space);
	
	v= rest%len(top_k_sim_targets_space);
	rest= rest/len(top_k_sim_targets_space);
	
	w= rest%len(sim_threshold_onewordrule_space);
	rest= rest/len(sim_threshold_onewordrule_space);
	
	x= rest%len(sum_confidence_limit_space);
	rest= rest/len(sum_confidence_limit_space);
	
	y= rest%len(conceptnet_sim_wt_space);
	rest= rest/len(conceptnet_sim_wt_space);
	
	z= rest%len(word2vec_sim_wt_space);
	
	choiceString = str(sim_threshold_space[u])+","+str(top_k_sim_targets_space[v])+","+\
	str(sim_threshold_onewordrule_space[w])+","+str(sum_confidence_limit_space[x])+","+\
	str(conceptnet_sim_wt_space[y])+","+str(word2vec_sim_wt_space[z]);
	if choiceString in choicesTriedAlready:
		continue;
	### Remove this later
	lastThreeChoiceString = str(sum_confidence_limit_space[x])+","+str(conceptnet_sim_wt_space[y])\
	+","+str(word2vec_sim_wt_space[z]);
	if lastThreeChoiceString == "1,1,1":
		continue;
	
	util.setParameters(sim_threshold_space[u],top_k_sim_targets_space[v],sim_threshold_onewordrule_space[w],\
	sum_confidence_limit_space[x],conceptnet_sim_wt_space[y],word2vec_sim_wt_space[z]);
		
	if parameterSearchTries >= 9999:
		break;
	
	print('\nIteration for try:%g' % (parameterSearchTries));
	sumAccuracy = 0;
	string = str(sim_threshold_space[u])+","+str(top_k_sim_targets_space[v])+","+str(sim_threshold_onewordrule_space[w])+","+\
	str(sum_confidence_limit_space[x])+","+str(conceptnet_sim_wt_space[y])+","+str(word2vec_sim_wt_space[z])+"\t"+str(sumAccuracy)+"\n";
	accuracyFile.write(string);
	accuracyFile.flush();
	
	with open(sortedFilePrefixList_file, 'r') as myfile:
		i=0;
		for prefix in myfile:
			prefix = prefix.replace("\n","");
			if i >= numberOfTrainingImages:
				break;
			try:
				print('Iteration for prefix:%g\t%s\n' % (parameterSearchTries,prefix));
				for part in partsList:
					trainingImage = detectionFolder+prefix+"_"+part+".txt";
					WordWeightsOptimization2.VERBOSE = False;
					reorderedSeedsFiles = WordWeightsOptimization2.reorderWeightsBasedOnPopularity(allSeedsDictionary,\
					detectionFolder,prefix,int(part),int(part));
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
					pslModelOneNewOptimization_v2.VERBOSE = False;
					finalReorderedTargetsFileName = pslModelOneNewOptimization_v2.optimizeAndInferConceptsModelOneNew(\
					allSeedsDictionary,seedsDetected_weights,\
					orderedSeedWordsList,reweightedSeedsFileName,sortedScoreAndIndexList, targetWordsList, targetWordsDictonary, \
					pairwiseDistancesTargetWords);
			
					acc= calculateRelativeAccuracy(prefix, finalReorderedTargetsFileName, 100);	
					print('\taccuracy %g' % acc);
					sumAccuracy = sumAccuracy+acc;
				i=i+1;
			except Exception as e:
				raise e
			
			if i%50==0:
				string = str(sim_threshold_space[u])+","+str(top_k_sim_targets_space[v])+","+str(sim_threshold_onewordrule_space[w])+","+\
				str(sum_confidence_limit_space[x])+","+str(conceptnet_sim_wt_space[y])+","+str(word2vec_sim_wt_space[z])+"\t"+str(sumAccuracy)+"\n";
				accuracyFile.write(string);
				accuracyFile.flush();
			
	string = str(sim_threshold_space[u])+","+str(top_k_sim_targets_space[v])+","+str(sim_threshold_onewordrule_space[w])+","+\
	str(sum_confidence_limit_space[x])+","+str(conceptnet_sim_wt_space[y])+","+str(word2vec_sim_wt_space[z])+"\t"+str(sumAccuracy)+"\n"
	accuracyFile.write(string);
	parameterSearchTries= parameterSearchTries+1;

accuracyFile.close();
