import util
import sys
import os
import numpy as np
from numpy import linalg as LA

#######################################################################
########		PIPELINE STAGE III.
########		Input: detected seeds and targets
########		Output: Merge into matrix and get top targets
#######################################################################

def calculateScalarProjection(queryVector, ontoVector):
	return np.dot(queryVector,ontoVector)/LA.norm(ontoVector);
	
	
	
'''
	##############################################################
	ASSUMPTION: For each seed, all the files exist and they are all 
		cleaned up i.e. the top targets are the most VISUALLY
		similar targets and last column is visual similarity and
		second-last is CNet similarity.
	TODO: Heuristic Combined Similarity:
		If Vis-similarity=1, sim = assoc-space-sim.
		else,                sim = (Vis-similarity+0.01*assoc-space-sim)/1.01
	##############################################################
'''	
def mergeTargetsFromDetectedSeeds(reweightedSeedsFileImg, allSeedsMapFile, alpha1=1,alpha2=0.01, targetsPerSeed=3000):
	#ALL_TARGETS_DIRECTORY ="intermediateFiles/allTargets/test1/";
	ALL_TARGETS_DIRECTORY ="intermediateFiles/allTargets/test_res/";
	# read the seed to seed_in_CNet mapping
	# detected_seed -> centrality_score, seed_in_conceptnet, amtconcScore
	allSeedsDictionary = util.populateModifiedSeedsAndConcreteScoreDictionary(allSeedsMapFile);
	# read the weights and seed_in_CNet
	seedsDetected_weights = util.readReweightedSeeds(reweightedSeedsFileImg,allSeedsDictionary);
	
	# Order the list of seeds.
	seedWordsList = list(seedsDetected_weights.keys());
	
	# key= target-word, value =list corresponding to each seed in @seedWordsList.
	targetWordsDictionary= {};
		
	#semanticSimSeeds = set();
	#semanticSimFile = 	"intermediateFiles/allTargets/semanticSimSeeds.txt";
	#with open(semanticSimFile,"r") as f:
	#	for line in f:
	#		semanticSimSeeds.add(line);

	#TODO: parallelize this for loop	
	for indexSeed in range(0,len(seedWordsList)):
		seedWord = seedWordsList[indexSeed];
		seedWeight = seedsDetected_weights[seedWord];
		amtConcSeedWord = allSeedsDictionary[seedWord][2];
		#if seedWord in allSeedsDetected_toCNet.keys():
		#targetFile = "intermediateFiles/opt/test1/"+allSeedsDetected_toCNet[seedWord][1]\
		#+"_targets__sorted.txt";
		#else:
		targetFile = ALL_TARGETS_DIRECTORY+seedWord+"_targets__sorted.txt";
		
		##################################################
		##### TODO: move this whole calc to pre-processing
		##################################################
		numOnesNansOrZeros = [0,0,0];
		if not os.path.isfile(targetFile):
			targetFile = ALL_TARGETS_DIRECTORY+seedWord+"_targets_.txt";
			if not os.path.isfile(targetFile):
				continue;
		
		with open(targetFile, "r") as f:
			i=0;
			for line in f:
				if i > 500:
					break;
				tokens = line.split("\t");
				if tokens[len(tokens)-1].strip() =="1.0":
					numOnesNansOrZeros[0] = numOnesNansOrZeros[0]+1;
				elif tokens[len(tokens)-1].strip() =="nan":
					numOnesNansOrZeros[1] = numOnesNansOrZeros[1]+1;
				elif tokens[len(tokens)-1].strip() =="0.0":
					numOnesNansOrZeros[2] = numOnesNansOrZeros[2]+1;
				i=i+1;
		
		#print numOnesNansOrZeros
		usingSemanticSim = False;		
		if max(numOnesNansOrZeros)/500.0 > 0.9 or amtConcSeedWord < util.CONCRETENESS_THRESHOLD_MERGET:
			## just use the similarity from CNet
			targetFile = ALL_TARGETS_DIRECTORY+seedWord+"_targets_.txt";
			usingSemanticSim = True;
		if VERBOSE:
			print "\t",targetFile
		
		# Load the whole text file in memory.
		lines=[];		
		with open(targetFile, "r") as f:
			i=0;
			for line in f:
				if i > targetsPerSeed:
					break;
				lines.append(line);
				i=i+1;

		#with open(targetFile, "r") as f:
		i=0;
		for line in lines:
			tokens = line.split("\t");
			tokens =  map(lambda x:x.strip(),tokens);
			if len(tokens) < 3:
				continue;
			targetWord = tokens[0][6:];
			
			finalSim = 0;
			normalizedAssocSpaceSimilarity = util.computeNormalizedValue(float(tokens[len(tokens)-2]),\
			1,-1);
			if usingSemanticSim:
				finalSim = normalizedAssocSpaceSimilarity;
				#targetWordsDictionary[targetWord][indexSeed] = normalizedAssocSpaceSimilarity;
			else:
				if tokens[len(tokens)-1] == "nan":
					visualSimilarity = 1.0;
				else:
					visualSimilarity = float(tokens[len(tokens)-1]);

				if visualSimilarity == 1.0:
					finalSim = normalizedAssocSpaceSimilarity;
					#targetWordsDictionary[targetWord][indexSeed] = normalizedAssocSpaceSimilarity;
				else:
					finalSim = (alpha1 *visualSimilarity + alpha2*normalizedAssocSpaceSimilarity)/(alpha1+alpha2);
					#targetWordsDictionary[targetWord][indexSeed] = (alpha1 *visualSimilarity + \
					#alpha2*normalizedAssocSpaceSimilarity)/(alpha1+alpha2);

			try:
				targetWordsDictionary[targetWord][indexSeed] = finalSim;
			except KeyError:
				targetWordsDictionary[targetWord] = np.zeros(len(seedWordsList));
				targetWordsDictionary[targetWord][indexSeed] = finalSim;
			#if targetWord not in targetWordsDictionary.keys():
			#	targetWordsDictionary[targetWord] = np.zeros(len(seedWordsList));
			i=i+1;
	
	print "\tthe target dictionary is loaded...";
	##############################
	## Score the targets
	##############################
	## scalar projection => score(target_i) = target_i (dot) V_img/|V_img|
	V_img = np.zeros(len(seedWordsList));
	for indexSeed in range(0,len(seedWordsList)):
		V_img[indexSeed] = seedsDetected_weights[seedWordsList[indexSeed]];
	
	targetWordsList = list(targetWordsDictionary.keys());
	scoreAndIndexList =[];
	for indexTargetWord in range(0,len(targetWordsList)):
		targetVector = targetWordsDictionary[targetWordsList[indexTargetWord]];
		scoreAndIndexList.append((indexTargetWord, calculateScalarProjection(targetVector, V_img)));
	print "\tthe vectors are scored...";
	##############################
	## Sort the targets
	##############################
	sortedScoreAndIndexList = sorted(scoreAndIndexList, key=lambda tup: tup[1],reverse=True);
	return sortedScoreAndIndexList, targetWordsList, targetWordsDictionary,\
	seedsDetected_weights,seedWordsList,allSeedsDictionary;

if __name__ == "__main__":
	# sys.arv[1] = reweighted set of seeds for an image
	# sys.argv[2] = detected seeds to modified map file
	# sys.argv[3] = targets for each seed
	
	VERBOSE= True;
	ALL_TARGETS_DIRECTORY ="intermediateFiles/allTargets/test1/";
	[sortedScoreAndIndexList, targetWordsList, targetWordsDictionary,seedsDetected_weights,seedWordsList,allSeedsDictionary] = \
	mergeTargetsFromDetectedSeeds(sys.argv[1],sys.argv[2],1,0.01);
	
	for i in range(0,len(sortedScoreAndIndexList)):
		if i > 3000:
			break;
		indexAndScore = sortedScoreAndIndexList[i];
		targetWord = targetWordsList[indexAndScore[0]];
		comb_similarity_list = targetWordsDictionary[targetWord];
		nonzeroSeedIndices = np.nonzero(comb_similarity_list)[0];
		nonZeroSeeds = map(lambda x:seedWordsList[x],nonzeroSeedIndices);
		print targetWord,"\t",indexAndScore[1],"\t",str(nonZeroSeeds);
