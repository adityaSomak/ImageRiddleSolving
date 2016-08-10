from __future__ import print_function

from conceptnet5.query import *
from assoc_space import AssocSpace
import collections
import util
import os
import numpy as np
import scipy

from gensim import models


#%%%%%%%%
## Get word2vec vectors for Targets for all images. 
## Calculate mean for all images. Order the targets and output.
#%%%%%%%%
def orderMergedTargetsAccordingToCentroid(mergeStageDSTuples, allSeedsDictionary, inferenceFolder, seedPrefix):
	vectors = [];
	words = set();
	for mergeDS in mergeStageDSTuples:
		sortedScoreAndIndexList= mergeDS[0];
		targetWordsList= mergeDS[1];
		targetWordsDictonary = mergeDS[2];
		for indexTarget in range(0,len(50)):
			indexAndScore = sortedScoreAndIndexList[indexTarget];
			targetWord = targetWordsList[indexAndScore[0]];
			words.add(targetWord);
	words = list(words);
	for targetWord in words:
		targetVector = conceptnet_util.getWord2VecVector(targetWord);
		vectors.append(targetVector);
	meanVector = np.mean(vectors);
	tuples =[];
	for i in xrange(len(words)):
		dist = scipy.spatial.distance.cosine(meanVector,vectors[i]);
		tuples.append((dist,word));
	sorted(tuples,key=lambda x:x[0]);
	outputFile = open(inferenceFolder+"opt_"+seedPrefix+"_inf_all.txt","w");
	for tup in tuples:
		print('%s\t%g\t%g' % (tup[0], tup[1], allSeedsDictionary[tup[0]][2]),file=outputFile);
	outputFile.close();
	return outputFile.name;

#%%%%%%%%
## Take the mean of the means for all images.
## Get the most similar words from word2vec model to this mean.
## Write down these words in a file.
#%%%%%%%%
def orderWordsAccordingToCentroid(centroids, reweightedSeedsFiles, allSeedsDictionary, inferenceFolder, seedPrefix):
	meanVector = np.mean(centroids);
	tuples = word2vec_model.similar_by_vector(meanVector,topn=20);
	outputFile = open(inferenceFolder+"opt_"+seedPrefix+"_inf_all.txt","w");
	for tup in tuples:
		print('%s\t%g\t%g' % (tup[0], tup[1], allSeedsDictionary[tup[0]][2]),file=outputFile);
	outputFile.close();
	return outputFile.name;

#%%%%%%%%
## Get word2vec vectors for Seeds. Calculate mean.
#%%%%%%%%
def calculateWord2vecCentroidAndHighestAcc(allSeedsDictionary, reweightedSeedsFileName):
	# read the weights and seed_in_CNet
	seedsDetected_weights = util.readReweightedSeeds(reweightedSeedsFileName,allSeedsDictionary);
	seedWordsList = list(seedsDetected_weights.keys());
	meanVector = None;
	for indexSeed in range(0,len(seedWordsList)):
		seedWord = seedWordsList[indexSeed];
		seedVector = getWord2VecVector(seedWord);
		if meanVector == None:
			meanVector = seedVector;
		else:
			meanVector = meanVector+seedVector;
	meanVector = meanVector/len(seedWordsList);		
	return meanVector;

def getHypernymFilter(seedsDetected_weights,allSeedsDictionary):
	detections = list(seedsDetected_weights.keys());
	weights=map(lambda x: seedsDetected_weights[x],detections);
	[newDetections,newWeights] = collapseSuperClasses(detections,weights,allSeedsDictionary);
	return set(newDetections);
	
	
def getOrderedDetectionsAndSuperclasses(detections,weights,allSeedsDictionary):
	detectionsDict = collections.OrderedDict(); # seed_detected -> weight, amtConcreteness, superclasses
	groupDetected=False;
	for i in range(0,len(detections)):
		detection_i = detections[i].strip();
		detection_cnet = allSeedsDictionary[detection_i][1];
		amtConcreteNessScore = allSeedsDictionary[detection_i][2];
		detectionsDict[detection_i] = (weights[i],amtConcreteNessScore,\
		set(getSuperOrSubclasses("/c/en/"+detection_cnet,True)));
		if detection_i == "group":
			groupDetected=True;

	########## sort based on concreteness-score
	detectionsDict = collections.OrderedDict(sorted(detectionsDict.iteritems(), key=lambda x: x[1]))
	return [detectionsDict,groupDetected]

######################################################
##### For animal,reptile,dinosaur => get dinosaur
######################################################
def collapseSuperClasses(detections,weights,allSeedsDictionary):
	[detectionsDict,groupDetected]= getOrderedDetectionsAndSuperclasses(detections,weights,allSeedsDictionary)
	
	## HEURISTIC used: if group is detected, then we eliminate only those which 
	## 		are too general i.e. reptile isA animal, dinosaur isA animal
	##		We consider animal as too general.	
	newDetections = [];
	newWeights = [];
	for item_j in detectionsDict.items():
		detection_j = item_j[0];
		key_j = "/c/en/"+detection_j;
		newWeight_j = item_j[1][0];
		
		## if the detection is in the list of superclasses in
		## any one of the other seeds, then DONOT include.
		isSuperclass = False;
		numSuperClasses = 0;
		for item_i in reversed(detectionsDict.items()):
			detection_i = item_i[0];
			if detection_i != detection_j:
				if key_j in item_i[1][2]:
					isSuperclass=True;
					if groupDetected:
						numSuperClasses = numSuperClasses+1;
					else:
						break;
		if (groupDetected and numSuperclasses >= 2) or (isSuperclass):
			continue;
		else:
			newDetections.append(detection_j);
			newWeights.append(newWeight_j);
	return [newDetections,newWeights];



def getSuperOrSubclasses(word,isSuperClass):
	superclasses =[];
	criteria ={};
	if isSuperClass:
		criteria['start']=word;
	else:
		criteria['end']=word;
	criteria['rel']="/r/IsA";
	for assertion in query(criteria):
		if isSuperClass and assertion['end'].startswith('/c/en'):
			superclasses.append(util.encode(assertion['end']));
		if (not isSuperClass) and assertion['start'].startswith('/c/en'):
			superclasses.append(util.encode(assertion['start']));
	return superclasses

def getSimilarity(word,target,normalize=False):
	if not word.startswith("/c/en"):
		word="/c/en/"+word;
	if not target.startswith("/c/en"):
		target="/c/en/"+target;
	similarity = assocSpace.assoc_between_two_terms(word,target);
	if normalize:
		return (similarity-minSimilarity)/(maxSimilarity-minSimilarity);
	return similarity;

def getWord2VecKeyFoundCode(term):
	if "_" in term:
		return 1;
	else:
		if term not in word2vec_model.vocab:
			return TOO_RARE_WORD_CODE;
	return 1;

def getWord2VecVector(word):
	words=[word];
	if "_" in word:
		words=word.split("_");
	if " " in word:
		words=word.split(" ");
	wordsPresent=[];
	for term in words:
		if term in word2vec_model.vocab:
			wordsPresent.append(term);
	if len(wordsPresent)==0:
		return np.zeros(300);
	vector=word2vec_model[wordsPresent[0]];
	for index in range(1,len(wordsPresent)):
		vector = vector+word2vec_model[wordsPresent[index]];
	vector= vector/len(wordsPresent);
	return vector
	
		
def getWord2VecSimilarity(word1,word2,normalize=False):
	usePhraseSimilarity = False;
	if "_" in word1 or "_" in word2:
		usePhraseSimilarity = True;
	if not usePhraseSimilarity:
		if word2 not in word2vec_model.vocab or word1 not in word2vec_model.vocab:
			return NOT_FOUND_IN_CORPUS_CODE;
		similarity = word2vec_model.similarity(word1,word2);
	else:
		words =  word1.split("_");
		words2 = word2.split("_");
		for word in words:
			if word not in word2vec_model.vocab:
				return NOT_FOUND_IN_CORPUS_CODE;
		for word in words2:
			if word not in word2vec_model.vocab:
				return NOT_FOUND_IN_CORPUS_CODE;
		similarity = word2vec_model.n_similarity(words,words2);
	if normalize:
		return util.computeNormalizedValue(similarity,1,-1);
	return similarity
	
def getNormalizedAssocSpaceSimilarity(similarity):
	return (similarity-minSimilarity)/(maxSimilarity-minSimilarity);

def getCentralityScore(word,normalize=False):
	if not word.startswith("/c/en"):
		word="/c/en/"+word;
	indexOfWord = names.index(word);
	centrality = assocSpace.u[indexOfWord,0];
	if normalize:
		return (centrality-minCentrality)/(maxCentrality-minCentrality);
	return centrality;

def filterEnglishWords(string):
	return string.startswith("/c/en/");

def getTermsSimilarToWord(word,limit=100):
	if not word.startswith("/c/en"):
		word="/c/en/"+word;
	vector = assocSpace.row_named(word);
	return assocSpace.terms_similar_to_vector(vector,filter=filterEnglishWords,num=limit);


def getTermsSimilarToWordFast(word,limit=100):
	sortedIndicesFileName = "intermediateFiles/opt/preprocessAssoc/"+word.replace("/","_")+".npz";
	simFileName = "intermediateFiles/opt/preprocessAssoc/"+word.replace("/","_")+"_sim.npz";
	if not os.path.isfile(sortedIndicesFileName):
		sim = assocSpace.assoc.dot(assocSpace.row_named("/c/en/"+word));
		indices = np.argsort(sim)[::-1];
		np.savez_compressed(sortedIndicesFileName,indices[:1000]);
		sim_first1k = np.array([sim[index] for index in indices[:1000]]);
		np.savez_compressed(simFileName,sim_first1k);
	
	sim =  np.load(simFileName);
	indices = np.load(sortedIndicesFileName);
	data = []
	for index in indices:
		if len(data) == limit:
			break
		if filterEnglishWords(names[index]):
			data.append((names[index], sim[index]))
	return data



minSimilarity=-1;
maxSimilarity= 1;
minCentrality= -0.00188222;
maxCentrality= 0.00324597;
assocDir = "../conceptnet5/data/assoc/assoc-space-5.4";
assocSpace = AssocSpace.load_dir(assocDir);
names = assocSpace.labels
word2vec_model =  models.word2vec.Word2Vec.load_word2vec_format('../../../DATASETS/GoogleNews-vectors-negative300.bin', binary=True);
TOO_RARE_WORD_CODE=-3;
NOT_FOUND_IN_CORPUS_CODE=-2;
