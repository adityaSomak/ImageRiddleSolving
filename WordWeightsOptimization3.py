from __future__ import print_function

import conceptnet_util
import util
import numpy as np
import os
import sys
from numpy import linalg as LA

def printSolution(outputFile,detectionAndSimilarity,allSeedsDictionary):
	detectionAndSimilarity = sorted(detectionAndSimilarity, key=lambda tup: tup[1],reverse=True);
	
	sumTotal = 0;
	for seedAndSimilarity in detectionAndSimilarity:
		sumTotal = sumTotal+seedAndSimilarity[1];
	
	for seedAndSimilarity in detectionAndSimilarity:
		seed_i = seedAndSimilarity[0];
		normalizedWeight = seedAndSimilarity[1]/sumTotal;
		if seed_i in allSeedsDictionary:
			print("0\t",seed_i,"\t",allSeedsDictionary[seed_i][0],"\t",normalizedWeight,file=outputFile);


def getSeedVector(allSeedsDictionary, seed, seed_detections_vector):
	try:
		seed_cnet= allSeedsDictionary[seed][1];
	except KeyError as ke:
		return [0]*len(seed_detections_vector);
	vector=[];
	for word in seed_detections_vector:
		try:
			word_cnet = allSeedsDictionary[word][1];
			vector.append(conceptnet_util.getSimilarity(seed_cnet, word_cnet, True));
		except KeyError as ke:
			vector.append(0);
	return vector

	
def getAverageSimilarity(allSeedsDictionary,seed, img_i, weights_all_images,detection_all_images,norms_images):
	avgSim=0;
	for img_j in range(1,5):
		if img_j != img_i:
			image_J_vector = np.array(weights_all_images[img_j]);
			seedVector = getSeedVector(allSeedsDictionary, seed, detection_all_images[img_j]);
			seedNorm = LA.norm(seedVector);
			if seedNorm > 0:
				cosineSim = np.dot(image_J_vector,seedVector)/(seedNorm*norms_images[img_j]);
				avgSim=avgSim+cosineSim;
	return avgSim/4.0;
	
'''
Iterate over each image, for each word- get the average similarity to the other
image-vectors.
'''	
def reorderWeightsBasedOnCloseness(allSeedsDictionary,detectionFolder,imagePrefix,\
inferenceFolder="intermediateFiles/opt_test/",apiUsed="clarifai"):
	outputFileNames=[];
	
	detection_all_images={};
	weights_all_images={};
	norms_images={}
	### Load all 4 image-vectors
	for img in range(1,5):
		if apiUsed == "clarifai":
			[detections, weights] = util.processClarifaiJsonFile(detectionFolder+imagePrefix+"_"+str(img)+".txt");
		else:
			[detections, weights] = util.getDetectionsFromTSVFile(detectionFolder+imagePrefix+"_"+str(img)+".txt");
			detections = conceptnet_util.chooseSingleRepresentativeDetection(allSeedsDictionary, detections);

		weights = map(lambda x: float(x),weights);
		detections = map(lambda x: x.strip(),detections);
		detection_all_images[img] = detections;
		weights_all_images[img] = weights;
		norms_images[img] = LA.norm(np.array(weights));
	
	for img_i in range(1,5):
		detections_i = detection_all_images[img_i];
		detectionAndSimilarity=[];
		for seed in detections_i:
			avgSimilarity = getAverageSimilarity(allSeedsDictionary,seed, img_i, weights_all_images,\
			detection_all_images,norms_images);
			detectionAndSimilarity.append((seed,avgSimilarity));
		
		outputFile = open(inferenceFolder+"opt_"+imagePrefix+"_"+str(img_i)+"_c.txt","w");
		printSolution(outputFile,detectionAndSimilarity,allSeedsDictionary);
		outputFileNames.append(outputFile.name);
	return outputFileNames;
		
########################################################################
############# Start of Script
########################################################################		
if __name__ == "__main__":
	if len(sys.argv) < 4:
		print("python ",sys.argv[0]," <seedsCentralityfile> <detectionsFolder> <target-name> <inferenceFolder>")
		sys.exit();

	VERBOSE = True;
	allSeedsDictionary = util.populateModifiedSeedsAndConcreteScoreDictionary(sys.argv[1]);
	reorderWeightsBasedOnCloseness(allSeedsDictionary,sys.argv[2],sys.argv[3],sys.argv[4]);
