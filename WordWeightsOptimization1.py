from __future__ import print_function

from assoc_space import AssocSpace

import scipy.optimize as optimize
import numpy as np
import os
import sys
'''
This script will first read the seed-> (centrality-score,seed_in_conceptnet) mapping.
Then it will take only 1 seed detection file.
The network of the words are separately optimized.
the output will be one optimized list of words and its weights.
'''

def computeNormalizedValue(value, maxV, minV):
	return (value-minV)/(maxV-minV);
	
def sumOfMax(x,edges):
	sumOfMax = 0;
	for i in range(len(edges)):
		## edge from u-> v => sum {max(c(u)-c(v)*\lambda,0)}
		## lambda = 0.75
		sumOfMax = sumOfMax + max(x[edges[i][0]]*10-x[edges[i][1]]*5,0);
	return sumOfMax

#def constraint(x):
#	global node_scores
#	return int(sum(x))-int(sum(node_scores))

def mergeBasedOnFunction(newWeight,detection,detectionIndices,node_scores,function):
	index = detectionIndices[detection];
	oldWeight = node_scores[index];
	if function=="max":
		node_scores[index] = max(newWeight,oldWeight);
	elif function=="min":
		node_scores[index] = min(newWeight,oldWeight);
	elif function=="avg":
		node_scores[index] = newWeight+oldWeight;
	
def processClarifaiJsonFile(fileName):
	with open(fileName, 'r') as myfile:
		line = myfile.read();
	line = line.replace("u\'", "");
	line = line.replace("\'", "");
	line = line.replace(":", "\n");
	lines = line.split("\n");
	return lines;

def exceedsThreshold(similarity):
	minV=-0.358846;
	maxV= 0.999747;
	value= (similarity-minV)/(maxV-minV);
	if value > 0.5:
		return True;
	else:
		return False;

def populateModifiedSeedsAndConcreteScoreDictionary(loadConcScores=True,weightConcScores=1.0):
	# detected_seed -> centrality_score, seed_in_conceptnet
	allSeedsDictionary={};
	with open(sys.argv[1], 'r') as f:
		for line in f:
			if line.startswith("##"):
				continue;
			tokens=line.split("\t");
			## conc. score = conc mean mapped to [0,2] scale+ 
			##               centrality mapped to [0,1];
			concereteNessScore = computeNormalizedValue(float(tokens[1].strip()),0.00324597,-0.00188222);
			if loadConcScores:
				concereteNessScore = concereteNessScore + \
				computeNormalizedValue(0-float(tokens[0].strip()),0,-5)*weightConcScores;
			if len(tokens) > 3:
				allSeedsDictionary[tokens[3].strip()] = [concereteNessScore,tokens[2].strip()];
			else:
				allSeedsDictionary[tokens[2].strip()] = [concereteNessScore,tokens[2].strip()];
	print("seeds_dict populated...");
	return allSeedsDictionary

'''
This method returns a directed acyclic graph based on the relative bias-score
of the nodes. 
i. TODO: Currently the bias-score is just centrality-score, TODO: change bias-score
to include prior-prob/application-bias.
ii. There should not be any transitive edges
'''
def populateEdgesRemoveTransitive(node_scores,bias_scores,assocSpace,detectionIndices):
	edges=[];
	bias_scores=sorted(bias_scores, key=lambda x: x[1],reverse=True)
	## Loop1: just try to put edges between (i,i+1) if it exceeds similarity.
	for i in range(0,len(bias_scores)):
		index_i = bias_scores[i][0];
		## We know at this point bias_score of node(index_i) > bias_score of node(index_j).
		## Without any other info, the graph will just be a chain.
		## HACK: We bring in similarities, if it is less than the median, then we donot include
		## such an edge.
		if i+1 < len(node_scores):
			_seed_i = detectionIndices[index_i];
			_seed_i1 = detectionIndices[bias_scores[i+1][0]];
			similarity = assocSpace.assoc_between_two_terms("/c/en/"+allSeedsDictionary[_seed_i][1],"/c/en/"+allSeedsDictionary[_seed_i1][1]);
			if exceedsThreshold(similarity):
				edges.append((index_i,bias_scores[i+1][0]));
	## Loop2: animal-> cute (not)-> dinosaur.
	## In above cases, we will lose information.
	## This loop, we will take 
	for j in range(len(bias_scores)-1,0,-1):
		index_j = bias_scores[j][0];
		index_jminus1 = bias_scores[j-1][0];
		edge = (index_jminus1,index_j);
		if edge not in edges:
			maxSimilarityAndNode=(-0.358846,-1);
			connected =False;
			for i in range(j-2,-1,-1):
				_seed_j = detectionIndices[index_j];
				_seed_i = detectionIndices[bias_scores[i][0]];
				similarity = assocSpace.assoc_between_two_terms("/c/en/"+allSeedsDictionary[_seed_i][1],"/c/en/"+allSeedsDictionary[_seed_j][1]);
				if exceedsThreshold(similarity):
					edges.append((bias_scores[i][0],index_j));
					connected = True;
					break;
				else:
					if similarity > maxSimilarityAndNode[0]:
						maxSimilarityAndNode = (similarity,bias_scores[i][0]);
			if not connected and maxSimilarityAndNode[1] != -1:
				edges.append((maxSimilarityAndNode[1],index_j));
	print(edges);
	return edges;

########################################################################
############# Start of Script
########################################################################		
if len(sys.argv) < 5:
	print("python ",sys.argv[0]," <seedsCentralityfile> <detectionsFolder> <target-name> <number-of-images>")
	sys.exit();

## Load Assoc-Space Matrix
assocDir="/windows/drive2/For PhD/KR Lab/UMD_vision_integration/Image_Riddle/conceptnet5/data/assoc/assoc-space-5.4";
assocSpace = AssocSpace.load_dir(assocDir);

allSeedsDictionary = populateModifiedSeedsAndConcreteScoreDictionary();

'''
Iterate over each image, create a network, optimize and 
write the outputs separately.
'''
for i in range(1,int(sys.argv[4])+1):
	detectionIndices = {}; ## detected seed -> index, index -> seed
	node_scores=[];  ## the initial confidences for each word.
	bias_scores=[];  ## the index,bias/centrality-scores for each word.
	lines = processClarifaiJsonFile(sys.argv[2]+sys.argv[3]+"_"+str(i)+".txt");
	
	detections = (lines[15][2:lines[15].index("]")]).split(",");
	weights = (lines[16][2:lines[16].index("]")]).split(",");
	print(detections)
	#detections = detectionLine.split(",");
	#weights = weightLine.split(",");
	for det_j in range(len(detections)):
		detection = detections[det_j].strip();
		newWeight = float(weights[det_j]);
		detectionIndices[detection]=len(node_scores);
		detectionIndices[len(node_scores)]=detection;
		node_scores.append(newWeight);
		bias_scores.append((detectionIndices[detection],allSeedsDictionary[detection][0]));
		
	print(detectionIndices);
	print(node_scores);			
	print("all detections populated for...",(sys.argv[2]+sys.argv[3]+"_"+str(i)+".txt"));

	## Create a DAG of the nodes based on the bias_scores, also the similarities
	## if too less, then donot include an edge.
	edges=populateEdgesRemoveTransitive(node_scores,bias_scores,assocSpace,detectionIndices);
		
	bounds = list(map(lambda v : (v/2,2.0), node_scores));
	result = optimize.minimize(sumOfMax, node_scores, args=(edges), method='L-BFGS-B');#bounds=bounds);
	print(result);
	outputFile = open("intermediateFiles/opt/opt_"+sys.argv[3]+"_"+str(i)+".txt","w");
	for seed_i in detectionIndices.keys():
		index_i = detectionIndices[seed_i];
		if seed_i in allSeedsDictionary.keys():
			print(index_i,"\t",seed_i,"\t",allSeedsDictionary[seed_i][0],"\t",result.x[index_i],file=outputFile);
