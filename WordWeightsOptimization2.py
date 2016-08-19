from __future__ import print_function

import conceptnet_util
import util
import collections
from gurobipy import *
import numpy as np
import os
import sys

#######################################################################
########		PIPELINE STAGE I.
########		Input: seeds+weights
########		Output: Reordered collapsed seeds
#######################################################################
'''
This script will first read the seed-> (centrality-score,seed_in_conceptnet) mapping.
Then it will take only 1 seed detection file.
The network of the words are separately optimized.
the output will be one optimized list of words and its weights.
'''

def printSolution(m,variables,outputFile,detectionIndices,allSeedsDictionary):
    #if m.status == GRB.Status.OPTIMAL:
    if VERBOSE:
		m.printAttr('x');
		print('\nDistance to Satisfaction: %g' % m.objVal)
		print('\nTargets:')
    seedsx = m.getAttr('x', variables)
    sumTotal = 0;
    for seed_i in seedsx:
		sumTotal = sumTotal+seedsx[seed_i];
    for seed_i in seedsx:
		index_i = detectionIndices[seed_i];
		normalizedWeight = seedsx[seed_i]/sumTotal;
		print(index_i,"\t",seed_i,"\t",allSeedsDictionary[seed_i][0],"\t",normalizedWeight,file=outputFile);


def mergeBasedOnFunction(newWeight,detection,detectionIndices,node_scores,function):
	index = detectionIndices[detection];
	oldWeight = node_scores[index];
	if function=="max":
		node_scores[index] = max(newWeight,oldWeight);
	elif function=="min":
		node_scores[index] = min(newWeight,oldWeight);
	elif function=="avg":
		node_scores[index] = newWeight+oldWeight;

########
## ASSUMPTION 1: similarity Threshold = 0.5 
########
def exceedsThreshold(similarity):
	minV=-1;
	maxV= 1;
	value= (similarity-minV)/(maxV-minV);
	if value > util.SIMILARITY_THRESHOLD_WORDWEIGHTS:
		return True;
	else:
		return False;



'''
This method returns a directed acyclic graph based on the relative bias-score
of the nodes. 
i. TODO: Currently the bias-score is just centrality-score, change bias-score
to include prior-prob/application-bias.
ii. There should not be any transitive edges
'''
def populateEdgesRemoveTransitive(allSeedsDictionary,node_scores,bias_scores,detectionIndices):
	edges=[];
	participants=set();
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
			similarity = conceptnet_util.assocSpace.assoc_between_two_terms("/c/en/"+\
			allSeedsDictionary[_seed_i][1],"/c/en/"+allSeedsDictionary[_seed_i1][1]);
			if exceedsThreshold(similarity):
				edges.append((index_i,bias_scores[i+1][0]));
				participants.add(index_i);
				participants.add(bias_scores[i+1][0]);
	
	## Loop2: animal-> cute (not)-> dinosaur.
	## In above cases, we will lose information.
	## This loop, we will take 
	for j in range(len(bias_scores)-1,0,-1):
		index_j = bias_scores[j][0];
		index_jminus1 = bias_scores[j-1][0];
		edge = (index_jminus1,index_j);
		if index_j not in participants:
			#if edge not in edges:
			maxSimilarityAndNode=(-0.358846,-1);
			connected =False;
			for i in range(j-2,-1,-1):
				_seed_j = detectionIndices[index_j];
				_seed_i = detectionIndices[bias_scores[i][0]];
				similarity = conceptnet_util.assocSpace.assoc_between_two_terms("/c/en/"+\
				allSeedsDictionary[_seed_i][1],"/c/en/"+allSeedsDictionary[_seed_j][1]);
				if exceedsThreshold(similarity):
					edges.append((bias_scores[i][0],index_j));
					participants.add(bias_scores[i][0]);
					participants.add(index_j);
					connected = True;
					break;
				else:
					if similarity > maxSimilarityAndNode[0]:
						maxSimilarityAndNode = (similarity,bias_scores[i][0]);
			if (not connected) and maxSimilarityAndNode[1] != -1 and exceedsThreshold(maxSimilarityAndNode[1]):
				edges.append((maxSimilarityAndNode[1],index_j));
				participants.add(maxSimilarityAndNode[1]);
				participants.add(index_j);
	if VERBOSE:
		print(edges);
	return [edges,participants];



'''
Iterate over each image, create a network, optimize and 
write the outputs separately.
NOTE: Currently not using collpaseHypernyms here.
'''	
def reorderWeightsBasedOnPopularity(allSeedsDictionary,detectionFolder,imagePrefix,numberOfImages,startRange=1,\
inferenceFolder="intermediateFiles/opt_test/", apiUsed="clarifai"):
	outputFileNames=[]
	print(util.SIMILARITY_THRESHOLD_WORDWEIGHTS);
	for img in range(startRange,int(numberOfImages)+1):
		detectionIndices = {}; ## detected seed -> index, index -> seed
		node_scores=[];  ## the initial confidences for each word.
		bias_scores=[];  ## the index,bias/centrality-scores for each word.
		if apiUsed == "clarifai":
			[detections, weights] = util.processClarifaiJsonFile(detectionFolder+imagePrefix+"_"+str(img)+".txt");
		else:
			[detections, weights] = util.getDetectionsFromTSVFile(detectionFolder+imagePrefix+"_"+str(img)+".txt");
			detections = conceptnet_util.chooseSingleRepresentativeDetection(allSeedsDictionary, detections);
		
		if VERBOSE:
			print(detections)
	
		#[detections,weights] = conceptnet_util.collapseSuperClasses(detections,weights,allSeedsDictionary);
	
		if VERBOSE:
			print("after collapsing...\n")
			print(detections);
		#detections = detectionLine.split(",");
		#weights = weightLine.split(",");
		for det_j in range(len(detections)):
			detection = detections[det_j].strip();
			newWeight = float(weights[det_j]);
			## if seed is not in conceptnet-just ignore
			if detection not in allSeedsDictionary.keys():
				continue;
			detectionIndices[detection]=len(node_scores);
			detectionIndices[len(node_scores)]=detection;
			node_scores.append(newWeight);
			bias_scores.append((detectionIndices[detection],allSeedsDictionary[detection][0]));
		
		if VERBOSE:
			print(detectionIndices);
			print(node_scores);	
			print("all detections populated for...",(detectionFolder+imagePrefix+"_"+str(img)+".txt"));

		## Create a DAG of the nodes based on the bias_scores, also the similarities
		## if too less, then donot include an edge.
		[edges,participants]=populateEdgesRemoveTransitive(allSeedsDictionary,node_scores,bias_scores,detectionIndices);
		
		# Model
		m = Model("organizingNetwork"+str(img));
		if not VERBOSE:
			setParam('OutputFlag', 0);
		variables={};
		initialSumOfScores=0;
		for i in range(len(node_scores)):
			if i in participants:
				variables[detectionIndices[i]] = m.addVar(lb=0.49, ub=node_scores[i]+node_scores[i]/2, name=detectionIndices[i]);
			else:
				variables[detectionIndices[i]] = m.addVar(lb=node_scores[i], ub=node_scores[i], name=detectionIndices[i]);
			initialSumOfScores += node_scores[i];
		m.update();
	
		constraint = LinExpr();
		for v in variables.keys():
			constraint += variables[v];
		m.addConstr(constraint,GRB.GREATER_EQUAL,initialSumOfScores-2);
		m.addConstr(constraint,GRB.LESS_EQUAL,initialSumOfScores+2);
		## Create the Objective
		objective = LinExpr();
	
		for edge in edges:
			src_node_var = variables[detectionIndices[edge[0]]];
			dest_node_var = variables[detectionIndices[edge[1]]];
			ruleVar = detectionIndices[edge[0]]+"_"+detectionIndices[edge[1]];
			ruleVariable = m.addVar(lb=0, ub=100.0, name=ruleVar);
			objective+= ruleVariable;
			m.update(); 
			m.addConstr(ruleVariable,GRB.GREATER_EQUAL,0);
			m.addConstr(ruleVariable,GRB.GREATER_EQUAL,2*src_node_var-dest_node_var);
	
		m.setObjective(objective);
		# The objective is to minimize the costs
		m.modelSense = GRB.MINIMIZE

		# Update model to integrate new variables
		m.update()
		### TODO: try using model.tune()
		m.optimize();
		#m.write('organizingNetwork.lp');
		#m.write('organizingNetwork.sol');
	
		outputFile = open(inferenceFolder+"opt_"+imagePrefix+"_"+str(img)+"_c.txt","w");
		printSolution(m,variables,outputFile,detectionIndices,allSeedsDictionary)
		outputFileNames.append(outputFile.name);
		outputFile.close();
	return outputFileNames;

########################################################################
############# Start of Script
########################################################################		
if __name__ == "__main__":
	if len(sys.argv) < 5:
		print("python ",sys.argv[0]," <seedsCentralityfile> <detectionsFolder> <target-name> <number-of-images>")
		sys.exit();

	VERBOSE = True;
	util.SIMILARITY_THRESHOLD_WORDWEIGHTS=0.9;
	allSeedsDictionary = util.populateModifiedSeedsAndConcreteScoreDictionary(sys.argv[1]);
	reorderWeightsBasedOnPopularity(allSeedsDictionary,sys.argv[2],sys.argv[3],sys.argv[4]);

