from __future__ import print_function

import conceptnet_util
import util
import numpy as np
import os
import sys
from numpy import linalg as LA
from gurobipy import *
			

def printSolution(m,variables,outputFile,detection_img,img_index,allSeedsDictionary):
	#if m.status == GRB.Status.OPTIMAL:
	seedsx = m.getAttr('x', variables)
	sumTotal = 0;
	for seed_i in detection_img:
		indexedSeed = seed_i+str(img_index);
		sumTotal = sumTotal+seedsx[indexedSeed];
	index_i=0;
	for seed_i in detection_img:
		indexedSeed = seed_i+str(img_index);
		normalizedWeight = seedsx[indexedSeed]/sumTotal;
		print(index_i,"\t",seed_i,"\t",allSeedsDictionary[seed_i][0],"\t",normalizedWeight,file=outputFile);
		index_i=index_i+1;

			
def processClarifaiJsonFile(fileName):
	with open(fileName, 'r') as myfile:
		line = myfile.read();
	line = line.replace("u\'", "");
	line = line.replace("\'", "");
	line = line.replace(":", "\n");
	lines = line.split("\n");
	return lines;
	
def getExpressionsForEachColumn(detection_img,wordvectors,variables,img_index):
	expressions = [];
	for i in range(wordvectors[0].shape[0]):
		expression= LinExpr();
		for j in range(len(detection_img)):
			expression+= variables[detection_img[j]+str(img_index)] * wordvectors[j][i];
		expressions.append(expression);
	return expressions;
	
def getInitialVector(weights_img,wordvectors,img_index):
	vector = np.zeros(wordvectors[0].shape[0]);
	for i in range(wordvectors[0].shape[0]):
		sumWeightVector = 0;
		for j in range(len(weights_img)):
			sumWeightVector = sumWeightVector+ weights_img[j] * wordvectors[j][i];
		vector[i]=sumWeightVector;
	return vector;
		
def getTheNormExpression(expressions_i):
	normExpression = QuadExpr();
	for expression in expressions_i:
		normExpression+= expression * expression;
	return normExpression

def addLowerBoundL1NormConstraints(m,expressions_i,img_i,l1norm,variables):
	index=0;
	for expression in expressions_i:
		name= str(index)+"_"+str(img_i);
		variables[name] = m.addVar(name=name);
		index=index+1;
	m.update();
	constraint= LinExpr();
	index=0;
	for expression in expressions_i:
		name= str(index)+"_"+str(img_i);
		m.addConstr(expression,GRB.LESS_EQUAL,variables[name]);
		m.addConstr(-variables[name],GRB.LESS_EQUAL, expression);
		constraint += variables[name];
		index=index+1;
	
	m.addConstr(constraint,GRB.LESS_EQUAL,l1norm);
	m.addConstr(constraint,GRB.GREATER_EQUAL,l1norm);
	
	#m.addConstr(constraint,GRB.LESS_EQUAL,l1norm-1);
	#m.addConstr(-1*constraint,GRB.GREATER_EQUAL,l1norm-1);
	
	
		
'''
Iterate over each image, for each word- get the average similarity to the other
image-vectors.
'''	
def reorderWeightsBasedOnCloseness(allSeedsDictionary,detectionFolder,imagePrefix,\
inferenceFolder="intermediateFiles/opt_test/"):
	outputFileNames=[];
	
	detection_all_images={};
	weights_all_images={};
	wordvectors_all_images={};
	### Load all 4 image-vectors
	for img in range(1,5):
		lines = processClarifaiJsonFile(detectionFolder+imagePrefix+"_"+str(img)+".txt");
	
		detections = (lines[15][2:lines[15].index("]")]).split(",");
		weights = (lines[16][2:lines[16].index("]")]).split(",");
		weights = map(lambda x: float(x),weights);
		detections = map(lambda x: x.strip(),detections);
		detection_all_images[img] = detections;
		weights_all_images[img] = weights;
		wordvectors=[]
		for detection in detections:
			wordvectors.append(conceptnet_util.getWord2VecVector(detection));
		wordvectors_all_images[img]=wordvectors;
	
	print("\tall word-vectors updated..")
	# Model
	m = Model("organizingNetwork"+str(imagePrefix));
	if not VERBOSE:
		setParam('OutputFlag', 0);
		
	## Add all the seed variables as weights to the word2vec vectors.
	variables={};
	for img_i in range(1,5):
		detections_img = detection_all_images[img_i];
		weights_img = weights_all_images[img_i];
		for j in range(len(detections_img)):
			varname= detections_img[j]+str(img_i);
			variables[varname] = m.addVar(lb=0.49, ub=weights_img[j]+weights_img[j]/2, name=varname);
	
	m.update();
	expressions_ ={};
	norms_={};
	for img_i in range(1,5):
		wordvectors = wordvectors_all_images[img_i];
		expressions = getExpressionsForEachColumn(detection_all_images[img_i],wordvectors,variables,img_i);
		expressions_[img_i] = expressions;
		vector = getInitialVector(weights_all_images[img_i],wordvectors,img_i);
		norm = LA.norm(vector);
		l1norm = LA.norm(vector,1);
		norms_[img_i] = (norm*norm,l1norm);
	
	## now add the objective
	objective = QuadExpr();
	for img_i in range(1,5):
		expressions_i = expressions_[img_i];
		normExpression = getTheNormExpression(expressions_i);
		m.addQConstr(normExpression,GRB.LESS_EQUAL,norms_[img_i][0]);
		addLowerBoundL1NormConstraints(m,expressions_i,img_i,norms_[img_i][1],variables);
		for img_j in range(img_i+1,5):
			expressions_j = expressions_[img_j];
			for k in range(len(expressions_j)):
				objective+= (expressions_i[k]-expressions_j[k]) * (expressions_i[k]-expressions_j[k]);
	
	m.setObjective(objective);
	# The objective is to minimize the costs
	m.modelSense = GRB.MINIMIZE

	# Update model to integrate new variables
	m.update()
	m.optimize();
	m.write('organizingNetwork'+imagePrefix+'.lp');
	m.write('organizingNetwork'+imagePrefix+'.sol');
	for img_i in range(1,5):
		outputFile = open(inferenceFolder+"opt_"+imagePrefix+"_"+str(img_i)+"_c.txt","w");
		printSolution(m,variables,outputFile,detection_all_images[img_i],img_i,allSeedsDictionary)
		outputFileNames.append(outputFile.name);
	if VERBOSE:
		m.printAttr('x');
		print('\nDistance to Satisfaction: %g' % m.objVal)
	return outputFileNames


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
