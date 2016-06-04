from __future__ import print_function

import mergeTargets
import clusterTargets
import sys
from gurobipy import *
from util import *
import conceptnet_util
import numpy as np

'''
#######################################################################
########		PIPELINE STAGE V. (TODO: test it)
########		Input: seeds, sorted target-matrix, target-clusters
########		Output: ranked targets
########		Assumption: 
					Here we collapse the hypernyms
					animal-> reptile-> dino => dino
#######################################################################
'''

def printSolution(m,targets,outputFile,targetsToCentralities):
    #if m.status == GRB.Status.OPTIMAL:
    m.printAttr('x');
    print('\nDistance to Satisfaction: %g' % m.objVal)
    print('\nTargets:')
    targetsx = m.getAttr('x', targets)
    for t in targetsx:
		if targets[t].x > 0.0001:
			print('%s\t%g\t%g' % (t, targetsx[t],targetsToCentralities[t]),file=outputFile);


#####################################################################
#### Create decision variables for the seeds
#### Limit them using constraints
#####################################################################	
def createDecisionVariableSeeds(m,seeds,variables,seedsDetected_weights,seedHypernymFilterSet):
	for c in seedsDetected_weights.keys():
		## if seed is a direct hypernym, dont add.
		if c not in seedHypernymFilterSet:
			continue;
		## add arbit string to distinguish from targets
		c1 =  c+"1";
		if c1 in variables:
			continue;
		seeds[c1] = m.addVar(lb=seedsDetected_weights[c], ub=seedsDetected_weights[c], name=c1);
		variables.add(c1);



#####################################################################
#### Create variables for the targets
#####################################################################	
def loadAllTargetsCNet(sortedScoreAndIndexList, targetWordsList, targetWordsDictonary,\
targetsToCentralities,variables,targets,m):
	for i in range(0,len(sortedScoreAndIndexList)):
		if i > 2000:
			break;
		
		### Example: sortedScoreAndIndexList: ([1,0.22), (0,0.19), (2,0.11)]
		indexAndScore = sortedScoreAndIndexList[i];
		targetWord = targetWordsList[indexAndScore[0]];
		
		targetsToCentralities[targetWord] = conceptnet_util.getCentralityScore(targetWord,True);
		targets[targetWord] = m.addVar(lb=0.0, ub=1.0, name=targetWord);
		variables.add(targetWord);
	m.update();


def createclusterDependencies(globalWeightSum,tupleOfObjectives,clusterAssignmentsOfTargets,\
pairwiseDistancesTargetWords,clusterVariables,m):
	
	numberOfClusters=clusterAssignmentsOfTargets.max();
	clusterMembers={};
	for target_i in range(0,len(clusterAssignmentsOfTargets)):
		assignment_i = clusterAssignmentsOfTargets[target_i];
		if assignment_i not in clusterMembers.keys():
			clusterMembers[assignment_i]=[];
		clusterMembers[assignment_i].append(target_i);
	
	##################################################################
	########## Represent cluster similarities
	##################################################################
	tuplesOfConstraints=[];
	clusterPairwiseSimilarity=np.zeros((numberOfClusters+1,numberOfClusters+1));
	for cl_i in range(1,numberOfClusters+1):
		rows_i = clusterMembers[cl_i];
		for cl_j in range(cl_i+1,numberOfClusters+1):
			columns_j = clusterMembers[cl_j];
			minDistance =1;
			for row_i in rows_i:
				distances_row_i = map(lambda x:pairwiseDistancesTargetWords[row_i,x],columns_j);  
				minForRow_i = min(distances_row_i);
				if minForRow_i < minDistance:
					minDistance = minForRow_i;
			clusterPairwiseSimilarity[cl_i,cl_j]= 1-minDistance;
		
		## sorted is increasing order
		sortedIndices = np.argsort(clusterPairwiseSimilarity[cl_i,:]);
		
		## traverse sortedIndices from end, get 3 max clusters
		for i in range(len(sortedIndices)-1,len(sortedIndices)-3,-1):
			cl_j = sortedIndices[i];
			maxSimilarity = clusterPairwiseSimilarity[cl_i,cl_j];
			if maxSimilarity > 0.7:
				ruleVar1 = "cluster"+str(cl_i)+"_"+"cluster"+str(cl_j);
				ruleVariable1 = m.addVar(name=ruleVar1);
				weight = maxSimilarity;
				globalWeightSum = globalWeightSum+weight;
				tupleOfObjectives.append((ruleVariable1,weight));
				tuplesOfConstraints.append((ruleVariable1,"cluster"+str(cl_i),"cluster"+str(cl_j)))
				
				ruleVar2 = "cluster"+str(cl_j)+"_"+"cluster"+str(cl_i);
				ruleVariable2 = m.addVar(name=ruleVar2);
				weight = maxSimilarity;
				globalWeightSum = globalWeightSum+weight;
				tupleOfObjectives.append((ruleVariable2,weight));
				tuplesOfConstraints.append((ruleVariable2,"cluster"+str(cl_j),"cluster"+str(cl_i)));
	m.update();
	
	for tupleC in tuplesOfConstraints:		
		#max(clusterVariables[cluster1]-clusterVariables[cluster2],0);
		m.addConstr(tupleC[0],GRB.GREATER_EQUAL,0);
		m.addConstr(tupleC[0],GRB.GREATER_EQUAL,clusterVariables[tupleC[1]]-clusterVariables[tupleC[2]]);
		
	return [globalWeightSum,tupleOfObjectives];



'''
########################################################################
############# Create PSL objective function
############# (I)	wt: word_i1 ^ word_i2 -> target_j
############# 		For all i1,i2 for which target_j was among top 2k
#############		wt = avg(sim_i1,sim_i2) + 2/centrality_j
############# (II)	wt: word1 -> target_j
#############		For all words which have similarity > 0.5
#############		wt = similarity + 2/centrality_j
		for each cluster cluster_i, take the most similar N clusters.
			add wt: cluster_i -> cluster_j
				wt: cluster_j -> cluster_i
			forces them to be similar
		ASSUMPTION/Heuristic:
				i. 2/centrality_j = compensates the large similarities for 
				abstract terms
				ii. avg(sim_i1,sim_i2) instead of avg-vector similarity.
########################################################################
'''
def createPSLBasedObjectiveFunction(m,objective,variables,seeds,targets,targetsToCentralities,\
seedHypernymFilterSet, seedsDetected_weights, orderedSeedWordsList, \
sortedScoreAndIndexList, targetWordsList, targetWordsDictonary, \
clusterAssignmentsOfTargets, pairwiseDistancesTargetWords, clusterVariables):
	
	globalWeightSum = 0;
	tupleOfObjectives=[];
	turnOn2WordRules = False;
	
	[globalWeightSum,tupleOfObjectives] = createclusterDependencies(globalWeightSum,tupleOfObjectives,\
	clusterAssignmentsOfTargets,pairwiseDistancesTargetWords,clusterVariables,m);
				
	# Iterate over the targets
	for indexTarget in range(0,len(sortedScoreAndIndexList)):
		if indexTarget > 2000:
			break; 
		indexAndScore = sortedScoreAndIndexList[indexTarget];
		targetWord = targetWordsList[indexAndScore[0]];
		comb_similarity_list = targetWordsDictonary[targetWord];
		nonzeroSeedIndices = np.nonzero(comb_similarity_list)[0];
		
		#######################################################
		####### n=#seeds, Get all nC2 rules for each target.
		####### For this, just choose all the column for each target
		####### and check non-zero indices from the mergeTarget matrix.
		#######################################################
		
		setOfSeedsIn2WordRules =set();
		centralityOfTarget = targetsToCentralities[targetWord];
		tuplesOfConstraints=[];
		if turnOn2WordRules:
			for k in range(0,len(nonzeroSeedIndices)):
				seedWord1 = orderedSeedWordsList[nonzeroSeedIndices[k]];
				similarity_word1 = comb_similarity_list[nonzeroSeedIndices[k]];
				# Hypernym, then dont consider
				if seedWord1 not in seedHypernymFilterSet:
					continue;
			
				for l in range(k+1,len(nonzeroSeedIndices)):
					seedWord2 = orderedSeedWordsList[nonzeroSeedIndices[l]];	
					similarity_word2 = comb_similarity_list[nonzeroSeedIndices[l]];
					# Hypernym, then dont consider
					if seedWord2 not in seedHypernymFilterSet:
						continue;
				
					## Using average Combined Similarity
					similarity = (similarity_word1+similarity_word2)/2;
					if similarity > 0.4:
						seedWord1_1=seedWord1+"1";
						seedWord2_1=seedWord2+"1";
						ruleVar = targetWord+"_"+seedWord1_1+"_"+seedWord2_1;
						ruleVariable = m.addVar(name=ruleVar);
						weight = similarity+1.0/centralityOfTarget;
						globalWeightSum = globalWeightSum+weight;
						tupleOfObjectives.append((ruleVariable,weight));
						#objective+= weight*ruleVariable; 
						
						tuplesOfConstraints.append((ruleVariable,seedWord1_1,seedWord2_1,targetWord));
						## book-keeping
						setOfSeedsIn2WordRules.add(seedWord1);
						setOfSeedsIn2WordRules.add(seedWord1);
						
			m.update();
		
			#### Add all the constraints in one shot
			for tupleC in tuplesOfConstraints:
				ruleVariable=tupleC[0];
				seedWord1_1=tupleC[1];
				seedWord2_1=tupleC[2];
				targetWord=tupleC[3];
				### TRICK: Use averaging conjunction
				m.addConstr(ruleVariable,GRB.GREATER_EQUAL,0);
				m.addConstr(ruleVariable,GRB.GREATER_EQUAL,(seeds[seedWord1_1]+\
				seeds[seedWord2_1])/2-targets[targetWord]);
		
		#######################################################
		####### n=#seeds, Get all n rules for each target.
		####### For all seeds, for which similarity exceeds a threshold
		#######################################################	
		tuplesOfConstraints=[];
		for seedIndex in range(len(orderedSeedWordsList)):
			seed = orderedSeedWordsList[seedIndex];
			sim_word1 = comb_similarity_list[seedIndex];
			# Hypernym, then dont consider
			if seed not in seedHypernymFilterSet:
					continue;
			
			#if sim_word1 > 0.4 and (seed not in setOfSeedsIn2WordRules):
			if sim_word1 > 0 and (seed not in setOfSeedsIn2WordRules):	
				seed1= seed+"1";
				ruleVar = targetWord+"_"+seed1;
				ruleVariable = m.addVar(name=ruleVar);
				## TODO: use normalized value for centrality
				penaltyForPopularity = computeNormalizedValue(1.0/centralityOfTarget,3.0,0.0);
				weight = sim_word1+penaltyForPopularity;#1.0/centralityOfTarget;
				#objective+= weight*ruleVariable; 
				globalWeightSum = globalWeightSum+weight;
				tupleOfObjectives.append((ruleVariable,weight));
				
				tuplesOfConstraints.append((ruleVariable,seed1,targetWord));
		
		m.update();
		#### Add all the constraints in one shot
		for tupleC in tuplesOfConstraints:		
			#max(seeds[seed1]-targets[target1],0);
			m.addConstr(tupleC[0],GRB.GREATER_EQUAL,0);
			m.addConstr(tupleC[0],GRB.GREATER_EQUAL,seeds[tupleC[1]]-targets[tupleC[2]]);
		
		########## Add cluster constraints
		assignment = clusterAssignmentsOfTargets[indexTarget];
		m.addConstr(targets[targetWord],GRB.LESS_EQUAL,clusterVariables["cluster"+str(assignment)]);
	
	################################################
	###### IMPORTANT: Normalize the weights
	###### before creating the objective function
	################################################
	for tupleO in tupleOfObjectives:
		weight = tupleO[1];#/globalWeightSum;
		objective+= weight*tupleO[0];
	
	return objective;



'''
	################################################################
	Creates necessary variables and constraints.
	Constraints:
		Sum (cluster_i) <= K
		t_j < cluster_i , if t_j is in cluster_i [done in createPSLBasedObjectiveFunction]
	################################################################
'''
def optimizeAndInferConceptsModelOneNew(allSeedsDictionary,seedsDetected_weights,orderedSeedWordsList,reweightedSeedsFile,\
sortedScoreAndIndexList, targetWordsList, targetWordsDictonary, clusters, pairwiseDistancesTargetWords):
	
	seedHypernymFilterSet = seedsDetected_weights.keys();
	#conceptnet_util.getHypernymFilter(seedsDetected_weights,\
	#allSeedsDictionary);
	
	print(seedHypernymFilterSet);
	# Model
	m = Model("psl1");
	variables= set();
	seeds = {}
	createDecisionVariableSeeds(m,seeds,variables,seedsDetected_weights,seedHypernymFilterSet);
	targets = {}
	targetsToCentralities={};
	loadAllTargetsCNet(sortedScoreAndIndexList, targetWordsList, targetWordsDictonary,\
	targetsToCentralities,variables,targets,m);
	
	numberOfClusters = clusters.max();
	clusterVariables={};
	for i in range(1,numberOfClusters+1):
		c1 = "cluster"+str(i);
		clusterVariables[c1] = m.addVar(name=c1);
		variables.add(c1);
	m.update();
	
	############# limit Sum_i I(cluster_i) = #targets_i_want*2
	m.addConstr(quicksum(clusterVariables["cluster"+str(i)] for i in range(1,numberOfClusters+1)),GRB.LESS_EQUAL,10.0);
	m.addConstr(quicksum(targets[c1] for c1 in targets),GRB.LESS_EQUAL,20.0);
	objective = LinExpr();
	
	######### create PSL based objective function ###########
	objective = createPSLBasedObjectiveFunction(m,objective,variables,seeds,targets,targetsToCentralities,\
	seedHypernymFilterSet, seedsDetected_weights, orderedSeedWordsList, \
	sortedScoreAndIndexList, targetWordsList, targetWordsDictonary,\
	clusters, pairwiseDistancesTargetWords, clusterVariables);
	print("\tall model constraints updated..");
	
	m.setObjective(objective);
	# The objective is to minimize the costs
	m.modelSense = GRB.MINIMIZE

	# Update model to integrate new variables
	m.update()
	m.optimize();
	
	detectedSeedsFileName = reweightedSeedsFile[reweightedSeedsFile.rindex("/")+1:reweightedSeedsFile.index(".")];
	m.write('optimization_output/out'+detectedSeedsFileName+'.lp');
	m.write('optimization_output/out'+detectedSeedsFileName+'.sol');
	
	filePrefix = reweightedSeedsFile[:reweightedSeedsFile.index(".")];
	outputFile = open(filePrefix+"_inf.txt","w");
	printSolution(m,targets,outputFile,targetsToCentralities);


########################################################################
############# Start of Script
########################################################################
if __name__ == "__main__":
	# sys.arv[1] = reweighted set of seeds for an image
	# sys.argv[2] = detected seeds to modified map file
	# sys.argv[3] = targets for each seed
	reweightedSeedsFile = sys.argv[1];
	
	#### Step 1: Merge targets from different seeds.
	[sortedScoreAndIndexList, targetWordsList, targetWordsDictonary,seedsDetected_weights,\
	orderedSeedWordsList,allSeedsDictionary] = mergeTargets.mergeTargetsFromDetectedSeeds(\
	reweightedSeedsFile, sys.argv[2]);
	print("\tmerging targets completed..");
	
	#### Step 2: cluster the merged set of targets
	[clusters,sortedScoreAndIndexList,Z,pairwiseDistancesTargetWords] = clusterTargets.returnClusters(sortedScoreAndIndexList, \
	targetWordsList, targetWordsDictonary, orderedSeedWordsList, 3000);
	print(pairwiseDistancesTargetWords.shape);
	print("\tclustering targets completed..");
	
	#### Step 3: create 1-word and 2-word model
	optimizeAndInferConceptsModelOneNew(allSeedsDictionary,seedsDetected_weights,orderedSeedWordsList,reweightedSeedsFile,\
sortedScoreAndIndexList, targetWordsList, targetWordsDictonary, clusters, pairwiseDistancesTargetWords);
