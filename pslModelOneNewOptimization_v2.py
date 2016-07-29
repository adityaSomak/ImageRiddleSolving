from __future__ import print_function

import mergeTargets
import clusterTargets
import sys
from gurobipy import *
import util
import conceptnet_util
import numpy as np

'''
#######################################################################
########		PIPELINE STAGE V.
########		Input: seeds, sorted target-matrix, target-clusters
########		Output: ranked targets
#######################################################################
'''

def printSolution(m,targets,outputFile,targetsToCentralities):
	#if m.status == GRB.Status.OPTIMAL:
	#m.printAttr('x');
	if VERBOSE:
		print('\nDistance to Satisfaction: %g' % m.objVal)
		print('\nTargets:')
	targetsx = m.getAttr('x', targets)
	targetsList=[];
	for t in targetsx:
		if targets[t].x > 0.0001:
			targetsList.append((t, targetsx[t],targetsToCentralities[t]));
	#print('%s\t%g\t%g' % (t, targetsx[t],targetsToCentralities[t]),file=outputFile);
	targetsList = sorted(targetsList, key=lambda tup: tup[1], reverse=True);
	for tup in targetsList:
		print('%s\t%g\t%g' % (tup[0], tup[1],tup[2]),file=outputFile);


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
#### Load centralities
#####################################################################	
def loadAllTargetsCNet(sortedScoreAndIndexList, targetWordsList, targetWordsDictonary,\
targetsToCentralities,variables,targets,m):
	for i in range(0,len(sortedScoreAndIndexList)):
		if i > util.MAX_TARGETS_PSL_ONE:
			break;
		
		### Example: sortedScoreAndIndexList: ([1,0.22), (0,0.19), (2,0.11)]
		indexAndScore = sortedScoreAndIndexList[i];
		targetWord = targetWordsList[indexAndScore[0]];
		
		targetsToCentralities[targetWord] = conceptnet_util.getCentralityScore(targetWord,True);
		targets[targetWord] = m.addVar(lb=util.TARGET_LOWER_BOUND_PSL_ONE, ub=1.0, name=targetWord,vtype=GRB.SEMICONT);
		variables.add(targetWord);
	m.update();


'''
########################################################################
############# Create PSL objective function
############# (I)	wt: word1 -> target_j
#############		For all words which have similarity > 0.5
#############		wt = similarity + 2/centrality_j
		ASSUMPTION/Heuristic:
				i. normalize(1/centrality_j) = normalize to 0-1
				ii. Targets considered=2000
				iii. similar targets (t_i <==> t_j) <= 5
########################################################################
'''
def createPSLBasedObjectiveFunction(m,objective,variables,seeds,targets,targetsToCentralities,\
seedHypernymFilterSet, seedsDetected_weights, orderedSeedWordsList, \
sortedScoreAndIndexList, targetWordsList, targetWordsDictonary, \
pairwiseDistancesTargetWords):
	
	globalWeightSum = 0;
	tupleOfObjectives=[];
	turnOn2WordRules = False;
				
	# Iterate over the targets
	for indexTarget in range(0,len(sortedScoreAndIndexList)):
		if indexTarget > util.MAX_TARGETS_PSL_ONE:
			break; 
		indexAndScore = sortedScoreAndIndexList[indexTarget];
		targetWord = targetWordsList[indexAndScore[0]];
		comb_similarity_list = targetWordsDictonary[targetWord];
		nonzeroSeedIndices = np.nonzero(comb_similarity_list)[0];
		
		centralityOfTarget = targetsToCentralities[targetWord];
		tuplesOfConstraints=[];
		
		word2vecKeyFoundCode = conceptnet_util.getWord2VecKeyFoundCode(targetWord);
		if word2vecKeyFoundCode == conceptnet_util.TOO_RARE_WORD_CODE:
			m.addConstr(targets[targetWord],GRB.LESS_EQUAL,0);
			continue;
		
		'''
		###########################################################
		######  Get top (500) similar other targets.
		######		 if the top targets are on the current 
		######		 suggested_targets list=> 
					 add wt: target_i -> target_j
						 wt: target_j -> target_i
						forces them to be similar
		###########################################################
		'''
		
		similarIndicesList = np.argsort(pairwiseDistancesTargetWords[indexTarget,:]);
		topk=0;
		tuplesOfConstraints=[];
		for topIndex in similarIndicesList:
			
			indexOfSimilarTerm = sortedScoreAndIndexList[topIndex][0];
			term = targetWordsList[indexOfSimilarTerm];
			## normalize u.v/|u|.|v|
			word2vecKeyFoundCode = conceptnet_util.getWord2VecKeyFoundCode(term);
			if word2vecKeyFoundCode==conceptnet_util.TOO_RARE_WORD_CODE:
				continue;
			
			word2vecsimilarity = conceptnet_util.getWord2VecSimilarity(targetWord,term,True);
			if word2vecsimilarity == conceptnet_util.NOT_FOUND_IN_CORPUS_CODE:
				similarity = util.computeNormalizedValue(1-\
				pairwiseDistancesTargetWords[indexTarget,topIndex],1,-1)
			else:
				similarity = util.computeNormalizedValue(1-\
				pairwiseDistancesTargetWords[indexTarget,topIndex],1,-1)+word2vecsimilarity;
				similarity = similarity/2.0;
			
			ruleVar1 = targetWord+"_"+term;
			ruleVar2 = term+"_"+targetWord;
			
			if similarity <= 0.92 or topk >= util.TOP_K_SIMILAR_TARGETS_PSL_ONE:
				#if VERBOSE:
				#	print('%s not-matches %s:%g' % (targetWord,term,similarity));
				break;
			
			if (term != targetWord) and (term in targets) and (ruleVar1 not in variables):
				weight = similarity;
				
				ruleVariable1 = m.addVar(name=ruleVar1);
				globalWeightSum = globalWeightSum+weight;
				tupleOfObjectives.append((ruleVariable1,weight));
				tuplesOfConstraints.append((ruleVariable1,targetWord,term));
				variables.add(ruleVar1);
				
				ruleVariable2 = m.addVar(name=ruleVar2);
				globalWeightSum = globalWeightSum+weight;
				tupleOfObjectives.append((ruleVariable2,weight));
				tuplesOfConstraints.append((ruleVariable2,term,targetWord));
				variables.add(ruleVar2);
				topk = topk+1;
				#if VERBOSE:
				#	print('%s matches %s:%g' % (targetWord,term,similarity));
		
		m.update();
		#### Add all the constraints in one shot
		for tupleC in tuplesOfConstraints:		
			#max(targets[target1]-targets[target2],0);
			m.addConstr(tupleC[0],GRB.GREATER_EQUAL,0);
			m.addConstr(tupleC[0],GRB.GREATER_EQUAL,targets[tupleC[1]]-targets[tupleC[2]]);
		
		#######################################################
		####### n=#seeds, Get all n rules for each target.
		####### For all seeds, for which similarity exceeds a threshold
		#######################################################	
		tuplesOfConstraints=[];
		for seedIndex in range(len(orderedSeedWordsList)):
			seed = orderedSeedWordsList[seedIndex];
			sim_word1 = comb_similarity_list[seedIndex];
			word2vecsimilarity = conceptnet_util.getWord2VecSimilarity(targetWord,seed,True);
			if word2vecsimilarity == conceptnet_util.NOT_FOUND_IN_CORPUS_CODE:
				similarity = sim_word1;
			else:
				similarity= (sim_word1*util.CONCEPTNET_SIMILARITY_WEIGHT+\
				word2vecsimilarity*util.WORD2VEC_SIMILARITY_WEIGHT)/\
				(util.CONCEPTNET_SIMILARITY_WEIGHT+util.WORD2VEC_SIMILARITY_WEIGHT);
			# Hypernym, then dont consider
			if seed not in seedHypernymFilterSet:
				continue;
			
			if sim_word1 > util.SIMILARITY_THRESHOLD_ONEWORDRULE_PSL_ONE:	
				seed1= seed+"1";
				ruleVar = targetWord+"_"+seed1;
				ruleVariable = m.addVar(name=ruleVar);
				## use normalized value for centrality
				penaltyForPopularity = 0.5 * util.computeNormalizedValue(1.0/centralityOfTarget,3.0,0.0);
				
				weight = similarity+penaltyForPopularity;#1.0/centralityOfTarget;
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
sortedScoreAndIndexList, targetWordsList, targetWordsDictonary, pairwiseDistancesTargetWords):
	
	seedHypernymFilterSet = seedsDetected_weights.keys();
	#conceptnet_util.getHypernymFilter(seedsDetected_weights,\
	#allSeedsDictionary);
	
	if VERBOSE:
		print(seedHypernymFilterSet);
	# Model
	m = Model("psl1");
	m.setParam(GRB.Param.TimeLimit, 20.0)
	m.setParam("LogFile", "");
	#m.setParam(GRB.Param.Cuts, 1);
	#m.setParam(GRB.Param.Heuristics, 0.01);
	# NOTE: m.setParam(GRB.Param.Presolve, 2) does not help
	
	if not VERBOSE:
		setParam('OutputFlag', 0);
			
	variables= set();
	seeds = {}
	createDecisionVariableSeeds(m,seeds,variables,seedsDetected_weights,seedHypernymFilterSet);
	targets = {}
	targetsToCentralities={};
	loadAllTargetsCNet(sortedScoreAndIndexList, targetWordsList, targetWordsDictonary,\
	targetsToCentralities,variables,targets,m);
	
	#numberOfClusters = clusters.max();
	#clusterVariables={};
	
	############# limit Sum_i I(cluster_i) = #targets_i_want*2
	m.addConstr(quicksum(targets[c1] for c1 in targets),GRB.LESS_EQUAL,util.SUM_CONFIDENCE_LIMIT_PSL_ONE);
	objective = LinExpr();
	
	######### create PSL based objective function ###########
	objective = createPSLBasedObjectiveFunction(m,objective,variables,seeds,targets,targetsToCentralities,\
	seedHypernymFilterSet, seedsDetected_weights, orderedSeedWordsList, \
	sortedScoreAndIndexList, targetWordsList, targetWordsDictonary,\
	pairwiseDistancesTargetWords);
	print("\tall model constraints updated..");
	
	m.setObjective(objective);
	# The objective is to minimize the costs
	m.modelSense = GRB.MINIMIZE

	# Update model to integrate new variables
	m.update()
	# m.tune()
	# for i in range(m.tuneResultCount):
	# 	m.getTuneResult(i)
	# 	m.write('tune'+str(i)+'.prm')
	m.optimize();
	
	if m.SolCount > 0:
		detectedSeedsFileName = reweightedSeedsFile[reweightedSeedsFile.rindex("/")+1:reweightedSeedsFile.index(".")];
		m.write('optimization_output/out'+detectedSeedsFileName+'.lp');
		m.write('optimization_output/out'+detectedSeedsFileName+'.sol');
	
	filePrefix = reweightedSeedsFile[:reweightedSeedsFile.index(".")];
	outputFile = open(filePrefix+"_inf.txt","w");
	if m.SolCount > 0:
		printSolution(m,targets,outputFile,targetsToCentralities);
		outputFile.close();
	return (filePrefix+"_inf.txt")


########################################################################
############# Start of Script
########################################################################
if __name__ == "__main__":
	# sys.argv[1] = reweighted set of seeds for an image
	# sys.argv[2] = detected seeds to modified map file
	# sys.argv[3] = targets for each seed
	
	if len(sys.argv) >= 4:
		VERBOSE= True;
		mergeTargets.VERBOSE = True
		
		for index in range(1,5):
			reweightedSeedsFile = sys.argv[1]+"_"+str(index)+"_c.txt";
	
			#### Step 1: Merge targets from different seeds.
			[sortedScoreAndIndexList, targetWordsList, targetWordsDictonary,seedsDetected_weights,\
			orderedSeedWordsList,allSeedsDictionary] = mergeTargets.mergeTargetsFromDetectedSeeds(\
			reweightedSeedsFile, sys.argv[2], 1500);
			print("\tmerging targets completed..");
	
			#### Step 2: cluster the merged set of targets
			[clusters,sortedScoreAndIndexList,Z,pairwiseDistancesTargetWords] = clusterTargets.returnClusters(sortedScoreAndIndexList, \
			targetWordsList, targetWordsDictonary, orderedSeedWordsList, 2500);
			print(pairwiseDistancesTargetWords.shape);
			print("\tclustering targets completed..");
	
			#### Step 3: create 1-word and 2-word model
			optimizeAndInferConceptsModelOneNew(allSeedsDictionary,seedsDetected_weights,orderedSeedWordsList,reweightedSeedsFile,\
			sortedScoreAndIndexList, targetWordsList, targetWordsDictonary, pairwiseDistancesTargetWords);
	else:
		VERBOSE= True;
		reweightedSeedsFile = sys.argv[1];
	
		#### Step 1: Merge targets from different seeds.
		mergeTargets.VERBOSE = True;
		[sortedScoreAndIndexList, targetWordsList, targetWordsDictonary,seedsDetected_weights,\
		orderedSeedWordsList,allSeedsDictionary] = mergeTargets.mergeTargetsFromDetectedSeeds(\
		reweightedSeedsFile, sys.argv[2], 1500);
		print("\tmerging targets completed..");
	
		#### Step 2: cluster the merged set of targets
		[clusters,sortedScoreAndIndexList,Z,pairwiseDistancesTargetWords] = clusterTargets.returnClusters(sortedScoreAndIndexList, \
		targetWordsList, targetWordsDictonary, orderedSeedWordsList, 2500);
		print(pairwiseDistancesTargetWords.shape);
		print("\tclustering targets completed..");
	
		#### Step 3: create 1-word and 2-word model
		optimizeAndInferConceptsModelOneNew(allSeedsDictionary,seedsDetected_weights,orderedSeedWordsList,reweightedSeedsFile,\
		sortedScoreAndIndexList, targetWordsList, targetWordsDictonary, pairwiseDistancesTargetWords);
