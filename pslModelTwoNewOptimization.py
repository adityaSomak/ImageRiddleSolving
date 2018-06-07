from __future__ import print_function

import sys

import conceptnet_util
import util

'''
#######################################################################
########		PIPELINE STAGE VI. (TODO: test it)
########		Input: seeds, ranked targets for all images,
########				Context Image-index
########		Output: Ranked Final Targets
#######################################################################
'''

def printSolution(m,targets,outputFile,targetsToCentralities):
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

## load Targets to centralities mapping
def loadTargetToCentralities(inferredTargetsFileName, targetsToCentralities):
	i=0;
	with open(inferredTargetsFileName, 'r') as f:
		for line in f:
			if i == util.PER_IMAGE_TARGET_LIMIT_PSL_TWO:
				break;
			tokens = line.split("\t");
			detection = tokens[0].strip();
			targetsToCentralities[detection] = float(tokens[2].strip());
			i=i+1;

#####################################################################
#### Create decision variables for the seeds
#### Limit them using constraints
#####################################################################	
def loadDecisionVariablesForSeeds(m,seeds,variables,seedsDetected_weights):
	for c_index in seedsDetected_weights.keys():
		if c_index in variables:
			continue;
		seeds[c_index] = m.addVar(lb=seedsDetected_weights[c_index], ub=seedsDetected_weights[c_index], name=c_index);
		variables.add(c_index);


#####################################################################
#### Create variables for the targets
#### Load centralities
#####################################################################	
def loadDecisionVariablesForTargets(m,targets,variables,targetsToCentralities):
	for targetWord in targetsToCentralities.keys():		
		targets[targetWord] = m.addVar(lb=0.1, ub=1.0, name=targetWord, vtype=GRB.SEMICONT);
		variables.add(targetWord);
	m.update();


########################################################################
## Create Objective function
## sim+1/centrality: target_j_(img) -> target_i
## sum(I(target_i));
########################################################################
def createObjective(m,seeds,targets,variables,objective,targetsToCentralities,seedsDetected_weights):
	globalWeightSum=0;
	tupleOfObjectives=[];
	#print(seedsDetected_weights.keys());
	for seed_index in seedsDetected_weights.keys():
		seedWithoutIndex =  seed_index[:len(seed_index)-1];
		tuplesOfConstraints=[];
		for targetWord in targetsToCentralities.keys():
			centralityOfTarget = targetsToCentralities[targetWord];
			
			#try:
			sim_word1 = conceptnet_util.getSimilarity(seedWithoutIndex,targetWord,True);
			word2vecsimilarity = conceptnet_util.getWord2VecSimilarity(seedWithoutIndex,targetWord,True);
			#except KeyError as e:
			#print("seedWithoutIndex:%s, targetWord:%s\n" % (seedWithoutIndex,targetWord));
			#raise e;
			if word2vecsimilarity == conceptnet_util.NOT_FOUND_IN_CORPUS_CODE:
				similarity = sim_word1;
			else:
				similarity= (sim_word1*util.CONCEPTNET_SIMILARITY_WEIGHT+\
				word2vecsimilarity*util.WORD2VEC_SIMILARITY_WEIGHT)/\
				(util.CONCEPTNET_SIMILARITY_WEIGHT+util.WORD2VEC_SIMILARITY_WEIGHT);
		
			if sim_word1 > util.SIMILARITY_THRESHOLD_ONEWORDRULE_PSL_ONE:	
				ruleVar = targetWord+"_"+seed_index;
				ruleVariable = m.addVar(name=ruleVar);
				## use normalized value for centrality
				penaltyForPopularity = 0.5 * util.computeNormalizedValue(1.0/centralityOfTarget,3.0,0.0);
				
				weight = similarity+penaltyForPopularity;
				#objective+= weight*ruleVariable; 
				globalWeightSum = globalWeightSum+weight;
				tupleOfObjectives.append((ruleVariable,weight));
				
				tuplesOfConstraints.append((ruleVariable,seed_index,targetWord));
		
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
	
########################################################################
############# Start of Script
########################################################################
def optimizeAllAndInferConceptsModelTwo(targetsToCentralities,seedsDetected_weights,targetPrefix,pathPrefix):
	# Model
	m = Model("psl2")
	m.setParam(GRB.Param.TimeLimit, 20.0);
	variables= set();
	seeds={};
	loadDecisionVariablesForSeeds(m,seeds,variables,seedsDetected_weights);
	
	targets = {}
	loadDecisionVariablesForTargets(m,targets,variables,targetsToCentralities);
	
	m.addConstr(quicksum(targets[c1] for c1 in targets),GRB.LESS_EQUAL,util.SUM_CONFIDENCE_LIMIT_PSL_TWO);
	## populate the rules
	objective = LinExpr();
	objective = createObjective(m,seeds,targets,variables,objective,targetsToCentralities,seedsDetected_weights);
				
	if VERBOSE:
		print("\tmodel updated...solution calculating..");
	m.update();
	m.setObjective(objective);
	
	# The objective is to minimize the costs
	m.modelSense = GRB.MINIMIZE

	# Update model to integrate new variables
	m.update()
	m.optimize();
	detectedSeedsFileName = targetPrefix;
	m.write('optimization_output/out'+detectedSeedsFileName+'2.lp');
	if m.SolCount > 0:
		m.write('optimization_output/out'+detectedSeedsFileName+'2.sol');
	else:
		m.computeIIS()
		m.write('optimization_output/out'+detectedSeedsFileName+'2.ilp')
	
	outputFile = open(pathPrefix+"_inf_all.txt","w");	
	if m.SolCount > 0:
		printSolution(m,targets,outputFile,targetsToCentralities);
		outputFile.close();
	return outputFile.name;

def addTopSeedsAsTargets(targetsToCentralities,detectedSeedsFileName, allSeedsDictionary, apiUsed):
	if apiUsed == "clarifai":
		return;
	[detections,weights] = util.getDetectionsFromTSVFile(detectedSeedsFileName);
	[detections,weights] = conceptnet_util.chooseSingleRepresentativeDetection(allSeedsDictionary, detections, weights);
	
	orderedSeedWordsList = detections;#sorted(detections,key=lambda s:seedsDetected_weights[s],reverse=True);
	prevWeight = 0.99;
	i=0;
	topSeeds=[];
	for seed in orderedSeedWordsList:
		if weights[i] >= max(0.01,prevWeight*0.1):
			topSeeds.append(seed);
		prevWeight = weights[i];
		i=i+1;
		if i==2:
			break;
	for seed in topSeeds:
		seedWithoutIndex = seed[:len(seed)-1];
		targetsToCentralities[seedWithoutIndex] = conceptnet_util.getCentralityScore(seedWithoutIndex,True);
		

def callPSLModelTwo(allSeedsDictionary,inferenceFolder,seedPrefix,detectionFolder,apiUsed):
	seedsDetected_weights={};
	targetsToCentralities={};
	for index in range(1,5):
		reweightedSeedsFileName = inferenceFolder+"opt_"+seedPrefix+"_"+str(index)+"_c.txt";
		newSeedsAndWeights= util.readReweightedSeeds(reweightedSeedsFileName,allSeedsDictionary,True,True,index);
		seedsDetected_weights.update(newSeedsAndWeights);
		#print(seedsDetected_weights);
		
		sortedTargetsFileName = inferenceFolder+"opt_"+seedPrefix+"_"+str(index)+"_c_inf.txt";
		loadTargetToCentralities(sortedTargetsFileName, targetsToCentralities);
		detectedSeedsFileName = detectionFolder+seedPrefix+"_"+str(index)+".txt";
		addTopSeedsAsTargets(targetsToCentralities,detectedSeedsFileName,allSeedsDictionary,apiUsed);
		
	fileName= optimizeAllAndInferConceptsModelTwo(targetsToCentralities,seedsDetected_weights,\
	seedPrefix,inferenceFolder+"opt_"+seedPrefix);
	return fileName;

########################################################################
############# Start of Script
########################################################################
if __name__ == "__main__":
	
	# sys.argv[1] = inference-folder
	# sys.argv[2] = seed-prefix 
	# sys.argv[3] = detected seeds to modified map file
	VERBOSE= True;
	
	allSeedsDictionary = util.populateModifiedSeedsAndConcreteScoreDictionary(sys.argv[3]);
	seedsDetected_weights={};
	targetsToCentralities={};
	for index in range(1,5):
		reweightedSeedsFileName = sys.argv[1]+"opt_"+sys.argv[2]+"_"+str(index)+"_c.txt";
		newSeedsAndWeights= util.readReweightedSeeds(reweightedSeedsFileName,allSeedsDictionary,True,True,index);
		seedsDetected_weights.update(newSeedsAndWeights);
		#print(seedsDetected_weights);
		
		sortedTargetsFileName = sys.argv[1]+"opt_"+sys.argv[2]+"_"+str(index)+"_c_inf.txt";
		loadTargetToCentralities(sortedTargetsFileName, targetsToCentralities)
		
	optimizeAllAndInferConceptsModelTwo(targetsToCentralities,seedsDetected_weights,sys.argv[2],sys.argv[1]+"opt_"+sys.argv[2]);
	
