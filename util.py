

def populateModifiedSeedsAndConcreteScoreDictionary(modifiedSeedsMapFile,loadConcScores=True,weightConcScores=1.0):
	# detected_seed -> centrality_score, seed_in_conceptnet, amtconcScore
	# seed_in_conceptnet -> centrality_score, seed_in_conceptnet, amtconcScore
	allSeedsDictionary={};
	with open(modifiedSeedsMapFile, 'r') as f:
		for line in f:
			if line.startswith("##"):
				continue;
			tokens=line.split("\t");
			## conc. score = conc mean mapped to [0,2] scale+ 
			##               centrality mapped to [0,1];
			concereteNessScore = computeNormalizedValue(float(tokens[1].strip()),0.00324597,-0.00188222);
			amtConcScore = float(tokens[0].strip());
			concereteNessScore = concereteNessScore + computeNormalizedValue(0-amtConcScore\
			,0,-5)*weightConcScores;
			if len(tokens) > 3:
				allSeedsDictionary[tokens[3].strip()] = [concereteNessScore,tokens[2].strip(),amtConcScore];
			allSeedsDictionary[tokens[2].strip()] = [concereteNessScore,tokens[2].strip(),amtConcScore];
	print("\tUTIL_PY: seeds_dict populated...");
	return allSeedsDictionary


########################################################################
############# Load All Seeds and Seeds_in_CNet
########################################################################
def loadAllSeedsAndModifiedSeedsCNet(modifiedSeedsMapFile,offset=0):
	allSeedsDetected_toCNet={};
	with open(modifiedSeedsMapFile, "r") as f:
		for line in f:
			if line.startswith("##"):
				continue;
			tokens = line.split("\t");
			tokens =  map(lambda x:x.strip(),tokens);
			if len(tokens) > offset+1:
				allSeedsDetected_toCNet[tokens[offset+1]] =  tokens[offset];
	return allSeedsDetected_toCNet;



## Read from the detected-seeds file (after re-weighting) from an image
def readReweightedSeeds(detectedSeedsFileName,allSeedsDictionary,addIndex=False,index=-1):
	seedsDetected_weights ={};
	with open(detectedSeedsFileName, "r") as f:
		for line in f:
			tokens = line.split("\t");
			tokens =  map(lambda x:x.strip(),tokens);
			weight = computeNormalizedValue(float(tokens[3]),2.0,0);
			#if tokens[1] not in allSeedsDetected_toCNet.keys():
			#seedsDetected_weights[tokens[1]]= weight;
			#else:
			seedInCNet = allSeedsDictionary[tokens[1]][1];
			if addIndex:
				seedsDetected_weights[seedInCNet+str(index)]= weight;
			else:
				seedsDetected_weights[seedInCNet]= weight;
	return seedsDetected_weights;


def computeNormalizedValue(value, maxV, minV, addOne=False):
	if addOne:
		return (value-minV+1)/(maxV-minV+1);
	return (value-minV)/(maxV-minV);

def encode(arg):
	if arg != None:
		return arg.encode('utf-8');
	return "";


def setParameters(sim_threshold,top_k_sim_targets,sim_threshold_onewordrule,sum_confidence_limit,
conceptnet_sim_wt, word2vec_sim_wt):
	global SIMILARITY_THRESHOLD_WORDWEIGHTS
	global CONCRETENESS_THRESHOLD_MERGET
	global TOP_K_SIMILAR_TARGETS_PSL_ONE
	global SUM_CONFIDENCE_LIMIT_PSL_ONE
	global CONCEPTNET_SIMILARITY_WEIGHT
	global WORD2VEC_SIMILARITY_WEIGHT
	SIMILARITY_THRESHOLD_WORDWEIGHTS= sim_threshold;
	TOP_K_SIMILAR_TARGETS_PSL_ONE = top_k_sim_targets;
	SIMILARITY_THRESHOLD_ONEWORDRULE_PSL_ONE = sim_threshold_onewordrule;
	SUM_CONFIDENCE_LIMIT_PSL_ONE = sum_confidence_limit;
	CONCEPTNET_SIMILARITY_WEIGHT = conceptnet_sim_wt;
	WORD2VEC_SIMILARITY_WEIGHT = word2vec_sim_wt;
	
SIMILARITY_THRESHOLD_WORDWEIGHTS = 0.8; ## if exceeds similarity, put an edge.

CONCRETENESS_THRESHOLD_MERGET = 2.8; ## above this=> concrete term
RATIO_COPH_DIST_FCLUSTER_THRESHOLD_CLUSTERT = 0.03; ## above this==> cut connections

MAX_TARGETS_PSL_ONE = 2000; ## max targets for creating rules.
TOP_K_SIMILAR_TARGETS_PSL_ONE = 1; ## maximum similar targets for each target
SIMILARITY_THRESHOLD_ONEWORDRULE_PSL_ONE = 0.4; ## add if not 0.
SUM_CONFIDENCE_LIMIT_PSL_ONE=2; ## controls number of observations we want to find.
TARGET_LOWER_BOUND_PSL_ONE=0.05;

CONCEPTNET_SIMILARITY_WEIGHT=1.0;
WORD2VEC_SIMILARITY_WEIGHT=4.0;	

PER_IMAGE_TARGET_LIMIT_PSL_TWO = 200;
SUM_CONFIDENCE_LIMIT_PSL_TWO = 2;

if __name__ == "__main__":	
	SIMILARITY_THRESHOLD_WORDWEIGHTS = 0.8; ## if exceeds similarity, put an edge.

	CONCRETENESS_THRESHOLD_MERGET = 2.8; ## above this=> concrete term
	RATIO_COPH_DIST_FCLUSTER_THRESHOLD_CLUSTERT = 0.03; ## above this==> cut connections

	MAX_TARGETS_PSL_ONE = 2000; ## max targets for creating rules.
	TOP_K_SIMILAR_TARGETS_PSL_ONE = 1; ## maximum similar targets for each target
	SIMILARITY_THRESHOLD_ONEWORDRULE_PSL_ONE = 0.4; ## add if not 0.
	SUM_CONFIDENCE_LIMIT_PSL_ONE=2; ## controls number of observations we want to find.
	TARGET_LOWER_BOUND_PSL_ONE=0.05;

	CONCEPTNET_SIMILARITY_WEIGHT=1.0;
	WORD2VEC_SIMILARITY_WEIGHT=4.0;
	
	PER_IMAGE_TARGET_LIMIT_PSL_TWO = 200;
	SUM_CONFIDENCE_LIMIT_PSL_TWO = 2;
