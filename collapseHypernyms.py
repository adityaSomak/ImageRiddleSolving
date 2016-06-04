import conceptnet_util
import sys

#######################################################################
################ IGNORE FILE
#######################################################################
def populateModifiedSeedsAndConcreteScoreDictionary():
	# detected_seed -> centrality_score, seed_in_conceptnet
	allSeedsDictionary={};
	with open(sys.argv[1], 'r') as f:
		for line in f:
			if line.startswith("##"):
				continue;
			tokens=line.split("\t");
			## conc. score = concrete ness score only
			concereteNessScore = computeNormalizedValue(float(tokens[0].strip()),5,0);
			if len(tokens) > 3:
				allSeedsDictionary[tokens[3].strip()] = [concereteNessScore,tokens[2].strip()];
			else:
				allSeedsDictionary[tokens[2].strip()] = [concereteNessScore,tokens[2].strip()];
	print("seeds_dict populated...");
	return allSeedsDictionary



def processClarifaiJsonFile(fileName):
	with open(fileName, 'r') as myfile:
		line = myfile.read();
	line = line.replace("u\'", "");
	line = line.replace("\'", "");
	line = line.replace(":", "\n");
	lines = line.split("\n");
	return lines;


###############################################
############# Start of script
###############################################
if 	len(sys.argv) < 3:
	print("python suggestTargets.py <seedsCentralityConcreteNessfile> <detectedSeedsFile>");
	sys.exit();

allSeedsDictionary = populateModifiedSeedsAndConcreteScoreDictionary();
lines = processClarifaiJsonFile(sys.argv[2]);
	
detections = (lines[15][2:lines[15].index("]")]).split(",");
weights = (lines[16][2:lines[16].index("]")]).split(",");

detectionsDict = OrderedDict();
for i in range(0,len(detections)):
	detection = detections[i];
	detection = allSeedsDictionary[detection][1];
	concreteNessScore = allSeedsDictionary[detection][0]);
	if concreteNessScore > 3:
		detectionsDict[detection] = (weights[i],concreteNessScore,None);
	detectionsDict[detection] = (weights[i],concreteNessScore,set(conceptnet_util.getSuperOrSubclasses("/c/en/"+word,True)));

########## sort based on concreteness-score
detectionsDict = OrderedDict(sorted(detectionsDict.iteritems(), key=lambda x: x[1]))
##### => animal -> reptile -> dinosaur.
##### for 0 to n -> try searching from end

newDetectionDict={};
########## TODO: if  for a word, it already occurs as a superclass, then dont include it.
for item1 in detectionsDict.items():
	
	for item2 in reversed(detectionsDict.items()):
########################################################################
