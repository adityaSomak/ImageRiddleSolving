from __future__ import print_function
from conceptnet5.query import *
from assoc_space import AssocSpace
from util import *
import sys
import os
import math
import numpy as np
import time

from concurrent.futures import ThreadPoolExecutor
import threading

#######################################################################
########		PIPELINE STAGE II(preprocessing)
########		Input: all possible seeds
########		Output: list of similar+visually similar targets
########		*** Run in server ****
#######################################################################

def filterEnglishWords(string):
	return string.startswith("/c/en/");

def simplify(string):
	string = string[6:];
	index = string.find("/");
	if index > 0:
		return string[:index];
	return string;
	
def _addToPropertiesDict(properties, assertion, relation, superclass_properties=None, superclass=None):
	if assertion['end'].startswith("/c/en"):
		simplifiedWord = simplify(encode(assertion['end']));
		properties[relation][simplifiedWord]=float(assertion['weight']);
		if superclass != None:
			superclass_properties[superclass][relation+"_"+simplifiedWord]=float(assertion['weight']);

def populateVisualProperties(FINDER,properties,word,visualRelations,superclasses):
	superclass_properties={};
	for relation in visualRelations:
		properties[relation] = {}; 
		criteria ={};
		criteria['start']=word;
		criteria['rel']="/r/"+relation;
		## Querying conceptnet
		for assertion in FINDER.query(criteria):
			_addToPropertiesDict(properties, assertion, relation);
	for superclass in superclasses:
		superclass_properties[superclass]={};
	
	for relation in ["HasA","HasProperty"]:	
		for superclass in superclasses:
			criteria ={};
			criteria['start']=superclass;
			criteria['rel']="/r/"+relation;
			## Querying conceptnet
			for assertion in FINDER.query(criteria):
				_addToPropertiesDict(properties, assertion, relation, superclass_properties, superclass);
	return superclass_properties

def _addToFeatureVector(targetFeatureVector, assertion, relation, featureDict, \
superclass, superclass_properties_cache):
	if assertion['end'].startswith("/c/en"):
		simplifiedWord = simplify(encode(assertion['end']));
		key = relation+"_"+simplifiedWord;
		if key in featureDict.keys():
			targetFeatureVector[featureDict[key]]=float(assertion['weight']);
			superclass_properties_cache[superclass][key]=float(assertion['weight']);

###############################################
#### Directly populate the target-feature vector
#### from seed-feature list
################################################
def populateVisualFeatureVectorFast(FINDER,targetFeatureVector,word,visualRelations,superclasses,\
featureDict,superclass_properties_cache):
	visualRelations = set(visualRelations);
	# Query-1 for word
	for assertion in FINDER.lookup(word):
		relation = assertion['rel'][3:]; #substring after "/r/"
		if relation in visualRelations:
			_addToFeatureVector(targetFeatureVector, assertion, relation, featureDict);
	for superclass in superclasses:
		if superclass in superclass_properties_cache.keys():
			propertiesDict = superclass_properties_cache[superclass];
			for key in propertiesDict.keys():
				targetFeatureVector[featureDict[key]]=propertiesDict[key];
		else:
			superclass_properties_cache[superclass]={}
			for relation in ["HasA","HasProperty"]:
				criteria ={};
				criteria['start']=superclass;
				criteria['rel']="/r/"+relation;
				for assertion in FINDER.query(criteria):
					if assertion['end'].startswith("/c/en"):
						_addToFeatureVector(targetFeatureVector, assertion, relation, featureDict, \
						superclass, superclass_properties_cache)
	return targetFeatureVector;
		
################################################################
######## Create list of features (all words connected through 
######## different relations) from both seed and target properties
################################################################
def createCombinedFeatureList(seed_properties):
	vector = set();
	for key in seed_properties.keys():
		dictOfWords = seed_properties[key];
		for word in dictOfWords:
			vector.add(key+"_"+word);
	return vector;

################################################################
######## Populate feature vector for a seed/target
######## Each element says that say hasA_leg: 1.0, IsA_animal:2.0 etc
################################################################
def getFeatureVector(properties,featureList):
	vec = np.zeros((len(featureList)));
	for i in range(0,len(featureList)):
		feature = featureList[i];
		keyAndFeaturename = feature.split("_");
		featuresAndWeights = properties[keyAndFeaturename[0]];
		if keyAndFeaturename[1] in featuresAndWeights.keys():
			vec[i] = featuresAndWeights[keyAndFeaturename[1]];
	return vec;


##### TODO: get better superclasses from wordnet.
##### dont need all of these.
def getSuperOrSubclasses(FINDER,word,isSuperClass):
	superclasses =[];
	criteria ={};
	if isSuperClass:
		criteria['start']=word;
	else:
		criteria['end']=word;
	criteria['rel']="/r/IsA";
	for assertion in FINDER.query(criteria):
		if isSuperClass and assertion['end'].startswith('/c/en'):
			superclasses.append(encode(assertion['end']));
		if (not isSuperClass) and assertion['start'].startswith('/c/en'):
			superclasses.append(encode(assertion['start']));
	return superclasses
	
####################################################################
### Compute cosine similairty between two vectors
####################################################################

def computeSemanticSimilarity(featureList,seedFeatureVector,target_properties,seedFeatureNormSquared):
	targetFeatureVector = getFeatureVector(target_properties,featureList);
	print(targetFeatureVector);
	return np.dot(seedFeatureVector,targetFeatureVector)/seedFeatureNormSquared;
	
def computeSimilarity(FINDER,term,featureList,featureDict,seedFeatureVector,visualRelations,\
seedFeatureNormSquared,superclass_properties_cache):
	superclasses = getSuperOrSubclasses(FINDER,term,True);
	targetFeatureVector = np.zeros((len(featureList)));
	targetFeatureVector = populateVisualFeatureVectorFast(FINDER,targetFeatureVector,term[0],\
	visualRelations,superclasses,featureDict,superclass_properties_cache)
	
	return np.dot(seedFeatureVector,targetFeatureVector)/seedFeatureNormSquared;

def getNormalizedSeedCentrality(seedWord, assocSpace):
	indexOfWord = assocSpace.labels.index(seedWord);
	return computeNormalizedValue(assocSpace.u[indexOfWord,0],0.00324597,-0.00188222);
	
'''
	####################################################################
	Start: Process and predict similar words for a single term
	####################################################################
'''
def processTargetsForSingleTerm(FINDER,seedWord,assocSpace,visualRelations,indexInList=-1):
	#threadLimiterSemaphore.acquire();
	termsList = assocSpace.terms_similar_to_vector(assocSpace.to_vector("/c/en/"+seedWord),\
	filter=filterEnglishWords,num=int(sys.argv[2]));
	
	##############################################
	###### Get all visual properties and relations 
	###### from the target term's superclasses (and subclasses).
	##############################################
	print(str(indexInList),"::got the list of similar terms");
	seed_superclasses = getSuperOrSubclasses(FINDER,"/c/en/"+sys.argv[1],True);
	#seed_subclasses = getSuperOrSubclasses("/c/en/"+sys.argv[1],False);
	print("superclasses updated..")
	seed_properties={};
	superclass_properties_cache = populateVisualProperties(FINDER,seed_properties,"/c/en/"+seedWord,\
	visualRelations,seed_superclasses);
	
	print("visual properties updated..")
	featureList = list(createCombinedFeatureList(seed_properties));
	print(featureList);
	featureDict={};
	featureDict = {x:i for i,x in enumerate(featureList)};
	outputFile = open("intermediateFiles/opt/test/"+seedWord+"_targets_.txt","w");
	seedFeatureVector = getFeatureVector(seed_properties,featureList);
	print(seedFeatureVector);
	seedFeatureNormSquared = np.dot(seedFeatureVector,seedFeatureVector);
	
	#seedCentrality = getNormalizedSeedCentrality("/c/en/"+seedWord, assocSpace);
	#print(seedCentrality);
	if len(featureList)<4:
		for i in range(1,len(termsList)):
			term=termsList[i];
			print(encode(term[0]),"\t",str(term[1]),"\t1.0",file=outputFile);
		outputFile.close()
		return;
	
	decodeError = False;
	
	#outputCache = "";
	for i in range(1,len(termsList)):
		###### Filtering based on Semantics #######
		###### Re-weight and keep only the first 2000, keep both similarities ######
		term = termsList[i];
		#print(str(i),"\t",term[0]);
		#termCentrality = getNormalizedSeedCentrality(term[0], assocSpace);
		#if (termCentrality < seedCentrality-0.05) or (termCentrality > seedCentrality+0.05):
		#	print("ignoring....",str(termCentrality));
		#	continue; 
		try:
			finalWeight = computeSimilarity(FINDER,term[0],featureList,featureDict,seedFeatureVector,visualRelations,\
			seedFeatureNormSquared,superclass_properties_cache);
			
			#print(str(i),"\t",term[0],"\t",str(term[1]),"\t",str(finalWeight));
			string = encode(term[0])+"\t"+str(term[1])+"\t"+str(finalWeight);
			#outputCache= outputCache+string+"\n";
			outputFile.write(string+"\n");
			#print(encode(term[0]),"\t",str(term[1]),"\t",str(finalWeight),file=outputFile);
		except UnicodeDecodeError as ude:
			print("unicodedecode error....ignoring");
			decodeError = True;
		if i%1000==0:
			outputFile.flush();
	
	outputFile.close();
	### Comment this part out
	if decodeError != True:
		outputFileName="intermediateFiles/opt/test/"+seedWord+"_targets_.txt";
		num_lines = sum(1 for line in open(outputFileName));
		if num_lines < 7999:
			sys.exit();
	### Comment this part out
	#threadLimiterSemaphore.release();



'''
	####################################################################
	Start: Process and predict similar words for all seeds from
	a file. 
	Assumption is: file-format - <bias-score> <cnet-term> <original-term>
	####################################################################
'''
def processAllSeedsFile(seedsFile,assocSpace,visualRelations):
	threads=[];
	i=0;
	with open(seedsFile, "r") as f:
		for line in f:
			i=i+1;
			if line.startswith("##"):
				continue;
			tokens = line.split("\t");
			seedWord = tokens[1].strip();
			FINDER = AssertionFinder();
			t = threading.Thread(target=processTargetsForSingleTerm, name=str(seedWord), \
			args=(FINDER,seedWord,assocSpace,visualRelations));
			
			t.start();
			threads.append(t);
			if (i+1)%5==0:
				for j in range(len(threads)):
					threads[j].join();
				threads=[];
			if i>=500:
				break;



'''
	####################################################################
				***ThreadPoolExecutor version*** 
	Start: Process and predict similar words for all seeds from
	a file. 
	Assumption is: file-format - <bias-score> <cnet-term> <original-term>
	####################################################################
'''
def processAllSeedsFileThreadPool(seedsFile,assocSpace,visualRelations):
	threads=[];
	#maxThreadsAvailable = multiprocessing.cpu_count();
	i=0;
	#executor=ThreadPoolExecutor(max_workers=30);
	partNumber = int(sys.argv[4]);
	numWorkers =int(sys.argv[3]);
	with ThreadPoolExecutor(max_workers=2) as executor:
		with open(seedsFile, "r") as f:
			for line in f:
				i=i+1;
				if line.startswith("##"):
					continue;
				if i < 420:
					continue;
				if i%10!= partNumber:
					continue;
				tokens = line.split("\t");
				seedWord = tokens[1].strip();
				outputFileName = "intermediateFiles/opt/test/"+seedWord+"_targets_.txt";
				if os.path.isfile(outputFileName):
					num_lines = sum(1 for line in open(outputFileName));
					if num_lines == 7999:
						print(str(i)," ignored as computation is complete...");
						continue;	
				FINDER = AssertionFinder();
				executor.submit(processTargetsForSingleTerm,FINDER,seedWord,assocSpace,visualRelations,i);

########################################################################
######### Start of Script
########################################################################
np.seterr(divide='ignore', invalid='ignore');
if 	len(sys.argv) < 3:
	print("python suggestTargets.py <word> <numberofSimilarTerms> <threads> <part>");
	print("python suggestTargets.py <word1> <word2> <anynumber>");
	sys.exit();

assocDir = "../conceptnet5/data/assoc/assoc-space-5.4";
assocSpace = AssocSpace.load_dir(assocDir);
visualRelations=["PartOf","MemberOf","HasA","HasProperty"];
if len(sys.argv) == 5:
	if sys.argv[1].endswith(".txt"):
		#processAllSeedsFile(sys.argv[1],assocSpace,visualRelations);
		processAllSeedsFileThreadPool(sys.argv[1],assocSpace,visualRelations);
	else:
		FINDER = AssertionFinder();
		processTargetsForSingleTerm(FINDER,sys.argv[1],assocSpace,visualRelations);
elif len(sys.argv)==4:
	FINDER = AssertionFinder();
	seed_superclasses = getSuperOrSubclasses(FINDER,"/c/en/"+sys.argv[1],True);
	print("superclasses updated..")
	seed_properties={};
	populateVisualProperties(FINDER,seed_properties,"/c/en/"+sys.argv[1],visualRelations,seed_superclasses);
	print("visual properties updated..")
	print(seed_properties);
	featureList = list(createCombinedFeatureList(seed_properties));
	print(featureList);
	seedFeatureVector = getFeatureVector(seed_properties,featureList);
	print(seedFeatureVector);
	seedFeatureNormSquared = np.dot(seedFeatureVector,seedFeatureVector);
	
	finalWeight = computeSimilarity("/c/en/"+encode(sys.argv[2]),featureList,seedFeatureVector,visualRelations,seedFeatureNormSquared);
	print(sys.argv[2]+"\t"+str(finalWeight));

