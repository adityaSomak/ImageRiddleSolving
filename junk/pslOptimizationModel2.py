from __future__ import print_function

import assoc_space
from assoc_space import AssocSpace
import sys
from gurobipy import *
import re



def computeNormalizedValue(value, maxV, minV, addOne=False):
	if addOne:
		return (value-minV+1)/(maxV-minV+1);
	return (value-minV)/(maxV-minV);

def printSolution(m,targets,outputFile):
    #if m.status == GRB.Status.OPTIMAL:
    m.printAttr('x');
    print('\nDistance to Satisfaction: %g' % m.objVal)
    print('\nTargets:')
    targetsx = m.getAttr('x', targets)
    pattern = re.compile("[0-9]");
    for t in targetsx:
		if targets[t].x > 0.0001 and pattern.search(t) is None:
			print('%s\t%g' % (t, targetsx[t]),file=outputFile);


## load Targets to centralities mapping
## load targets to image-indices and weights in each image
def loadTargetWordsFromAllImages(targetsToCentralities,targetsToImageIndicesAndWeights):
	for i in range(1,int(sys.argv[3])+1):
		fileName = sys.argv[1]+ sys.argv[2]+"_"+str(i)+"_inf"+".txt";
		with open(fileName, 'r') as f:
			for line in f:
				tokens = line.split("\t");
				detection = tokens[0].strip();
				weight = float(tokens[1].strip());
				targetsToCentralities[detection] = float(tokens[2].strip());
				if detection not in targetsToImageIndicesAndWeights.keys():
					targetsToImageIndicesAndWeights[detection] = [];
				targetsToImageIndicesAndWeights[detection].append((i,weight));


########################################################################
############# Create decision variables for the targets
############# for wildlife-> [(1,0.98),(2,0.99)]
############# add wildlife, wildlife1, wildlife2.
############# Limit them using constraints
########################################################################
def loadDecisionVariablesForTargets(m,targets,variables,targetsToImageIndicesAndWeights):
	for target in targetsToImageIndicesAndWeights.keys():
		targets[target] = m.addVar(lb=0.0, ub=1.0, name=target);
		variables.add(target);
		for elem in targetsToImageIndicesAndWeights[target]:
			target_i = target+str(elem[0]);
			targets[target_i] = m.addVar(lb=elem[1], ub=elem[1], name=target_i);
			variables.add(target_i);

########################################################################
## Create Objective function
## sim+1/centrality: target_j_(img) -> target_i
## sum(I(target_i));
########################################################################
def createObjective(m,targets,variables,objective,assocSpace,targetsToCentralities,targetsToImageIndicesAndWeights):
	for key_i in targetsToImageIndicesAndWeights.keys():
		objective+= 1.0*targets[key_i]; 
		for key_j in targetsToImageIndicesAndWeights.keys():
			## TODO: if similar add 1: key_(img_index) -> key
			## TODO: if not, get similarity+1/centrality_score from assoc_space.
			similar = (key_i==key_j);
			for elem in targetsToImageIndicesAndWeights[key_j]:
				target_j = key_j+str(elem[0]);
				if similar:
					similarity =1;
				else:
					similarity = assocSpace.assoc_between_two_terms("/c/en/"+key_i,"/c/en/"+key_j);
					similarity = computeNormalizedValue(similarity,0.999747,-0.358846);
				ruleVar = key_i+"_"+target_j;
				ruleVariable = m.addVar(lb=0.0, ub=1.0, name=ruleVar);
				weight = similarity+1.0/targetsToCentralities[key_i];
				objective+= weight*ruleVariable; 
				### TODO: dirty hack. 
				m.update();				
				m.addConstr(ruleVariable,GRB.GREATER_EQUAL,0);
				m.addConstr(ruleVariable,GRB.GREATER_EQUAL,targets[target_j]-targets[key_i]);
	return objective;

########################################################################
############# Start of Script
########################################################################
def optimizeAllAndInferConceptsModelTwo(assocDir):
	## load assocSpace
	assocSpace = AssocSpace.load_dir(assocDir);
	## targets and image-indices dictionary
	targetsToImageIndicesAndWeights={}; # target-word ->[(index,weight_i)...]
	targetsToCentralities={}; #target-word -> centrality-score
	loadTargetWordsFromAllImages(targetsToCentralities,targetsToImageIndicesAndWeights);
	# Model
	m = Model("psl2")
	variables= set();
	targets = {}
	loadDecisionVariablesForTargets(m,targets,variables,targetsToImageIndicesAndWeights);
	## TODO: populate the rules
	objective = LinExpr();
	objective = createObjective(m,targets,variables,objective,assocSpace,targetsToCentralities,targetsToImageIndicesAndWeights);
				
	m.update();
	m.setObjective(objective);
	
	# The objective is to minimize the costs
	m.modelSense = GRB.MINIMIZE

	# Update model to integrate new variables
	m.update()
	m.optimize();
	m.write('out2.lp');
	m.write('out2.sol');
	outputFile = open(sys.argv[1]+ sys.argv[2]+"_inferred.txt","w");
	printSolution(m,targets,outputFile);
	
if len(sys.argv) < 4:
	print("python/gurobi.sh ",sys.argv[0]," <detectionsFolder> opt_<targetName> <number-of-images>")
	sys.exit();
else:
	assocDir = "/windows/drive2/For PhD/KR Lab/UMD_vision_integration/Image_Riddle/conceptnet5/data/assoc/assoc-space-5.4";
	optimizeAllAndInferConceptsModelTwo(assocDir);
