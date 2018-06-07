from __future__ import print_function
import sys
from gurobipy import *
from util import *
	
def printSolution(m,targets,outputFile,targetsToCentralities):
    #if m.status == GRB.Status.OPTIMAL:
    m.printAttr('x');
    print('\nDistance to Satisfaction: %g' % m.objVal)
    print('\nTargets:')
    targetsx = m.getAttr('x', targets)
    for t in targetsx:
		if targets[t].x > 0.0001:
			print('%s\t%g\t%g' % (t, targetsx[t],targetsToCentralities[t]),file=outputFile);
    #else:
    #    print('No solution')

########################################################################
############# Load All targets_in_CNet (TODO: just for the image)
########################################################################	
def loadAllTargetsCNet(targets,variables,m,targetsToCentralities):
	with open(sys.argv[3], "r") as f:
		for line in f:
			if line.startswith("##"):
				continue;
			tokens = line.split("\t");
			target = tokens[1].strip();
			if target in variables:
				continue;
			targetsToCentralities[target] = float(tokens[0].strip());
			targets[target] = m.addVar(lb=0.0, ub=1.0, name=target);
			variables.add(target);
	m.update();

########################################################################
############# Create decision variables for the seeds
############# Limit them using constraints
########################################################################	
def createDecisionVariableSeeds(m,seeds,variables,seedsDetected_weights):
	for c in seedsDetected_weights.keys():
		## add arbit string to distinguish from targets
		c1 =  c+"1";
		if c1 in variables:
			continue;
		seeds[c1] = m.addVar(lb=seedsDetected_weights[c], ub=seedsDetected_weights[c], name=c1);
		variables.add(c1);
		#m.addConstr(seeds[c1] == seedsDetected_weights[c],c1);

# create a PSL like objective function
# based on rules read from a centrality file.
# rules of the form wt: seed_j => target_i
# converted to max(I(seed_j)-I(target_i),0).
def createPSLBasedObjectiveFunction(m,objective,variables,seeds,targets,seedsDetected_weights):
	with open(sys.argv[4], "r") as f:
		lineNo =0;
		for line in f:
			tokens = line.split("\t");
			tokens =  map(lambda x:x.strip(),tokens);
			seed = tokens[0][6:len(tokens[0])];
			target1 = tokens[1][6:len(tokens[1])];		
			similarity = float(tokens[2]);
			centralityOfTarget = float(tokens[3]);
			if centralityOfTarget != 0 and similarity > 0.64 and (seed in seedsDetected_weights.keys()):
				seed1 = seed+"1";
				ruleVar = target1+"_"+seed1;
				ruleVariable = m.addVar(name=ruleVar);
				weight = similarity;#+1.0/centralityOfTarget;
				objective+= weight*ruleVariable; 
				### TODO: dirty hack. 
				m.update();
				#max(seeds[seed1]-targets[target1],0);
				m.addConstr(ruleVariable,GRB.GREATER_EQUAL,0);
				m.addConstr(ruleVariable,GRB.GREATER_EQUAL,seeds[seed1]-targets[target1]);
	m.update();
	return objective;

########################################################################
############# Start of Script
########################################################################
def optimizeAndInferConceptsModelOne(detectedSeedsFile):
	allSeedsDetected_toCNet = util.loadAllSeedsAndModifiedSeedsCNet(sys.argv[1]);
	seedsDetected_weights = util.readReweightedSeeds(detectedSeedsFile,allSeedsDetected_toCNet);
	# Model
	m = Model("psl1");
	variables= set();
	seeds = {}
	createDecisionVariableSeeds(m,seeds,variables,seedsDetected_weights);
	targets = {}
	targetsToCentralities={};
	loadAllTargetsCNet(targets,variables,m,targetsToCentralities);
	############# Sum_i I(target_i) = 1, behaves like probabilities
	m.addConstr(quicksum(targets[c1] for c1 in targets),GRB.LESS_EQUAL,10.0);
	objective = LinExpr();
	objective = createPSLBasedObjectiveFunction(m,objective,variables,seeds,targets,seedsDetected_weights);
	m.setObjective(objective);

	# The objective is to minimize the costs
	m.modelSense = GRB.MINIMIZE

	# Update model to integrate new variables
	m.update()
	m.optimize();
	detectedSeedsFileName = detectedSeedsFile[detectedSeedsFile.rindex("/")+1:detectedSeedsFile.index(".")];
	m.write('out'+detectedSeedsFileName+'.lp');
	m.write('out'+detectedSeedsFileName+'.sol');
	filePrefix = detectedSeedsFile[:detectedSeedsFile.index(".")];
	outputFile = open(filePrefix+"_inf.txt","w");
	printSolution(m,targets,outputFile,targetsToCentralities);

if len(sys.argv) < 5:
	print("python/gurobi.sh ",sys.argv[0]," <seedsCentralityfile> <detectedSeedsFile> <allTargetsFile> <allRulesCentrality>")
	sys.exit();
else:
	if not sys.argv[2].endswith(".txt"):
		for i in range(1,5):
			optimizeAndInferConceptsModelOne(sys.argv[2]+str(i)+"_.txt");
	else:
		optimizeAndInferConceptsModelOne(sys.argv[2]);
