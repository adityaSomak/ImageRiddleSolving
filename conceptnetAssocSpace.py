from conceptnet5.query import *
from assoc_space import AssocSpace
import sys
import threading
import math


def computeNormalizedValue(value, maxV, minV, addOne=False):
	if addOne:
		return (value-minV+1)/(maxV-minV+1);
	return (value-minV)/(maxV-minV);

if len(sys.argv) < 4:
	print "python conceptnetAssocSpace.py <seedsfile> <targetfile> <AssocSpaceDirectory>";
	sys.exit();
assocSpace = AssocSpace.load_dir(sys.argv[3]);
words = [];
minSimilarity=-0.358846;
maxSimilarity= 0.999747;
minCentrality= -0.00188222;
maxCentrality= 0.00324597;
with open(sys.argv[1], "r") as f:
	i=0;
	for line in f:
		if line.startswith("##"):
			continue;
		words=line.split("\t");
		word1="/c/en/"+words[0].strip();
		with open(sys.argv[2], "r") as f2:
			for line in f2:
				if line.startswith("##"):
					continue;
				targets = line.split("\t");
				centralityscore = float(targets[0].strip());
				centralityscore = computeNormalizedValue(centralityscore, maxCentrality, minCentrality);
				target = "/c/en/"+targets[1].strip();
				similarity = assocSpace.assoc_between_two_terms(word1,target);
				similarity = computeNormalizedValue(similarity, maxSimilarity, minSimilarity,True);
				if similarity>0:
					print word1,"\t",target,"\t",similarity,"\t",centralityscore;
