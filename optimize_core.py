from __future__ import print_function

import scipy.optimize as optimize
import numpy as np

def sumOfMax(x):
	global edges
	sumOfMax = 0;
	for i in range(edges.shape[1]):
		## edge from u-> v => sum {max(c(u)-c(v),0)}
		sumOfMax = sumOfMax + max(x[edges[0,i]]-x[edges[1,i]]*0.75,0);
	return sumOfMax

def constraint(x):
	global node_scores
	return int(sum(x))-int(sum(node_scores))

	
## u'dinosaur', u'animal', u'illustration', u'jurassic', u'evolution', u'primitive', u'panoramic', 
## 0.9996858835220337, 0.9985666275024414, 0.9977409839630127, 0.9971405267715454, 0.9963473081588745, 0.9951061606407166, 
## u'reptile', u'mammal', u'vertebrate', u'monstrous', u'monster', u'wildlife', 
## 0.9924546480178833, 0.992313802242279, 0.9914475083351135, 0.9865827560424805, 0.9778661727905273, 0.9755235314369202, 
## u'lizard', u'paleontology', u'wild'
## 0.9732449054718018, 0.9597081542015076, 0.9316698312759399, 0.9302905797958374

edges = np.array([[1,0,7,1,1,15], \
                  [0,3,0,8,9,12]]);
node_scores = np.array([0.9996858835220337, 0.9985666275024414, 0.9977409839630127, 0.9971405267715454, 0.9963473081588745, 0.9951061606407166, \
0.9924546480178833, 0.992313802242279, 0.9914475083351135, 0.9865827560424805, 0.9778661727905273, 0.9755235314369202,  \
0.9732449054718018, 0.9597081542015076, 0.9316698312759399, 0.9302905797958374]);

bounds = list(map(lambda v : (v/2,1.0), node_scores));
cons = [{'type':'eq', 'fun': constraint}]
result = optimize.minimize(sumOfMax, node_scores, method='L-BFGS-B',constraints=cons,bounds=bounds);
print(result)
