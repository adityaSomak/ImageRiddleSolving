import numpy as np
import sys
import os
import mergeTargets
import scipy.cluster.hierarchy as hac
from sklearn.cluster import KMeans
from scipy.spatial.distance import pdist,squareform
import conceptnet_util
import util


'''
#######################################################################
########		PIPELINE STAGE IV. (TODO: test it)
########		Input: target-matrix for seeds
########		Output: Cluster them and output clusters
			Experiments show:
				HAC works better single linkage
#######################################################################
'''
def getNumpyArrayOfTargetWords(sortedScoreAndIndexList, targetWordsList, targetWordsDictonary, \
numSeeds, maxTargets=3000):
	targetwordMatrix = [];
	vectorLength =1;
	modifiedSortedScoreAndIndexList=[];
	for i in range(0,len(sortedScoreAndIndexList)):
		if i > maxTargets:
			break;
		indexAndScore = sortedScoreAndIndexList[i];
		targetWord = targetWordsList[indexAndScore[0]];
		try:
			targetVector = conceptnet_util.assocSpace.row_named(util.encode("/c/en/"+targetWord));
			targetwordMatrix.append(targetVector);
			vectorLength = len(targetVector);
			modifiedSortedScoreAndIndexList.append(indexAndScore);
		except UnicodeDecodeError as ude:
			continue;
			#print "\terror ignoring";
			# TODO: take care of this
			#targetwordMatrix.append(np.zeros(vectorLength));
	return [np.vstack(targetwordMatrix),modifiedSortedScoreAndIndexList];



def returnClustersFast(sortedScoreAndIndexList, targetWordsList, targetWordsDictionary,seedWordsList, maxTargets=3000):
	[targetwordMatrix,modifiedSortedScoreAndIndexList]= getNumpyArrayOfTargetWords(sortedScoreAndIndexList, targetWordsList, \
	targetWordsDictionary, len(seedWordsList), maxTargets);
	print targetwordMatrix.shape
	print "\tcreating the linkage matrix";
	pairwiseDistancesTargetWords = pdist(targetwordMatrix,metric='cosine');
	pairwiseDistancesTargetWords = squareform(pairwiseDistancesTargetWords);
	return [modifiedSortedScoreAndIndexList,pairwiseDistancesTargetWords];
	
	
	
def returnClusters(sortedScoreAndIndexList, targetWordsList, targetWordsDictionary,seedWordsList, maxTargets=3000):
	[targetwordMatrix,modifiedSortedScoreAndIndexList]= getNumpyArrayOfTargetWords(sortedScoreAndIndexList, targetWordsList, \
	targetWordsDictionary, len(seedWordsList), maxTargets);
	print targetwordMatrix.shape
	#print targetwordMatrix[5,:];
	#print targetwordMatrix[10,:];
	print "\tcreating the linkage matrix";
	Z = hac.linkage(targetwordMatrix, method='single',metric='cosine');
	pairwiseDistancesTargetWords = pdist(targetwordMatrix,metric='cosine');
	c, coph_dists = hac.cophenet(Z,pairwiseDistancesTargetWords);
	print c;
	clusters = hac.fcluster(Z, util.RATIO_COPH_DIST_FCLUSTER_THRESHOLD_CLUSTERT*coph_dists.max(), criterion='distance');
	pairwiseDistancesTargetWords = squareform(pairwiseDistancesTargetWords);
	return [clusters,modifiedSortedScoreAndIndexList,Z,pairwiseDistancesTargetWords];

def returnClustersKMeanspp(sortedScoreAndIndexList, targetWordsList, targetWordsDictionary,seedWordsList, maxTargets=3000):
	[targetwordMatrix,modifiedSortedScoreAndIndexList]= getNumpyArrayOfTargetWords(sortedScoreAndIndexList, targetWordsList, \
	targetWordsDictionary, len(seedWordsList), maxTargets);
	print targetwordMatrix.shape
	print "creating the clusters";
	clusters =  KMeans(n_clusters=1500, init='k-means++').fit_predict(targetwordMatrix)
	return [clusters,modifiedSortedScoreAndIndexList];

	
def mergeTargetsAndReturnClusters(reweightedSeedsFileImg, allSeedsMapFile, maxTargets=3000):
	[sortedScoreAndIndexList, targetWordsList, targetWordsDictionary,seedsDetected_weights,seedWordsList,allSeedsDictionary] = \
	mergeTargetsFromDetectedSeeds(reweightedSeedsFileImg,allSeedsMapFile);
	return returnClusters(sortedScoreAndIndexList, targetWordsList, targetWordsDictionary,seedWordsList,maxTargets);



if __name__ == "__main__":
	# sys.arv[1] = reweighted set of seeds for an image
	# sys.argv[2] = detected seeds to modified map file
	# sys.argv[3] = clustering choice
	
	[sortedScoreAndIndexList, targetWordsList, targetWordsDictionary,seedsDetected_weights,seedWordsList,allSeedsDictionary] = \
	mergeTargets.mergeTargetsFromDetectedSeeds(sys.argv[1],sys.argv[2],1,0.1);
	
	if sys.argv[3] == "hac":
		[clusters,modifiedSortedScoreAndIndexList,Z,pairwiseDistancesTargetWords] = returnClusters(sortedScoreAndIndexList, targetWordsList, \
		targetWordsDictionary,seedWordsList, 3000)
	
		print clusters.shape
		for i in range(clusters.shape[0]):
			print targetWordsList[modifiedSortedScoreAndIndexList[i][0]],"\t",clusters[i],"\t**";
	
		labelsList=[];
		for c in range(0,len(modifiedSortedScoreAndIndexList)):
			try:
				clusterIndex = clusters[c];
				indexAndScore = modifiedSortedScoreAndIndexList[c];
				label = util.encode(targetWordsList[indexAndScore[0]]);
				labelsList.append(label+str(clusters[c]));
			except UnicodeDecodeError as ude:
				labelsList.append("NA");
	
		np.save("labels.npy",np.array(labelsList));
		np.save("Z.npy",Z);
	elif sys.argv[3] == "kmpp":
		[clusters,modifiedSortedScoreAndIndexList] = returnClustersKMeanspp(sortedScoreAndIndexList, \
		targetWordsList, targetWordsDictionary,seedWordsList, 3000)
		
		print clusters.shape
		for i in range(clusters.shape[0]):
			print targetWordsList[modifiedSortedScoreAndIndexList[i][0]],"\t",clusters[i],"\t**";

