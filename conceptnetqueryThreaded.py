from conceptnet5.query import *
import sys
import threading

#class WordAndKey:
#	def __init__(self,word,key):
#		self.w = word;
#		self.k = key;

def encode(arg):
	if arg != None:
		return arg.encode('utf-8');
	return "";

def dfs_paths(FINDER,start, goal, path, level, paths,targetFilePerThread,maxTries):
	if path is None:
		path = [start]
	if start == goal:
		paths.append(path);
		print "path:",path;
		#print "paths:",paths;
	if len(paths) >= 3:
		maxTries[0]=0;
		return;	
	if level == 6:
		maxTries[0] = maxTries[0]-1;
		#print "returning on...",start,"tries remaining:",maxTries;
		return;
	nodes = [];
	try:
		for assertion in FINDER.lookup(start):
			startWord = encode(assertion['start']);
			endWord = encode(assertion['end']);
			rel = encode(assertion['rel']);
			weight = assertion['weight'];
			# TODO: Traverse only meaningful paths, previous relation should guide the new one
			if startWord.startswith('/c/en') and endWord.startswith('/c/en') and (rel not in restrictedRelations) and weight >= 0.015:
				if startWord == start:
					nodes.append(endWord);
				else:
					nodes.append(startWord);
	except UnicodeDecodeError as ude:
		print "UnicodeDecodeError found. Ignoring..."
	i=0;
	for next in set(nodes) - set(path):
		if maxTries[0] <= 0:
			break;
		dfs_paths(FINDER,next, goal, path + [next], level+1, paths,targetFilePerThread,maxTries)
		i = i+1;

def findPathsToTargetWords(word1,targetWordsFile,targetFilePerThread,FINDER):
	with open(targetWordsFile, "r") as f2:
		for line in f2:
			target = line.strip();
			maxTries = [10000];
			paths =[];
			print "finding path between:",word1," and ",target;
			targetFilePerThread.write("::"+target+"\n");
			dfs_paths(FINDER,str('/c/en/')+word1,str('/c/en/')+target,None,0,paths,targetFilePerThread,maxTries);
			print paths;
			targetFilePerThread.write(str(paths));
			targetFilePerThread.write("\n");
			print "------------"

restrictedRelations = set();
restrictedRelations.add("/r/NotIsA");
restrictedRelations.add("/r/NotHasA");
restrictedRelations.add("/r/NotCauses");
restrictedRelations.add("/r/NotMadeOf");

#from nltk.corpus import wordnet
words = [];
with open(sys.argv[1], "r") as f:
	i=0;
	for line in f:
		word=line.strip();
		words.append(word);

simMatrix = [[-1 for x in range(len(words))] for x in range(len(words))];
targetFilesPerThread =[];
threads=[]	
for i in range(0,len(words)):
	word1 = words[i];
	wordsdict = [];

	FINDER = AssertionFinder();
	targetFilePerThread = open("paths/"+word1+".txt","w");
	t = threading.Thread(target=findPathsToTargetWords, name=str(word1), args=(word1,sys.argv[2],targetFilePerThread,FINDER));
	t.start();
	threads.append(t);
	targetFilesPerThread.append(targetFilePerThread);
	if (i+1)%5==0:
		for j in range(len(threads)):
			threads[j].join();
			targetFilesPerThread[j].close();
		threads = [];
		targetFilesPerThread =[];
