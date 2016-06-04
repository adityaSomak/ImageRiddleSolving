from conceptnet5.query import *
import sys

class WordAndKey:
	def __init__(self,word,key):
		self.w = word;
		self.k = key;

def encode(arg):
	if arg != None:
		return arg.encode('utf-8');
	return "";

def dfs_paths(start, goal, path, level, paths):
	global maxTries
	if path is None:
		path = [start]
	if start == goal:
		paths.append(path);
		print "path:",path;
		#print "paths:",paths;
	if len(paths) >= 3:
		maxTries=0;
		return;	
	if level == 7:
		maxTries = maxTries-1;
		print "returning on...",start,"tries remaining:",maxTries;
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
		if maxTries <= 0:
			break;
		dfs_paths(next, goal, path + [next], level+1, paths)
		i = i+1;

restrictedRelations = set();
restrictedRelations.add("/r/NotIsA");
restrictedRelations.add("/r/NotHasA");
restrictedRelations.add("/r/NotCauses");
restrictedRelations.add("/r/NotMadeOf");
maxTries = 100000;

if len(sys.argv) == 2:
	#print "python conceptnetquery.py <word>";
	word = str('/c/en/')+str(sys.argv[1]);
	assertions = lookup(word);
	for assertion in assertions:
		if assertion['end'].startswith('/c/en') and assertion['start'].startswith('/c/en'):
			print encode(assertion['end']),",",encode(assertion['start']),",",encode(assertion['surfaceText']),",",\
			str(assertion['weight']),",",encode(assertion['rel']);
elif len(sys.argv) == 3:
	word1 = str('/c/en/')+str(sys.argv[1]);
	word2 = str('/c/en/')+str(sys.argv[2]);
	
	paths =[];
	dfs_paths(word1,word2,None,0,paths);
	print paths
elif len(sys.argv) == 4:
	from nltk.corpus import wordnet
	words = [];
	with open(sys.argv[1], "r") as f:
		i=0;
		for line in f:
			word=line.strip();
			#word = str('/c/en/')+word;
			words.append(word);

	simMatrix = [[-1 for x in range(len(words))] for x in range(len(words))];
	for i in range(0,len(words)):
		word1 = words[i];
		wordsdict = [];
		
		with open(sy.argv[2], "r") as f2:
			for line in f2:
				target = line.strip();				
				paths =[];
				print "finding path between:",word1," and ",target;
				dfs_paths(str('/c/en/')+word1,str('/c/en/')+target,None,0,paths)
				maxTries = 10000;
				print paths
				print "------------"		
else:
	criteria={};
	criteria['rel']='/r/InstanceOf';
	criteria['end']='/c/en/person';
	
	assertions = query(criteria);
	i=0;
	for assertion in assertions:
		i=i+1;
		if assertion['end'].startswith('/c/en') and assertion['start'].startswith('/c/en'):
			print encode(assertion['end']),",",encode(assertion['start']),",",encode(assertion['surfaceText']),",",encode(assertion['rel']);
		if i > 10:
			break;
