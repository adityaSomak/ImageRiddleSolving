from conceptnet5.query import *
from py2neo import Graph,authenticate
from py2neo import Node,Relationship
from py2neo import neo4j
import sys
import threading
import time

def encode(arg):
	if arg != None:
		return arg.encode('utf-8');
	return "";

def getNodeFromDB(word,idx,graph):
	nodes = idx.get("name",word);
	if len(nodes):
		u1 = nodes[0];
	else:
		u1 = graph.merge_one("Entity", "name", word);
		#u1['completed'] =0;
		idx.add("name",word,u1);
	return u1;

def wordMatchesWithoutPosTag(startWord,word):
	ar = startWord.split("/");
	sW = '';
	for i in range(1,4):
		sW =sW+"/"+ar[i];
	if sW == word:
		return 1;
	return 0;

'''
  word = the query node
  idx = index to get the referenced node easily
  graph = connection to the database
  FINDER = a new conceptnet instance for each thread
  level = level of traversal.
'''
def recursivelyAddNodesAndEdges(word,idx,graph,FINDER,level):
	if level == 3:
		return;
	if not word.startswith('/c/'):
		word = str('/c/en/')+word;
	u1 = getNodeFromDB(word,idx,graph);
	rlock = threading.RLock();
	with rlock:
		if word in words_dict:
			return;
		words_dict[word]=1;
	try:
		#assertions = lookup(word);
		#for assertion in assertions:
		#	relation=encode(assertion['rel']);
		#	if relation == "/r/InstanceOf" or relation=="/r/IsA":
		#		end=encode(assertion['end']);
		#		if end == "/c/en/person":
		#			return;
		otherWords = [];
		for assertion in FINDER.lookup(word):
			startWord = encode(assertion['start']);
			endWord = encode(assertion['end']);
			if startWord == "" or endWord == "":
				continue;
			if startWord.startswith("-") or startWord.endswith("-"):
				continue;
			if endWord.startswith("-") or endWord.endswith("-"):
				continue;
			if startWord.startswith('/c/en') and endWord.startswith('/c/en'):
				otherWord = None;
				if startWord == word or wordMatchesWithoutPosTag(startWord,word):
					u2=getNodeFromDB(endWord,idx,graph);
					otherWord = endWord;
					#entity.add(u2);
				else:
					u2 = u1;
					u1=getNodeFromDB(startWord,idx,graph);
					otherWord = startWord;
					#entity.add(u1);
				print otherWord," added.., level=", level,"word=",word;
				# TODO: Add only unique relationships
				graph.create(Relationship(u1,encode(assertion['rel']),u2,weight=assertion['weight']));
				otherWords.append(otherWord);
		## This converts the problem to BFS. Not-so memory intensive
		for otherWord in otherWords:
			recursivelyAddNodesAndEdges(otherWord,idx,graph,FINDER,level+1);		
	except UnicodeDecodeError as ude:
		print "UnicodeDecodeError found. Ignoring..."

# Authenticate and create graph
authenticate("localhost:7474", "neo4j", "somak");
graph = Graph();
idx = graph.legacy.get_or_create_index(neo4j.Node, "Entities")
# TODO: everytime database is reset, create the constraint
#graph.schema.create_uniqueness_constraint("Entity", "name")
words_dict={};

if len(sys.argv) < 2:
	print "python conceptnetneo4j.py <seedsfile>";
	sys.exit();
with open(sys.argv[1], "r") as f:
	for line in f:
		word=line.strip();
		word = str('/c/en/')+word;
		n = graph.merge_one("Entity", "name", word);
		#n['completed'] = 0;
		idx.add("name",encode(n["name"]),n);

with open(sys.argv[1], "r") as f:
	i=0;
	threads = [];
	for line in f:
		word=line.strip();
		word = str('/c/en/')+word;
		FINDER = AssertionFinder();
		t = threading.Thread(target=recursivelyAddNodesAndEdges, name=str(i), args=(word,idx,graph,FINDER,0));
		t.start();
		threads.append(t);
		i = i+1;
		if i%10 == 0:
			for t in threads:
				t.join();
				print "#####Thread ", t.getname(),"finishes..."; 
			threads=[];
		print str(i),":",word," processed..##########";
		if i==20:
			break;

	for t in threads:
			t.join();
