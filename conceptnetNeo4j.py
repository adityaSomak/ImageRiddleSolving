from conceptnet5.query import lookup
from neo4jrestclient.client import GraphDatabase
import sys

def encode(arg):
	if arg != None:
		return arg.encode('utf-8');
	return "";

def getNodeFromDB(word,idx,db):
	nodes=idx["name"][word];
	if len(nodes):
		u1 = nodes[0];
	else:
		u1 = db.nodes.create(name=word);
		#u1['completed'] =0;
		idx["name"][word]= u1;	
	return u1;

def wordMatchesWithoutPosTag(startWord,word):
	ar = startWord.split("/");
	sW = '';
	for i in range(1,4):
		sW =sW+"/"+ar[i];
	if sW == word:
		return 1;
	return 0;

def recursivelyAddNodesAndEdges(word,idx,entity,db,level):
	if level == 3:
		return;
	if not word.startswith('/c/'):
		word = str('/c/en/')+word;
	u1 = getNodeFromDB(word,idx,db);
	if word in words_dict:
		return;
	try:
		assertions = lookup(word);
		for assertion in assertions:
			relation=encode(assertion['rel']);
			if relation == "/r/InstanceOf" or relation=="/r/IsA":
				end=encode(assertion['end']);
				if end == "/c/en/person":
					return;
		for assertion in lookup(word):
			startWord = encode(assertion['start']);
			endWord = encode(assertion['end']);
			if startWord == "" or endWord == "":
				continue;
			if startWord.startswith('/c/en') and endWord.startswith('/c/en'):
				otherWord = None;
				if startWord == word or wordMatchesWithoutPosTag(startWord,word):
					u2=getNodeFromDB(endWord,idx,db);
					otherWord = endWord;
					#entity.add(u2);
				else:
					u2 = u1;
					u1=getNodeFromDB(startWord,idx,db);
					otherWord = startWord;
					#entity.add(u1);
				print otherWord," added.., level=", level,"word=",word;
				# TODO: Add only unique relationships
				u1.relationships.create(encode(assertion['rel']), u2,weight=assertion['weight']);
				recursivelyAddNodesAndEdges(otherWord,idx,entity,db,level+1);
		words_dict[word]=1;
	except UnicodeDecodeError as ude:
		print "UnicodeDecodeError found. Ignoring..."

#entities_dict={};
db = GraphDatabase("http://localhost:7474", username="neo4j", password="somak")
# Create some nodes with labels
entity = db.labels.create("Entity")
idx =  db.nodes.indexes.create("entities")
words_dict={};

if len(sys.argv) < 2:
	print "python conceptnetneo4j.py <seedsfile>";
	sys.exit();
with open(sys.argv[1], "r") as f:
	for line in f:
		word=line.strip();
		word = str('/c/en/')+word;
		n=db.nodes.create(name=word);
		#n['completed'] = 0;
		entity.add(n);
		idx["name"][n['name']]= n;

with open(sys.argv[1], "r") as f:
	i=0;
	for line in f:
		word=line.strip();
		word = str('/c/en/')+word;
		recursivelyAddNodesAndEdges(word,idx,entity,db,0);
		i = i+1;
		print str(i),":",word," processed..##########";
		if i==3:
			break;
