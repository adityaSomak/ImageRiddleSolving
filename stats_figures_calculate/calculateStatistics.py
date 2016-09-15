from nltk.corpus import wordnet as wn
import sys

adjCount=0;
nounCount=0;
verbCount=0;
adverbCount=0;
notFound=0;
with open(sys.argv[1], 'r') as namesFile:
	for line in namesFile:
		name = line.strip();
		s = wn.synsets(name);
		print name;
		if len(s) > 0:
			found=False
			#for synset in s:
				#lemmaNames = map(lambda x: (str(x)).lower(), synset.lemma_names());
				#if name in lemmaNames:
			pos = str(s[0].pos());
			if pos.startswith("n"):
				nounCount = nounCount+1;
			elif pos.startswith("v"):
				verbCount = verbCount+1;
			elif pos.startswith("a") or pos.startswith("s"):
				adjCount = adjCount+1;
			else:
				adverbCount=adverbCount+1;
		else:
			notFound=notFound+1;

string= str(adjCount)+","+str(nounCount)+","+str(verbCount)+","+str(adverbCount)+","+str(notFound);
print string;
