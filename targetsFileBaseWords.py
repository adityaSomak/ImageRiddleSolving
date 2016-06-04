from nltk.stem import WordNetLemmatizer
from assoc_space import AssocSpace
import sys

assocDir="/windows/drive2/For PhD/KR Lab/UMD_vision_integration/Image_Riddle/conceptnet5/data/assoc/assoc-space-5.4";
if len(sys.argv) > 2:
	assocDir = sys.argv[2];
assocSpace = AssocSpace.load_dir(assocDir);
wordnet_lemmatizer = WordNetLemmatizer()
with open(sys.argv[1], "r") as f:
	for line in f:
		targets = line.strip().split("\t");
		target = "/c/en/"+targets[0];
		targets[0] = targets[0].strip();
		if assocSpace.labels.__contains__(target):
			print targets[0];
			continue;
		multiWordTarget = '_'.join(x.strip() for x in targets[0].split(' '))
		if multiWordTarget != targets[0]:
			if assocSpace.labels.__contains__("/c/en/"+multiWordTarget):
				print multiWordTarget.strip(),"\t",targets[0];
				continue;
		else:
			target = wordnet_lemmatizer.lemmatize(targets[0]);
			if assocSpace.labels.__contains__("/c/en/"+target):
				print target.strip(),"\t",targets[0];
				continue;
			target = wordnet_lemmatizer.lemmatize(targets[0],"v");
			if assocSpace.labels.__contains__("/c/en/"+target):
				print target.strip(),"\t",targets[0];
				continue;
			print "###",targets[0]," and ",target," not found...";
