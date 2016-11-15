from sets import Set
import sys

wordSet = Set();
with open(sys.argv[1], "r") as f:
	for line in f:
		words = line.split("\t");
		for word in words:
			word = word.strip();
			if word !="":
				wordSet.add(word);

for word in wordSet:
	print word;
