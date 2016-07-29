from __future__ import print_function
import os

summaryFile = 'intermediateFiles/allTargets/semanticSimSeeds.txt';
outputFile = open(summaryFile,"w");
for file in os.listdir("intermediateFiles/allTargets/test1/"):
	if file.endswith("_targets__sorted.txt"):
		numOnesNansOrZeros = [0,0,0];

		with open("intermediateFiles/allTargets/test1/"+file, "r") as f:
			i=0;
			for line in f:
				if i > 500:
					break;
				tokens = line.split("\t");
				if tokens[len(tokens)-1].strip() =="1.0":
					numOnesNansOrZeros[0] = numOnesNansOrZeros[0]+1;
				elif tokens[len(tokens)-1].strip() =="nan":
					numOnesNansOrZeros[1] = numOnesNansOrZeros[1]+1;
				elif tokens[len(tokens)-1].strip() =="0.0":
					numOnesNansOrZeros[2] = numOnesNansOrZeros[2]+1;
				i=i+1;

		word = file[:file.index("_")];
		if max(numOnesNansOrZeros)/500.0 > 0.9:
			print(word,file=outputFile);

outputFile.close();

