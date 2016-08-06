import conceptnet_util
import sys
import os

def calculateRelativeAccuracy(expectedWord, finalReorderedTargetsFileName, limitSuggestions=10):
	if "-" in expectedWord:
		expectedWord=expectedWord[:expectedWord.index("-")];
	avgSimilarity = 0.0;
	with open(finalReorderedTargetsFileName, 'r') as f:
		i=0.0;
		for line in f:
			tokens =line.split("\t");
			if i==limitSuggestions:
				break;
			similarity = conceptnet_util.getWord2VecSimilarity(expectedWord,tokens[0].strip(),True);
			avgSimilarity = avgSimilarity+similarity;
			i=i+1;
	avgSimilarity = avgSimilarity/i;
	return avgSimilarity;


for root, directories, filenames in os.walk(sys.argv[1]):
	totalSim = 0;
	totalDetected=0;
	for filename in filenames:
		filePath = str(os.path.join(root,filename));
		if filePath.endswith("_inf_all.txt"):
			expectedWord = filename[4:].replace("_inf_all.txt","");
			print expectedWord;
			sim = calculateRelativeAccuracy(expectedWord, filePath, 10);
			totalSim = sim+totalSim;
			totalDetected = totalDetected+1;
	string = str(totalDetected)+","+str(totalSim);
	print string;