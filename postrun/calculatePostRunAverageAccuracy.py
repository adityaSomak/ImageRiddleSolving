import conceptnet_util
import sys
import os

'''
 This file is called after the IUR/UR/GUR scripts are run and it calculates 
 an accuracy comparing the target answer and the final predicted words in
 _inf_all.txt files.

 NOTE: Currently as a cleanup, it can be used to delete all other files to save space.
'''
def calculateAverageAccuracy(expectedWord, finalReorderedTargetsFileName, limitSuggestions=10):
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

def calculateMaxAccuracy(expectedWord, finalReorderedTargetsFileName, limitSuggestions=10):
	if "-" in expectedWord:
		expectedWord=expectedWord[:expectedWord.index("-")];
	maxSimilarity = 0.0;
	with open(finalReorderedTargetsFileName, 'r') as f:
		i=0.0;
		for line in f:
			tokens =line.split("\t");
			if i==limitSuggestions:
				break;
			similarity = conceptnet_util.getWord2VecSimilarity(expectedWord,tokens[0].strip(),True);
			if similarity > maxSimilarity:
				maxSimilarity = similarity;
			i=i+1;
	return maxSimilarity;

if __name__ == "__main__":
	cleanup = False
	inferenceFolder = sys.argv[1];
	calculateMax = False;
	if sys.argv[2] == "max":
		calculateMax = True;
	if len(sys.argv) > 3 and sys.argv[3] == "del":
		cleanup = True;

	for root, directories, filenames in os.walk(inferenceFolder):
		totalSim = 0;
		totalDetected=0;
		for filename in filenames:
			filePath = str(os.path.join(root,filename));
			if filePath.endswith("_inf_all.txt"):
				expectedWord = filename[4:].replace("_inf_all.txt","");
				print expectedWord;
				if calculateMax:
					sim = calculateMaxAccuracy(expectedWord, filePath, 20);
				else:
					sim = calculateAverageAccuracy(expectedWord, filePath, 10);	
				totalSim = sim+totalSim;
				totalDetected = totalDetected+1;
			elif cleanup:
				os.remove(filePath);
		string = str(totalDetected)+","+str(totalSim);
		print string;
