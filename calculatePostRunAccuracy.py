import conceptnet_util
import sys
import os
import argparse

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
	similarWord = None;
	with open(finalReorderedTargetsFileName, 'r') as f:
		i=0;
		for line in f:
			tokens =line.split("\t");
			if i==limitSuggestions:
				break;
			similarity = conceptnet_util.getWord2VecSimilarity(expectedWord,tokens[0].strip(),True);
			if similarity > maxSimilarity:
				maxSimilarity = similarity;
				similarWord = tokens[0].strip();
			i=i+1;
	return [maxSimilarity,similarWord];

def updateHistogram(histogram, sim):
	if sim < 0.6:
		histogram[0] = histogram[0]+1;
	elif sim <0.7:
		histogram[1] = histogram[1]+1;
	elif sim <0.8:
		histogram[2] = histogram[2]+1;
	elif sim <0.9:
		histogram[3] = histogram[3]+1;
	else:
		histogram[4] = histogram[4]+1;


if __name__ == "__main__":
	parser = argparse.ArgumentParser();
	parser.add_argument("inferenceFolder");
	parser.add_argument("maxOrAvg");
	parser.add_argument("-cleanup",action="store",default=False,type=bool);
	parser.add_argument("-summaryFile",action="store",default=None);
	parser.add_argument("-ignoreDevDataFile",default=None);
	argsdict = vars(parser.parse_args(sys.argv[1:]));

	inferenceFolder = argsdict["inferenceFolder"];
	cleanup = bool(argsdict["cleanup"]);

	calculateMax = False;
	if argsdict["maxOrAvg"] == "max":
		calculateMax = True;
	summaryFileW = None;
	if argsdict["summaryFile"]!= None:
		summaryFileW = open(argsdict["summaryFile"],'w');

	imagesInDevelopment = set();
	if argsdict["ignoreDevDataFile"]!= None:
		i=0;
		with open(argsdict["ignoreDevDataFile"],'r') as filelist:
			for line in filelist:
				imagesInDevelopment.add(line.strip());
				if i==500:
					break;
				i=i+1;

	# Less than 0.6, < 0.7, < 0.8, <0.9, <1;	
	histogram =[0,0,0,0,0];


	for root, directories, filenames in os.walk(inferenceFolder):
		totalSim = 0;
		totalDetected=0;
		for filename in filenames:
			filePath = str(os.path.join(root,filename));
			if filePath.endswith("_inf_all.txt"):
				expectedWord = filename[4:].replace("_inf_all.txt","");
				print expectedWord;
				# Ignore the word if it is used in Dev Set
				if expectedWord in imagesInDevelopment:
					continue;

				if calculateMax:
					[sim,similarWord] = calculateMaxAccuracy(expectedWord, filePath, 20);
					if summaryFileW != None:
						if similarWord == None:
							summaryFileW.write(expectedWord+"\tNONE\t"+str(sim)+"\n");
						else:
							summaryFileW.write(expectedWord+"\t"+similarWord+"\t"+str(sim)+"\n");
				else:
					sim = calculateAverageAccuracy(expectedWord, filePath, 10);	
				totalSim = sim+totalSim;
				totalDetected = totalDetected+1;
				updateHistogram(histogram, sim);
			elif cleanup:
				os.remove(filePath);
		string = str(totalDetected)+","+str(totalSim);
		print string;
		stats = "< 0.6:"+str(histogram[0])+",0.6--0.7:"+str(histogram[1])+",0.7--0.8:"+str(histogram[2])+\
		",0.8--0.9:"+str(histogram[3])+",0.9--1.0:"+str(histogram[4]);
		print stats;
		histogram = map(lambda x: float(x)/totalDetected, histogram);
		stats = "< 0.6:"+str(histogram[0])+",0.6--0.7:"+str(histogram[1])+",0.7--0.8:"+str(histogram[2])+\
		",0.8--0.9:"+str(histogram[3])+",0.9--1.0:"+str(histogram[4]);
		print stats;
	if summaryFileW != None:	
		summaryFileW.close();
