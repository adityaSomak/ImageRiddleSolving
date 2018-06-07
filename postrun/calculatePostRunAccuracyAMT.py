import conceptnet_util
import SimilarityUtil
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
		expectedWord=expectedWord[:expectedWord.index("-")]
	avgSimilarity = 0.0
	with open(finalReorderedTargetsFileName, 'r') as f:
		i=0.0
		for line in f:
			tokens =line.split("\t")
			if i==limitSuggestions:
				break
			similarity = conceptnet_util.getWord2VecSimilarity(expectedWord,tokens[0].strip(),True)
			avgSimilarity = avgSimilarity+similarity
			i=i+1
	avgSimilarity = avgSimilarity/i
	return avgSimilarity

def calculateMaxAccuracy(expectedWord, finalReorderedTargetsFileName, predicted_words=None, 
	limitSuggestions=10):
	if "-" in expectedWord:
		expectedWord=expectedWord[:expectedWord.index("-")]
	maxSimilarity = 0.0
	similarWord = None
	if finalReorderedTargetsFileName is None:
		for phrase in predicted_words:
			similarity = conceptnet_util.getWord2VecSimilarity(expectedWord,phrase,True)
			if similarity > maxSimilarity:
				maxSimilarity = similarity
				similarWord = phrase
	else:
		with open(finalReorderedTargetsFileName, 'r') as f:
			i=0
			for line in f:
				tokens =line.split("\t")
				if i==limitSuggestions:
					break
				similarity = conceptnet_util.getWord2VecSimilarity(expectedWord,tokens[0].strip(),True)
				if similarity > maxSimilarity:
					maxSimilarity = similarity
					similarWord = tokens[0].strip()
				i=i+1
	return [maxSimilarity,similarWord]

def calculateMaxWordnetAccuracy(expectedWord, finalReorderedTargetsFileName, 
	predicted_words=None, limitSuggestions=10):
	if "-" in expectedWord:
		expectedWord=expectedWord[:expectedWord.index("-")]
	maxSimilarity = 0.0
	similarWord = None
	if finalReorderedTargetsFileName is None:
		for phrase in predicted_words:
			words = phrase.lower().split(" ")
			avg_sim = 0
			for word in words:
				similarity = SimilarityUtil.word_similarity(expectedWord,word)
				avg_sim += similarity
			avg_sim = avg_sim/float(len(words))
			if avg_sim > maxSimilarity:
				maxSimilarity = avg_sim
				similarWord = phrase
	else:
		with open(finalReorderedTargetsFileName, 'r') as f:
			i=0
			for line in f:
				tokens =line.split("\t")
				if i==limitSuggestions:
					break
				words = tokens[0].strip().split("_")
				avg_sim = 0
				for word in words:
					similarity = SimilarityUtil.word_similarity(expectedWord,word)
					avg_sim += similarity
				avg_sim = avg_sim/float(len(words))
				if avg_sim > maxSimilarity:
					maxSimilarity = avg_sim
					similarWord = tokens[0].strip()
				i=i+1
	return [maxSimilarity,similarWord]

def updateHistogram(histogram, sim):
	if sim < 0.6:
		histogram[0] = histogram[0]+1
	elif sim <0.7:
		histogram[1] = histogram[1]+1
	elif sim <0.8:
		histogram[2] = histogram[2]+1
	elif sim <0.9:
		histogram[3] = histogram[3]+1
	else:
		histogram[4] = histogram[4]+1


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("amtFile")
	parser.add_argument("maxOrAvg")
	parser.add_argument("-cleanup",action="store",default=False,type=bool)
	parser.add_argument("-summaryFile",action="store",default=None)
	parser.add_argument("-ignoreDevDataFile",default=None)
	parser.add_argument("-useK",default=20,type=int)      
	argsdict = vars(parser.parse_args(sys.argv[1:]))

	amtCSVFile = argsdict["amtFile"]
	cleanup = bool(argsdict["cleanup"])

	calculateMax = False
	if argsdict["maxOrAvg"] == "max":
		calculateMax = True
	summaryFileW = None
	if argsdict["summaryFile"]!= None:
		summaryFileW = open(argsdict["summaryFile"],'w')

	imagesInDevelopment = set()
	if argsdict["ignoreDevDataFile"]!= None:
		i=0
		with open(argsdict["ignoreDevDataFile"],'r') as filelist:
			for line in filelist:
				imagesInDevelopment.add(line.strip())
				if i==500:
					break
				i=i+1

	# Less than 0.6, < 0.7, < 0.8, <0.9, <1	
	histogram =[0,0,0,0,0]

	import csv
	totalSim = 0
	totalDetected = 0
	totalSim_at1 = 0
	with open(amtCSVFile, 'rb') as csvfile:
		amtreader = csv.reader(csvfile, delimiter=',')
		amtreader.next()
		for row in amtreader:
			# print row
			# import pdb
			# pdb.set_trace()
			expectedWord = row[31].replace("\"","")
			answer_text = row[32].lower().replace("\"","")
			predicted_words = row[32].replace("\"","").lower().split(",")

			if row[17].strip() == "Rejected":
				continue 
			if "not found" in answer_text or "image_url" in answer_text \
				or "nothing shows" in answer_text or "could not view" in answer_text\
				or "blank" in answer_text:
				continue

			if calculateMax:
				# [sim,similarWord] = calculateMaxAccuracy(expectedWord, filePath, 20)
				try:
					[sim_at_1,_] = calculateMaxAccuracy(expectedWord, None, predicted_words, 1)
					[sim,similarWord] = calculateMaxAccuracy(expectedWord, 
						None, predicted_words, argsdict["useK"]) #20)
					# [sim_at_1,_] = calculateMaxWordnetAccuracy(expectedWord, None, predicted_words, 1)
					# [sim,similarWord] = calculateMaxWordnetAccuracy(expectedWord, 
					# 	None, predicted_words, argsdict["useK"]) #20)
					if summaryFileW != None:
						if similarWord == None:
							summaryFileW.write(expectedWord+"\tNONE\t"+str(sim)+"\n")
						else:
							summaryFileW.write(expectedWord+"\t"+similarWord+"\t"+str(sim)+"\n")
				except Exception, e:
					raise e
					sim = 0
					sim_at_1 = 0
					totalDetected -= 1
			else:
				sim = calculateAverageAccuracy(expectedWord, None, predicted_words, 10)
			if similarWord is not None and similarWord != "None":
				totalSim = sim + totalSim
				totalSim_at1 += sim_at_1
				totalDetected = totalDetected + 1
				updateHistogram(histogram, sim)
				print "GT: %s, Pred: %s: (%f) " % (expectedWord, similarWord, sim)
		string = str(totalDetected)+","+str(totalSim)
		print "Examples: %d, Total: %f, Avg: %f" % (totalDetected, totalSim, (totalSim/float(totalDetected)))
		print "Avg sim @1: %f" % (totalSim_at1/float(totalDetected))
		stats = "< 0.6:"+str(histogram[0])+",0.6--0.7:"+str(histogram[1])+",0.7--0.8:"+str(histogram[2])+\
		",0.8--0.9:"+str(histogram[3])+",0.9--1.0:"+str(histogram[4])
		print stats
		histogram = map(lambda x: float(x)/totalDetected, histogram)
		stats = "< 0.6:"+str(histogram[0])+",0.6--0.7:"+str(histogram[1])+",0.7--0.8:"+str(histogram[2])+\
		",0.8--0.9:"+str(histogram[3])+",0.9--1.0:"+str(histogram[4])
		print stats
	if summaryFileW != None:	
		summaryFileW.close()
