import conceptnet_util
import sys


def getMostSimilarWord(expectedWord, finalReorderedTargetsFileName, limitSuggestions=10):
	if "-" in expectedWord:
		expectedWord=expectedWord[:expectedWord.index("-")]
	maxSimilarity = 0.0
	similarWord = None
	try:
		with open(finalReorderedTargetsFileName, 'r') as f:
			i=0
			for line in f:
				tokens =line.split("\t")
				if i==limitSuggestions:
					break
				if not tokens[0].isupper():
					similarity = conceptnet_util.getWord2VecSimilarity(expectedWord,tokens[0].strip(),True)
					if similarity > maxSimilarity:
						maxSimilarity = similarity
						similarWord = tokens[0]
					i=i+1
	except IOError:
		return None
	return similarWord

if __name__ == "__main__":
	filelist = "all3333RiddleNames.txt"
	iurInferenceFolder = sys.argv[1]
	baselineInferenceFolder = sys.argv[2]
	gurInferenceFolder = sys.argv[3]
	numberOfImages = 3333
	baseURL = "http://www.umiacs.umd.edu/~yzyang/riddles/"
	amtCSVFile = open("misc/amtCSV.csv",'w')

	amtCSVFile.write("image_url1,image_url2,image_url3,image_url4,text1,text2,text3,text4\n")
	with open(filelist,'r') as f:
		i=0
		for line in f:
			riddlePrefix = line.strip()
			iurWord = getMostSimilarWord(riddlePrefix, iurInferenceFolder+"opt_"+riddlePrefix+"_inf_all.txt",20)
			baselineWord = getMostSimilarWord(riddlePrefix, baselineInferenceFolder+"opt_"+riddlePrefix+"_inf_all.txt",20)
			gurWord = getMostSimilarWord(riddlePrefix, gurInferenceFolder+"opt_"+riddlePrefix+"_inf_all.txt",20)
			opString = ""
			print riddlePrefix
			if iurWord!= None and gurWord != None and baselineWord != None:
				for j in range(1,5):
					opString = opString+baseURL+riddlePrefix+"_"+str(j)+".png,"
				#opString = opString+riddlePrefix+","+iurWord.replace("_"," ")+","+baselineWord.replace("_"," ")+"\n"
				opString = opString+riddlePrefix+","+iurWord.replace("_"," ")+","+gurWord.replace("_"," ")+","+baselineWord.replace("_"," ")+"\n"
				amtCSVFile.write(opString)
			i=i+1
			if i%25==0:
				amtCSVFile.flush()
			if i == numberOfImages:
				break
		amtCSVFile.close()