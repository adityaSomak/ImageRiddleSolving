import sys


positiveImages = True;

if len(sys.argv) > 3  and sys.argv[3] == "neg":
	positiveImages = False;
	summaryFile = open('negativeSummaryFile.txt','w');
else:
	summaryFile = open('summaryFile.txt','w');


summaryFile.write("ImageName\tGUR\tBaseline\tGUR-wt\tBaseline-wt\tDifference\n");
gurDict = {};
with open(sys.argv[1],'r') as gurFile:
	for line in gurFile:
		tokens = line.split("\t");
		tokens = map(lambda x:x.strip(),tokens); 
		if len(tokens) > 2:
			gurDict[tokens[0]] = (tokens[1],float(tokens[2]));
			
with open(sys.argv[2],'r') as clarifaiFile:
	for line in clarifaiFile:
		tokens = line.split("\t");
		tokens = map(lambda x:x.strip(),tokens); 
		if tokens[0] in gurDict:
			diff = gurDict[tokens[0]][1] - float(tokens[2]);
			if positiveImages and diff > 0.1:
				summaryFile.write(tokens[0]+"\t"+gurDict[tokens[0]][0]+"\t"+tokens[1]+"\t"+\
					str(gurDict[tokens[0]][1])+"\t"+str(tokens[2])+"\t"+str(diff)+"\n");
			elif (not positiveImages) and diff < -0.1:
				summaryFile.write(tokens[0]+"\t"+gurDict[tokens[0]][0]+"\t"+tokens[1]+"\t"+\
					str(gurDict[tokens[0]][1])+"\t"+str(tokens[2])+"\t"+str(diff)+"\n");

summaryFile.close();
