from assoc_space import AssocSpace
import numpy as np
#import pylab as Plot
import matplotlib.pyplot as Plot
from matplotlib.patches import Circle
from sklearn.manifold import TSNE

def processClarifaiJsonFile(fileName):
	with open(fileName, 'r') as myfile:
		line = myfile.read();
	line = line.replace("u\'", "");
	line = line.replace("\'", "");
	line = line.replace(":", "\n");
	lines = line.split("\n");

	detections = (lines[15][2:lines[15].index("]")]).split(",");
	if lines[15].endswith('probs'):
		weights = (lines[16][2:lines[16].index("]")]).split(",");
	else:
		weights = (lines[17][2:lines[17].index("]")]).split(",");
	return [detections, weights];
       
def filterEnglishWords(word):
	return word.encode('utf-8').startswith("/c/en")
           
def cleanUnicodeString(word):
	word = word.encode('utf-8')
	word = word.replace("u\'","")
	word = word.replace("'","")
	return word
	 
assocDir = "../../conceptnet5/data/assoc/assoc-space-5.4";
assocSpace = AssocSpace.load_dir(assocDir);
cn_words = assocSpace.labels

dir="/windows/drive2/For PhD/KR Lab/UMD_vision_integration/Image_Riddle/First250riddles/riddlesDetectionAll/"
[detections,_] = processClarifaiJsonFile(dir+"Detection/aardvark_1.txt")
detections = [word.strip().replace(" ","_") for word in detections]
#target_words = [line.strip().lower() for line in open(dir+"filelist.txt")][:500]
target_words = []
termset = set()
for word in detections:
	word  = "/c/en/"+ word
	if word in cn_words:
		vector = assocSpace.u[cn_words.index(word)]
		data = assocSpace.terms_similar_to_vector(vector,filter=filterEnglishWords,num=10)
		terms = [cleanUnicodeString(item[0]) for item in data]
		print terms
		for term in terms:
			termset.add(term)
	print "looping for "+str(word)
#pairs = []
#for term in termset:
	#if term in cn_words:
		#sim = 	assocSpace.assoc_between_two_terms("/c/en/aardvark",term);
	#pairs.append((sim,term))
#pairs = sorted(pairs,key=lambda x:x[0],reverse=True)
#for pair in pairs[:150]:
#	target_words.append(pair[1]);
target_words = set()
target_words.update(termset)
target_words.update(detections)
target_words.add("/c/en/aardvark")
target_words = list(target_words)
#print target_words

rows = []
for word in target_words:
	if not word.startswith("/c/en"):
		word  = "/c/en/"+word
	if word in cn_words:
		rows.append(cn_words.index(word))
#print rows
#rows = [cn_words.index("/c/en/"+word) for word in target_words if ("/c/en/"+word in cn_words]
target_matrix = assocSpace.u[rows,:]

print "target_matrix loaded"
reduced_matrix = TSNE(n_components=2, perplexity=40, verbose=2).fit_transform(target_matrix)

print "reduced_matrix loaded"
fig = Plot.figure(figsize=(70, 70))
max_x = np.amax(reduced_matrix, axis=0)[0]
max_y = np.amax(reduced_matrix, axis=0)[1]
Plot.xlim((-max_x,max_x))
Plot.ylim((-max_y,max_y))

mymap = Plot.get_cmap("RdYlBu")
colors = []
for word_id in rows:
	if cn_words[word_id][6:] in detections:
		 colors.append(10);
		 continue
	sim = assocSpace.assoc_between_two_terms("/c/en/aardvark",cn_words[word_id]);
	sim = (sim+1)/2.0
	# map from 100 to 200
	sim = int(sim *200 +50)
	colors.append(sim);
	
ax = fig.add_subplot(1, 1, 1)
Plot.scatter(reduced_matrix[:, 0], reduced_matrix[:, 1], 2000, c=colors, cmap=mymap); 

for row_id in range(0, len(rows)):
    target_word = cn_words[rows[row_id]]
    x = reduced_matrix[row_id, 0]
    y = reduced_matrix[row_id, 1]
    if target_word[6:] in detections:
		circ = Circle((x,y),15, facecolor="none",edgecolor='r')
		#circ.set_alpha(0.5)
		ax.add_patch(circ)
    Plot.annotate(target_word[6:], (x,y), fontsize=20)
   
Plot.show()
Plot.savefig("cn_aardvark.png");
