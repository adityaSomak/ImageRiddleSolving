from __future__ import print_function
from clarifai.client import ClarifaiApi
import sys

for i in range(1,5):
	imageName = sys.argv[1]+"_"+str(i)+".png";
	opFileName = sys.argv[1]+"_"+str(i)+".txt"
	clarifai_api = ClarifaiApi() # assumes environment variables are set.
	result = clarifai_api.tag_images(open(imageName, 'rb'))
	outputFile = open(opFileName,"w");
	print(result,file=outputFile);
