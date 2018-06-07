import argparse
import sys

if len(sys.argv) < 4:
	print("python ",sys.argv[0]," <seedsCentralityfile> <detectionsFolder> <number-of-puzzles> \
		<inferenceFolder> <stage> <from>,<to> <api_used> parallel")
	print("Stage options are: clarifai/merge/all.")
	print("API_used options are: clarifai/resnet.")
	sys.exit()

parser = argparse.ArgumentParser()
parser.add_argument("seedsCentralityfile")
parser.add_argument("detectionsFolder")
parser.add_argument("numPuzzles")
parser.add_argument("inferenceFolder")
parser.add_argument("api",action="store", choices=["clarifai","resnet"])
parser.add_argument("-stage",action="store")
parser.add_argument("-from",action="store")
parser.add_argument("-to",action="store")
parser.add_argument("-par",action="store")
args = parser.parse_args(sys.argv[1:])
dct = vars(args)
for key in dct:
	print key,"\t",dct[key]