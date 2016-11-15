import os
import re
import sys

for root, directories, filenames in os.walk(sys.argv[1]):
	for filename in filenames:
		filePath = str(os.path.join(root,filename));
		if filePath.endswith(".txt"):
			print "Detection/"+filename
