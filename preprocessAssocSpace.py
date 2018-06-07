from assoc_space import AssocSpace
import numpy as np
import os
import gzip

assocDir = "../conceptnet5/data/assoc/assoc-space-5.4"
assocSpace = AssocSpace.load_dir(assocDir)

for name in assocSpace.labels:
	if name.startswith("/c/en"):
		prefix = name[6:].replace("/","_")
		fileName = "intermediateFiles/opt/preprocessAssoc/"+prefix+""
		simFileName = "intermediateFiles/opt/preprocessAssoc/"+prefix+"_sim"
		if os.path.isfile(simFileName):
			continue
		vec = assocSpace.row_named(name)
		sim = assocSpace.assoc.dot(vec)
		indices = np.argsort(sim)[::-1]
		np.savez_compressed(fileName,indices[:1000])
		sim_first1k = np.array([sim[index] for index in indices[:1000]])
		np.savez_compressed(simFileName,sim_first1k)
		print name.encode('utf-8')," completed.."
