#!/usr/bin/env python

import string, numpy, gzip
from sklearn.manifold import TSNE
from assoc_space import AssocSpace

#o = gzip.open("testdata/english-embeddings.turian.txt.gz", "rb")
#titles, x = [], []
#for l in o:
#    toks = string.split(l)
#    titles.append(toks[0])
#    x.append([float(f) for f in toks[1:]])
#x = numpy.array(x)

assocDir = "../conceptnet5/data/assoc/assoc-space-5.4";
assocSpace = AssocSpace.load_dir(assocDir);

#from tsne import tsne
#from calc_tsne import tsne
#out = tsne(x, no_dims=2, perplexity=30, initial_dims=30, USE_PCA=False)
#out = tsne(x, no_dims=2, perplexity=30, initial_dims=30, use_pca=False)
out = TSNE(n_components=2, perplexity=40, verbose=2).fit_transform(assocSpace.u)
#TSNE(x, n_components=2, perplexity=30, initial_dims=30)

import render
render.render([(title, point[0], point[1]) for title, point in zip(assocSpace, out)], "test-output.rendered.png", width=3000, height=1800)
