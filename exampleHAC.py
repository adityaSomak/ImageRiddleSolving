from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage, cophenet, fcluster
from scipy.spatial.distance import pdist
import numpy as np


np.random.seed(4711)  # for repeatability of this tutorial
a = np.random.multivariate_normal([10, 0], [[3, 1], [1, 4]], size=[100,])
b = np.random.multivariate_normal([0, 20], [[3, 1], [1, 4]], size=[50,])
X = np.concatenate((a, b),)
#print X.shape  # 150 samples with 2 dimensions
#plt.scatter(X[:,0], X[:,1])
#plt.show()

Z = linkage(X, metric='cosine');
c, coph_dists = cophenet(Z, pdist(X));

print Z[:20]
print 0.5*coph_dists.max()

#~ plt.title('Hierarchical Clustering Dendrogram (truncated)')
#~ plt.xlabel('sample index')
#~ plt.ylabel('distance')
#~ dendrogram(
    #~ Z,
    #~ truncate_mode='lastp',  # show only the last p merged clusters
    #~ p=12,  # show only the last p merged clusters
    #~ show_leaf_counts=False,  # otherwise numbers in brackets are counts
    #~ leaf_rotation=90.,
    #~ leaf_font_size=12.,
    #~ show_contracted=True,  # to get a distribution impression in truncated branches
#~ )
#~ plt.show()

clusters = fcluster(Z,0.5*coph_dists.max(), criterion='distance');
print clusters
