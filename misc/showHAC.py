import numpy as np
from matplotlib import pyplot as plt
import scipy.cluster.hierarchy as hac

Z= np.load("Z.npy");
labelsArr = np.load("labels.npy");

ct=Z[-500,2];
#plt.figure(figsize=(50, 30))
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('words')
plt.ylabel('distance')
hac.dendrogram(
	Z,
	truncate_mode='lastp',
	p=1000,
	leaf_rotation=45.,  # rotates the x axis labels
	leaf_font_size=10.,  # font size for the x axis labels
	color_threshold=ct,
	labels=labelsArr);
plt.show();
#hac.set_link_color_palette(None);
