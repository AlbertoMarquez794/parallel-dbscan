from sklearn.cluster import DBSCAN
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import numpy as np

# Indicate the number of points
n_points = 4000

# randomly generating data points and noise
points, y_true = make_blobs(n_samples=n_points,
                            centers=4,
                            cluster_std=0.06,
                            random_state=11,
                            center_box=(0, 1.0))

# only positive points and with three decimals
points = np.round(np.abs(points[:, ::-1]), 3)

# storing points into a csv file
np.savetxt(str(n_points)+"_data.csv", points, delimiter=",",  fmt="%.3f")

# clustering and detecting noise with dbscan
clusters = DBSCAN(eps=0.03, min_samples=10).fit_predict(points)
print(clusters)
# plotting noise
