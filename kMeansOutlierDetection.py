import pandas as pd
import numpy as np
from sklearn import cluster
from scipy.spatial.distance import cdist

def getData(location):
	data = pd.read_csv(location)
	return data

def getImages(data, outliers):
	outliersArr = []

	for index, row in data.iterrows():
		for entry in outliers:
			if entry[0] == row['mse']:
				outliersArr.append(row['image'])
    
	print(outliersArr)
	return outliersArr

def runKmeans(location, n_clusters, percentile):
	data = getData(location)
	X = data.iloc[:,2:].values
	km = cluster.KMeans(n_clusters=n_clusters)
	clusters = km.fit_predict(X)

	# obtaining the centers of the clusters
	centroids = km.cluster_centers_
	print(centroids)
	# points array will be used to reach the index easy
	points = np.empty((0,len(X[0])), float)
	# distances will be used to calculate outliers
	distances = np.empty((0,len(X[0])), float)
	# getting points and distances
	for i, center_elem in enumerate(centroids):
	# cdist is used to calculate the distance between center and other points
		distances = np.append(distances, cdist([center_elem],X[clusters == i], 'euclidean'))
		points = np.append(points, X[clusters == i], axis=0)

	outliers = points[np.where(distances > np.percentile(distances, percentile))]

	return outliers

