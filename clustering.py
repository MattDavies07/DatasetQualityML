import pandas as pd
import numpy as np
#from sklearn.model_selection import GridSearchCV
from sklearn import cluster
from scipy.spatial.distance import cdist
import pickle

# Clustering method of finding centroids and distances, taken and adapted from
# medium blog by A. Kubra Kuyucu. Can be viewed from this link:
# https://medium.datadriveninvestor.com/outlier-detection-with-k-means-clustering-in-python-ee3ac1826fb0

def dataPreprocess():
	non_ref_data = pd.read_csv("data/training/training_metrics.csv")
	ref_data = pd.read_csv("data/training/training_37_ref_metrics.csv")

	data = ref_data
	data['brisque'] = non_ref_data['brisque']
	data['pydom_sharpness'] = non_ref_data['pydom_sharpness']

	data = data.drop([16])
	data = data.drop(columns=['msssim'])

	ssim = []
	adr = []
	for index, row in data.iterrows():
		ssim.append(eval(row['ssim'])[0])
		adr.append(eval(row['adapted_rand_error'])[0])

	data['ssim'] = ssim
	data['adapted_rand_error'] = adr

	data.to_csv('data/training/pre_training_metrics.csv', index=False)
	return data

def train(data):
	X = data.iloc[:,2:].values
	km = cluster.KMeans(n_clusters=1).fit(X)
	filename = 'finalised_clustering.sav'
	pickle.dump(km, open(filename, 'wb'))

def getOutliers(data, percentile=80):
	X = data.iloc[:,2:].values

	km = pickle.load(open('finalised_clustering.sav', 'rb'))
	clusters = km.predict(X)
	centroids = km.cluster_centers_
	points = np.empty((0,len(X[0])), float)
	distances = np.empty((0,len(X[0])), float)

	for i, center_elem in enumerate(centroids):
		distances = np.append(distances, cdist([center_elem],X[clusters == i], 'euclidean')) 
		points = np.append(points, X[clusters == i], axis=0)

	outliers = points[np.where(distances > np.percentile(distances, percentile))]
	outliersArr = []
	for index, row in data.iterrows():
		for entry in outliers:
			if entry[0] == row['mse']:
				outliersArr.append(row['image'])
	print(outliersArr)
	return outliersArr

if __name__ == "__main__":
	data = dataPreprocess()
	train(data)
	getOutliers(data)
 