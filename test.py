from kMeansOutlierDetection import runKmeans
from kMeansOutlierDetection import getImages
from kMeansOutlierDetection import getData

res = runKmeans('data/training/pre_training_metrics.csv', 2, 80)
print(res)

data = getData('data/training/pre_training_metrics.csv')

img = getImages(data, res)
print(img)