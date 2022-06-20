
import imutils
import sys,os,cv2

class RGBHistogram:
	def __init__(self, bins):
		self.bins = bins

	def describe(self, image):
		hist = cv2.calcHist([image], [0, 1, 2],
			None, self.bins, [0, 256, 0, 256, 0, 256])

		if imutils.is_cv2():
			hist = cv2.normalize(hist)

		else:
			hist = cv2.normalize(hist,hist)

		return hist.flatten()

import numpy as np

class Searcher:
	def __init__(self, index):
		self.index = index

	def search(self, queryFeatures):
		results = {}

		for (k, features) in self.index.items():
			d = self.chi2_distance(features, queryFeatures)

			results[k] = d

		results = sorted([(v, k) for (k, v) in results.items()])

		return results

	def chi2_distance(self, histA, histB, eps = 1e-10):
		d = 0.5 * np.sum([((a - b) ** 2) / (a + b + eps)
			for (a, b) in zip(histA, histB)])

		return d
        
        
        
        

'''
COLOR FEATURE EXTRACTION
'''
from imutils.paths import list_images
import argparse
import pickle
import cv2

index = {}

desc = RGBHistogram([8, 8, 8])

for imagePath in list_images(sys.argv[1]):
    k = imagePath[imagePath.rfind("/") + 1:]
    k=imagePath.split('/')[-1]
    
    
    
    print(k)
    image = cv2.imread(imagePath)
    
    features = desc.describe(image)
    index[k] = features

f = open("index.pickle", "wb")
f.write(pickle.dumps(index))
f.close()

print("[INFO]   done...indexed {} images".format(len(index)))




index = pickle.loads(open("index.pickle", "rb").read())
searcher = Searcher(index)

pathq = sys.argv[2]
queryImage = cv2.imread(pathq)
qfeatures = desc.describe(queryImage)

results = searcher.search(qfeatures)

(score, imageName) = results[1]
path = os.path.join(sys.argv[1], imageName)
result = cv2.imread(path)


cv2.imwrite("Results1.jpg",result)
