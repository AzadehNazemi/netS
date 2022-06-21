import numpy as np
from imutils import paths
import vptree
import pickle
import cv2,sys

def dhash(image, hashSize=8):
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	
	resized = cv2.resize(gray, (hashSize + 1, hashSize))

	diff = resized[:, 1:] > resized[:, :-1]

	return sum([2 ** i for (i, v) in enumerate(diff.flatten()) if v])

def convert_hash(h):
	return int(np.array(h, dtype="float64"))

def hamming(a, b):
	return bin(int(a) ^ int(b)).count("1")



imagePaths = list(paths.list_images(sys.argv[1]))
hashes = {}

for (i, imagePath) in enumerate(imagePaths):
	print("[INFO] processing image {}/{}".format(i + 1,
		len(imagePaths)))
	image = cv2.imread(imagePath)

	h = dhash(image)
	h = convert_hash(h)

	l = hashes.get(h, [])
	l.append(imagePath)
	hashes[h] = l

print("[INFO] building VP-Tree...")
points = list(hashes.keys())
tree = vptree.VPTree(points, hamming)

print("[INFO] serializing VP-Tree...")
f = open("tree.pickle", "wb")
f.write(pickle.dumps(tree))
f.close()

print("[INFO] serializing hashes...")
f = open("hashes.pickle", "wb")
f.write(pickle.dumps(hashes))
f.close()

print("[INFO] loading VP-Tree and hashes...")
tree = pickle.loads(open("tree.pickle", "rb").read())
hashes = pickle.loads(open("hashes.pickle", "rb").read())
image = cv2.imread(sys.argv[2])
cv2.imshow("Query", image)
queryHash = dhash(image)
queryHash = convert_hash(queryHash)

print("[INFO] performing search...")
distance=10
results = tree.get_all_in_range(queryHash,distance)
results = sorted(results)

for (d, h) in results:
	resultPaths = hashes.get(h, [])
	print("[INFO] {} total image(s) with d: {}, h: {}".format(
		len(resultPaths), d, h))
	for resultPath in resultPaths:
		result = cv2.imread(resultPath)
		cv2.imshow("Result", result)
		cv2.waitKey(0)
