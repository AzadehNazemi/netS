from sklearn.ensemble import IsolationForest
import sys
import pickle
from imutils import paths
import cv2
import numpy as np

def quantify_image(image, bins=(4, 6, 3)):
	hist = cv2.calcHist([image], [0, 1, 2], None, bins,
		[0, 180, 0, 256, 0, 256])
	hist = cv2.normalize(hist, hist).flatten()

	return hist

def load_dataset(datasetPath, bins):
	imagePaths = list(paths.list_images(datasetPath))
	data = []

	for imagePath in imagePaths:
		image = cv2.imread(imagePath)
		image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

		features = quantify_image(image, bins)
		data.append(features)

	return np.array(data)
    
'''
TRAIN
'''

data = load_dataset(sys.argv[1], bins=(3, 3, 3))

print("[INFO] fitting anomaly detection model...")
model = IsolationForest(n_estimators=100, contamination=0.01,
	random_state=42)
model.fit(data)
f = open("anomaly.model", "wb")
f.write(pickle.dumps(model))
f.close()
'''
PREDICT
'''

print("[INFO] loading an)omaly detection model...")
model = pickle.loads(open("anomaly.model", "rb").read())    

image = cv2.imread(sys.argv[2])
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
features = quantify_image(hsv, bins=(3, 3, 3))

preds = model.predict([features])[0]
label = "anomaly" if preds == -1 else "normal"
color = (0, 0, 255) if preds == -1 else (0, 255, 0)

cv2.putText(image, label, (10,  25), cv2.FONT_HERSHEY_SIMPLEX,
	0.7, color, 2)

cv2.imwrite("Output.jpg", image)
