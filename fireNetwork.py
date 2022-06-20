from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from pyimagesearch.learningratefinder import LearningRateFinder
from pyimagesearch.firedetectionnet import FireDetectionNet
from pyimagesearch import config
from imutils import paths
import numpy as np
import cv2
import sys
'''
variables
FIRE_PATH
NON_FIRE_PATH
NUM_EPOCHS
INIT_LR 
TEST_SPLIT
MODEL_PATH
OUTPUT_IMAGE_PATH
SAMPLE_SIZE
'''
def load_dataset(datasetPath):
	imagePaths = list(paths.list_images(datasetPath))
	data = []

	for imagePath in imagePaths:
		image = cv2.imread(imagePath)
		image = cv2.resize(image, (128, 128))

		data.append(image)

	return np.array(data, dtype="float32")

print("[INFO] loading data...")
fireData = load_dataset(FIRE_PATH)
nonFireData = load_dataset(NON_FIRE_PATH)

fireLabels = np.ones((fireData.shape[0],))
nonFireLabels = np.zeros((nonFireData.shape[0],))

data = np.vstack([fireData, nonFireData])
labels = np.hstack([fireLabels, nonFireLabels])
data /= 255

labels = to_categorical(labels, num_classes=2)
classTotals = labels.sum(axis=0)
classWeight = classTotals.max() / classTotals

(trainX, testX, trainY, testY) = train_test_split(data, labels,
	test_size=TEST_SPLIT, random_state=42)

aug = ImageDataGenerator(
	rotation_range=30,
	zoom_range=0.15,
	width_shift_range=0.2,
	height_shift_range=0.2,
	shear_range=0.15,
	horizontal_flip=True,
	fill_mode="nearest")

print("[INFO] compiling model...")
opt = SGD(lr=INIT_LR, momentum=0.9,
	decay=INIT_LR / NUM_EPOCHS)
model = FireDetectionNet.build(width=128, height=128, depth=3,
	classes=2)
model.compile(loss="binary_crossentropy", optimizer=opt,
	metrics=["accuracy"])
print("[INFO] finding learning rate...")
lrf = LearningRateFinder(model)
lrf.find(
    aug.flow(trainX, trainY, batch_size=config.BATCH_SIZE),
    1e-10, 1e+1,
    stepsPerEpoch=np.ceil((trainX.shape[0] / float(BATCH_SIZE))),
    epochs=20,
    batchSize=BATCH_SIZE,
    classWeight=classWeight)

print("[INFO] training network...")
H = model.fit_generator(
	aug.flow(trainX, trainY, batch_size=BATCH_SIZE),
	validation_data=(testX, testY),
	steps_per_epoch=trainX.shape[0] // BATCH_SIZE,
	epochs=NUM_EPOCHS,
	class_weight=classWeight,
	verbose=1)

print("[INFO] evaluating network...")
predictions = model.predict(testX, batch_size=BATCH_SIZE)
print(classification_report(testY.argmax(axis=1),
	predictions.argmax(axis=1), target_names=config.CLASSES))

print("[INFO] serializing network to '{}'...".format(config.MODEL_PATH))
model.save(MODEL_PATH)
'''
PREDICT
'''
from tensorflow.keras.models import load_model
from pyimagesearch import config
from imutils import paths
import numpy as np
import imutils
import random
import cv2
import os

print("[INFO] loading model...")
model = load_model(MODEL_PATH)

print("[INFO] predicting...")
firePaths = list(paths.list_images(FIRE_PATH))
nonFirePaths = list(paths.list_images(NON_FIRE_PATH))

imagePaths = firePaths + nonFirePaths
random.shuffle(imagePaths)
imagePaths = imagePaths[:SAMPLE_SIZE]

for (i, imagePath) in enumerate(imagePaths):
	image = cv2.imread(imagePath)
	output = image.copy()

	image = cv2.resize(image, (128, 128))
	image = image.astype("float32") / 255.0
		
	preds = model.predict(np.expand_dims(image, axis=0))[0]
	j = np.argmax(preds)
	label = config.CLASSES[j]

	text = label if label == "NoFire" else "Fire!"
	output = imutils.resize(output, width=500)
	cv2.putText(output, text, (35, 50), cv2.FONT_HERSHEY_SIMPLEX,
		1.25, (0, 255, 0), 5)

	filename = "{}.png".format(i)
	p = os.path.sep.join([OUTPUT_IMAGE_PATH, filename])
	cv2.imwrite(p, output)
