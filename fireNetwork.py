

import os

FIRE_PATH = os.Fire"
NON_FIRE_PATH = "noFire"

CLASSES = ["Non-Fire", "Fire"]
TRAIN_SPLIT = 0.75
TEST_SPLIT = 0.25
INIT_LR = 1e-2
BATCH_SIZE = 64
NUM_EPOCHS = 50
MODEL_PATH = os.path.sep.join(["output", "fire_detection.model"])
LRFIND_PLOT_PATH = os.path.sep.join(["output", "lrfind_plot.png"])
TRAINING_PLOT_PATH = os.path.sep.join(["output", "training_plot.png"])
OUTPUT_IMAGE_PATH = os.path.sep.join(["output", "examples"])
SAMPLE_SIZE = 50

from tensorflow.keras.callbacks import LambdaCallback
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt
import numpy as np
import tempfile

class LearningRateFinder:
	def __init__(self, model, stopFactor=4, beta=0.98):
		self.model = model
		self.stopFactor = stopFactor
		self.beta = beta

		self.lrs = []
		self.losses = []

		self.lrMult = 1
		self.avgLoss = 0
		self.bestLoss = 1e9
		self.batchNum = 0
		self.weightsFile = None

	def reset(self):
		self.lrs = []
		self.losses = []
		self.lrMult = 1
		self.avgLoss = 0
		self.bestLoss = 1e9
		self.batchNum = 0
		self.weightsFile = None

	def is_data_iter(self, data):
		iterClasses = ["NumpyArrayIterator", "DirectoryIterator",
			 "DataFrameIterator", "Iterator", "Sequence"]

		return data.__class__.__name__ in iterClasses

	def on_batch_end(self, batch, logs):
		lr = K.get_value(self.model.optimizer.lr)
		self.lrs.append(lr)

		l = logs["loss"]
		self.batchNum += 1
		self.avgLoss = (self.beta * self.avgLoss) + ((1 - self.beta) * l)
		smooth = self.avgLoss / (1 - (self.beta ** self.batchNum))
		self.losses.append(smooth)

		stopLoss = self.stopFactor * self.bestLoss

		if self.batchNum > 1 and smooth > stopLoss:
			self.model.stop_training = True
			return

		if self.batchNum == 1 or smooth < self.bestLoss:
			self.bestLoss = smooth

		lr *= self.lrMult
		K.set_value(self.model.optimizer.lr, lr)

	def find(self, trainData, startLR, endLR, epochs=None,
		stepsPerEpoch=None, batchSize=32, sampleSize=2048,
		classWeight=None, verbose=1):
		self.reset()

		useGen = self.is_data_iter(trainData)

		if useGen and stepsPerEpoch is None:
			msg = "Using generator without supplying stepsPerEpoch"
			raise Exception(msg)

		elif not useGen:
			numSamples = len(trainData[0])
			stepsPerEpoch = np.ceil(numSamples / float(batchSize))

		if epochs is None:
			epochs = int(np.ceil(sampleSize / float(stepsPerEpoch)))

		numBatchUpdates = epochs * stepsPerEpoch

		self.lrMult = (endLR / startLR) ** (1.0 / numBatchUpdates)

		self.weightsFile = tempfile.mkstemp()[1]
		self.model.save_weights(self.weightsFile)

		origLR = K.get_value(self.model.optimizer.lr)
		K.set_value(self.model.optimizer.lr, startLR)

		callback = LambdaCallback(on_batch_end=lambda batch, logs:
			self.on_batch_end(batch, logs))

		if useGen:
			self.model.fit_generator(
				trainData,
				steps_per_epoch=stepsPerEpoch,
				epochs=epochs,
				class_weight=classWeight,
				verbose=verbose,
				callbacks=[callback])

		else:
			self.model.fit(
				trainData[0], trainData[1],
				batch_size=batchSize,
				epochs=epochs,
				class_weight=classWeight,
				callbacks=[callback],
				verbose=verbose)

		self.model.load_weights(self.weightsFile)
		K.set_value(self.model.optimizer.lr, origLR)

	def plot_loss(self, skipBegin=10, skipEnd=1, title=""):
		lrs = self.lrs[skipBegin:-skipEnd]
		losses = self.losses[skipBegin:-skipEnd]

		plt.plot(lrs, losses)
		plt.xscale("log")
		plt.xlabel("Learning Rate (Log Scale)")
		plt.ylabel("Loss")

		if title != "":
			plt.title(title)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import SeparableConv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense

class FireDetectionNet:
	@staticmethod
	def build(width, height, depth, classes):
		model = Sequential()
		inputShape = (height, width, depth)
		chanDim = -1

		model.add(SeparableConv2D(16, (7, 7), padding="same",
			input_shape=inputShape))
		model.add(Activation("relu"))
		model.add(BatchNormalization(axis=chanDim))
		model.add(MaxPooling2D(pool_size=(2, 2)))

		model.add(SeparableConv2D(32, (3, 3), padding="same"))
		model.add(Activation("relu"))
		model.add(BatchNormalization(axis=chanDim))
		model.add(MaxPooling2D(pool_size=(2, 2)))

		model.add(SeparableConv2D(64, (3, 3), padding="same"))
		model.add(Activation("relu"))
		model.add(BatchNormalization(axis=chanDim))
		model.add(SeparableConv2D(64, (3, 3), padding="same"))
		model.add(Activation("relu"))
		model.add(BatchNormalization(axis=chanDim))
		model.add(MaxPooling2D(pool_size=(2, 2)))

		model.add(Flatten())
		model.add(Dense(128))
		model.add(Activation("relu"))
		model.add(BatchNormalization())
		model.add(Dropout(0.5))

		model.add(Dense(128))
		model.add(Activation("relu"))
		model.add(BatchNormalization())
		model.add(Dropout(0.5))

		model.add(Dense(classes))
		model.add(Activation("softmax"))

		return model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
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
