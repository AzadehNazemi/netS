import os
import sys,cv2
from sklearn.model_selection import train_test_split    

from imutils import paths
IMG_SHAPE = (28, 28, 1)

BATCH_SIZE = 64
EPOCHS = 100

BASE_OUTPUT = "output"

MODEL_PATH = os.path.sep.join([BASE_OUTPUT, "siamese_model"])
PLOT_PATH = os.path.sep.join([BASE_OUTPUT, "plot.png"])
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.layers import MaxPooling2D

def build_siamese_model(inputShape, embeddingDim=48):
	inputs = Input(inputShape)

	x = Conv2D(64, (2, 2), padding="same", activation="relu")(inputs)
	x = MaxPooling2D(pool_size=(2, 2))(x)
	x = Dropout(0.3)(x)

	x = Conv2D(64, (2, 2), padding="same", activation="relu")(x)
	x = MaxPooling2D(pool_size=2)(x)
	x = Dropout(0.3)(x)

	pooledOutput = GlobalAveragePooling2D()(x)
	outputs = Dense(embeddingDim)(pooledOutput)

	model = Model(inputs, outputs)

	return model
import tensorflow.keras.backend as K
import matplotlib.pyplot as plt
import numpy as np

def make_pairs(images, labels):
    pairImages = []
    pairLabels = []

    numClasses = len(np.unique(labels))
    idx = [np.where(labels == i)[0] for i in range(0, numClasses)]
    print(idx)
    for idxA in range(len(images)):
        currentImage = images[idxA]
        label = labels[idxA]
        
        for idxB in range(len(images)):
            if labels[idxB]==label and idxA!=idxB:
        
                posImage = images[idxB]

                pairImages.append([currentImage, posImage])
                pairLabels.append([1])
            if labels[idxB]!=label and idxA!=idxB:

        
                negImage = images[idxB]
                                 
                pairImages.append([currentImage, negImage])
                pairLabels.append([0])

    return (np.array(pairImages), np.array(pairLabels))

def euclidean_distance(vectors):
	(featsA, featsB) = vectors

	sumSquared = K.sum(K.square(featsA - featsB), axis=1,
		keepdims=True)

	return K.sqrt(K.maximum(sumSquared, K.epsilon()))

def plot_training(H, plotPath):
	plt.style.use("ggplot")
	plt.figure()
	plt.plot(H.history["loss"], label="train_loss")
	plt.plot(H.history["val_loss"], label="val_loss")
	plt.plot(H.history["accuracy"], label="train_acc")
	plt.plot(H.history["val_accuracy"], label="val_acc")
	plt.title("Training Loss and Accuracy")
	plt.ylabel("Loss/Accuracy")
	plt.legend(loc="lower left")
	plt.savefig(plotPath)

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Lambda
from tensorflow.keras.datasets import mnist
import numpy as np

def load_dataset(datasetPath,w,h):
    imagePaths = list(paths.list_images(datasetPath))
    data = []
    label=[]
    for imagePath in imagePaths:
        image = cv2.imread(imagePath,0)         
        image=cv2.resize(image,(w,h))
        k=imagePath.split("/")[-1].split("_")[0]
        k=int(k)
        data.append(image)
        label.append(k)
    return np.array(data),label       
        


W=28
H=28
data,label=load_dataset(sys.argv[1],W,H)
print("[INFO] loading  dataset...")
(trainX,testX,trainY,testY) = train_test_split(data,label, test_size=0.25, random_state=42)


print("[INFO] loading MNIST dataset...")
trainX = trainX / 255.0
testX = testX / 255.0

trainX = np.expand_dims(trainX, axis=-1)
testX = np.expand_dims(testX, axis=-1)

print("[INFO] preparing positive and negative pairs...")
(pairTrain, labelTrain) = make_pairs(trainX, trainY)
(pairTest, labelTest) = make_pairs(testX, testY)

print("[INFO] building siamese network...")
imgA = Input(shape=IMG_SHAPE)
imgB = Input(shape=IMG_SHAPE)
featureExtractor = build_siamese_model(IMG_SHAPE)
featsA = featureExtractor(imgA)
featsB = featureExtractor(imgB)

distance = Lambda(euclidean_distance)([featsA, featsB])
outputs = Dense(1, activation="sigmoid")(distance)
model = Model(inputs=[imgA, imgB], outputs=outputs)

print("[INFO] compiling model...")
model.compile(loss="binary_crossentropy", optimizer="adam",
	metrics=["accuracy"])
'''
print("[INFO] training model...")
history = model.fit(
	[pairTrain[:, 0], pairTrain[:, 1]], labelTrain[:],
	validation_data=([pairTest[:, 0], pairTest[:, 1]], labelTest[:]),
	batch_size=BATCH_SIZE, 
	epochs=EPOCHS)

print("[INFO] saving siamese model...")
model.save(MODEL_PATH)

print("[INFO] plotting training history...")
plot_training(history, PLOT_PATH)

'''
from tensorflow.keras.models import load_model
from imutils.paths import list_images
import matplotlib.pyplot as plt
import numpy as np
import sys
import cv2
w,h=28,28
testImagePaths = list(list_images(sys.argv[2]))
np.random.seed(42)
pairs = np.random.choice(testImagePaths, size=(10, 2))
print("[INFO] loading siamese model...")
model = load_model(MODEL_PATH)
for (i, (pathA, pathB)) in enumerate(pairs):
    imageA = cv2.imread(pathA, 0)
    imageB = cv2.imread(pathB, 0)
    imageA=cv2.resize(imageA,(w,h))
    imageB=cv2.resize(imageB,(w,h))
        
    origA = imageA.copy()
    origB = imageB.copy()
    imageA = np.expand_dims(imageA, axis=-1)
    imageB = np.expand_dims(imageB, axis=-1)
    imageA = np.expand_dims(imageA, axis=0)
    imageB = np.expand_dims(imageB, axis=0)
    imageA = imageA / 255.0
    imageB = imageB / 255.0
    preds = model.predict([imageA, imageB])
    proba = preds[0][0]
	fig = plt.figure("Pair #{}".format(i + 1), figsize=(4, 2))
	
    plt.suptitle("Similarity: {:.2f}".format(proba))
    ax = fig.add_subplot(1, 2, 1)
    plt.imshow(origA, cmap=plt.cm.gray)
    plt.axis("off")
    ax = fig.add_subplot(1, 2, 2)
    plt.imshow(origB, cmap=plt.cm.gray)
    plt.axis("off")
    plt.show()
