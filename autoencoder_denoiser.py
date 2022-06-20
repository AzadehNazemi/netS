from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Reshape
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from sklearn.model_selection import train_test_split    
import numpy as np
import sys,cv2,pickle
from imutils import paths 
class ConvAutoencoder:
	def build(width, height, depth, filters=(32, 64), latentDim=16):
		inputShape = (height, width, depth)
		chanDim = -1

		inputs = Input(shape=inputShape)
		x = inputs

		for f in filters:
			x = Conv2D(f, (3, 3), strides=2, padding="same")(x)
			x = LeakyReLU(alpha=0.2)(x)
			x = BatchNormalization(axis=chanDim)(x)

		volumeSize = K.int_shape(x)
		x = Flatten()(x)
		latent = Dense(latentDim)(x)

		encoder = Model(inputs, latent, name="encoder")

		latentInputs = Input(shape=(latentDim,))
		x = Dense(np.prod(volumeSize[1:]))(latentInputs)
		x = Reshape((volumeSize[1], volumeSize[2], volumeSize[3]))(x)

		for f in filters[::-1]:
			x = Conv2DTranspose(f, (3, 3), strides=2,
				padding="same")(x)
			x = LeakyReLU(alpha=0.2)(x)
			x = BatchNormalization(axis=chanDim)(x)

		x = Conv2DTranspose(depth, (3, 3), padding="same")(x)
		outputs = Activation("sigmoid")(x)

		decoder = Model(latentInputs, outputs, name="decoder")

		autoencoder = Model(inputs, decoder(encoder(inputs)),
			name="autoencoder")

		return (encoder, decoder, autoencoder)

import matplotlib
matplotlib.use("Agg")

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt
import numpy as np
import argparse
import cv2

def load_dataset(datasetPath):
    imagePaths = list(paths.list_images(datasetPath))
    data = []
    label=[]
    for imagePath in imagePaths:
        image = cv2.imread(imagePath,0)
        image=cv2.resize(image,(28,28))
        labelNoise = np.random.normal(loc=0.5, scale=0.5, size=image.shape)
        labelNoisy = np.clip(image+labelNoise, 0, 1)
        label.append(labelNoisy)
        data.append(image)

    return np.array(data),np.array(label)
EPOCHS = 25
BS = 32
data,label=load_dataset(sys.argv[1])
print("[INFO] loading MNIST dataset...")
(trainX,testX,trainXNoisy,testXNoisy) = train_test_split(data,label, test_size=0.1, random_state=42)



print("[INFO] building autoencoder...")
(encoder, decoder, autoencoder) = ConvAutoencoder.build(28, 28, 1)
opt = Adam(lr=1e-3)
autoencoder.compile(loss="mse", optimizer=opt)
print(len(trainX),len(trainXNoisy))
H = autoencoder.fit(
	trainXNoisy, trainX,
	validation_data=(testXNoisy, testX),
	epochs=EPOCHS,
	batch_size=BS)
N = np.arange( 0, EPOCHS)



decoded = autoencoder.predict(testXNoisy)
outputs = None

for i in range(0, len(testXNoisy)):
	original = (testXNoisy[i] * 255).astype("uint8")
	recon = (decoded[i] * 255).astype("uint8")

	output = np.hstack([recon])

	if outputs is None:
		outputs = output

	else:
		outputs = np.vstack([outputs, output])

cv2.imwrite('out.jpg', outputs)
