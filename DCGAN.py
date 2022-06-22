from PIL import Image
from sklearn.model_selection import train_test_split    
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Reshape
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.optimizers import Adam
from sklearn.utils import shuffle
from imutils import build_montages

from imutils import paths
import numpy as np
import sys
import cv2
import os

class DCGAN:
	@staticmethod
	def build_generator(dim, depth, channels=1, inputDim=100,
		outputDim=512):
		model = Sequential()
		inputShape = (dim, dim, depth)
		chanDim = -1

		model.add(Dense(input_dim=inputDim, units=outputDim))
		model.add(Activation("relu"))
		model.add(BatchNormalization())

		model.add(Dense(dim * dim * depth))
		model.add(Activation("relu"))
		model.add(BatchNormalization())

		model.add(Reshape(inputShape))
		model.add(Conv2DTranspose(32, (5, 5), strides=(2, 2),
			padding="same"))
		model.add(Activation("relu"))
		model.add(BatchNormalization(axis=chanDim))

		model.add(Conv2DTranspose(channels, (5, 5), strides=(2, 2),
			padding="same"))
		model.add(Activation("tanh"))

		return model

	@staticmethod
	def build_discriminator(width, height, depth, alpha=0.2):
		model = Sequential()
		inputShape = (height, width, depth)

		model.add(Conv2D(32, (5, 5), padding="same", strides=(2, 2),
			input_shape=inputShape))
		model.add(LeakyReLU(alpha=alpha))

		model.add(Conv2D(64, (5, 5), padding="same", strides=(2, 2)))
		model.add(LeakyReLU(alpha=alpha))

		model.add(Flatten())
		model.add(Dense(512))
		model.add(LeakyReLU(alpha=alpha))

		model.add(Dense(1))
		model.add(Activation("sigmoid"))

		return model
 
def load_dataset(datasetPath,w,h):
    imagePaths = list(paths.list_images(datasetPath))
    data = []
    for imagePath in imagePaths:
        image = cv2.imread(imagePath,0)         
        image=cv2.resize(image,(w,h))
        data.append(image)

    return np.array(data)       
        


NUM_EPOCHS = 50
BATCH_SIZE =28
INIT_LR = 2e-4
W=28
H=28
data=load_dataset(sys.argv[1],W,H)
print("[INFO] loading  dataset...")
(trainX,testX,trainX,testX) = train_test_split(data,data, test_size=0.25, random_state=42)
trainImages = np.concatenate([trainX, testX])

trainImages = np.expand_dims(trainImages, axis=-1)
trainImages = (trainImages.astype("float") - 127.5) / 127.5

print("[INFO] building generator...")
gen = DCGAN.build_generator(7, 64, channels=1
)

print("[INFO] building discriminator...")
disc = DCGAN.build_discriminator(28,28,1)
discOpt = Adam(lr=INIT_LR, beta_1=0.5, decay=INIT_LR / NUM_EPOCHS)
disc.compile(loss="binary_crossentropy", optimizer=discOpt)

print("[INFO] building GAN...")
disc.trainable = False
ganInput = Input(shape=(100,))
ganOutput = disc(gen(ganInput))
gan = Model(ganInput, ganOutput)

ganOpt = Adam(lr=INIT_LR, beta_1=0.5, decay=INIT_LR / NUM_EPOCHS)
gan.compile(loss="binary_crossentropy", optimizer=discOpt)

print("[INFO] starting training...")
benchmarkNoise = np.random.uniform(-1, 1, size=(256, 100))

for epoch in range(0, NUM_EPOCHS):
	print("[INFO] starting epoch {} of {}...".format(epoch + 1,
		NUM_EPOCHS))
	batchesPerEpoch = int(trainImages.shape[0] / BATCH_SIZE)

	for i in range(0, batchesPerEpoch):
		p = None

		imageBatch = trainImages[i * BATCH_SIZE:(i + 1) * BATCH_SIZE]
		noise = np.random.uniform(-1, 1, size=(BATCH_SIZE, 100))

		genImages = gen.predict(noise, verbose=0)

		X = np.concatenate((imageBatch, genImages))
		y = ([1] * BATCH_SIZE) + ([0] * BATCH_SIZE)
		y = np.reshape(y, (-1,))
		(X, y) = shuffle(X, y)

		discLoss = disc.train_on_batch(X, y)

		noise = np.random.uniform(-1, 1, (BATCH_SIZE, 100))
		fakeLabels = [1] * BATCH_SIZE
		fakeLabels = np.reshape(fakeLabels, (-1,))
		ganLoss = gan.train_on_batch(noise, fakeLabels)

		if i == batchesPerEpoch - 1:
			p = [   "output", "epoch_{}_output.png".format(
				str(epoch + 1).zfill(4))]

		else:
			if epoch < 10 and i % 25 == 0:
				p = ["output", "epoch_{}_step_{}.png".format(
					str(epoch + 1).zfill(4), str(i).zfill(5))]

			elif epoch >= 10 and i % 100 == 0:
				p =[ "output", "epoch_{}_step_{}.png".format(
					str(epoch + 1).zfill(4), str(i).zfill(5))]
		if p is not None:
			print("[INFO] Step {}_{}: discriminator_loss={:.6f}, "
				"adversarial_loss={:.6f}".format(epoch + 1, i,
					discLoss, ganLoss))

			images = gen.predict(benchmarkNoise)
			images = ((images * 127.5) + 127.5).astype("uint8")
			images = np.repeat(images, 3, axis=-1)
			vis = build_montages(images, (28, 28), (16, 16))[0]

			p = os.path.sep.join(p)
			cv2.imwrite(p, vis)
