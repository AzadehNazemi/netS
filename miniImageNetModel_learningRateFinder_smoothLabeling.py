from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import concatenate
from tensorflow.keras import backend as K
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.datasets import cifar10
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
matplotlib.use("Agg")
def smooth_labels(labels, factor=0.1):
	labels *= (1 - factor)
	labels += (factor / labels.shape[1])

	return labels



class MiniGoogLeNet:
	@staticmethod
	def conv_module(x, K, kX, kY, stride, chanDim, padding="same"):
		x = Conv2D(K, (kX, kY), strides=stride, padding=padding)(x)
		x = BatchNormalization(axis=chanDim)(x)
		x = Activation("relu")(x)

		return x

	@staticmethod
	def inception_module(x, numK1x1, numK3x3, chanDim):
		conv_1x1 = MiniGoogLeNet.conv_module(x, numK1x1, 1, 1,
			(1, 1), chanDim)
		conv_3x3 = MiniGoogLeNet.conv_module(x, numK3x3, 3, 3,
			(1, 1), chanDim)
		x = concatenate([conv_1x1, conv_3x3], axis=chanDim)

		return x

	@staticmethod
	def downsample_module(x, K, chanDim):
		conv_3x3 = MiniGoogLeNet.conv_module(x, K, 3, 3, (2, 2),
			chanDim, padding="valid")
		pool = MaxPooling2D((3, 3), strides=(2, 2))(x)
		x = concatenate([conv_3x3, pool], axis=chanDim)

		return x

	@staticmethod
	def build(width, height, depth, classes):
		inputShape = (height, width, depth)
		chanDim = -1

		if K.image_data_format() == "channels_first":
			inputShape = (depth, height, width)
			chanDim = 1

		inputs = Input(shape=inputShape)
		x = MiniGoogLeNet.conv_module(inputs, 96, 3, 3, (1, 1),
			chanDim)

		x = MiniGoogLeNet.inception_module(x, 32, 32, chanDim)
		x = MiniGoogLeNet.inception_module(x, 32, 48, chanDim)
		x = MiniGoogLeNet.downsample_module(x, 80, chanDim)

		x = MiniGoogLeNet.inception_module(x, 112, 48, chanDim)
		x = MiniGoogLeNet.inception_module(x, 96, 64, chanDim)
		x = MiniGoogLeNet.inception_module(x, 80, 80, chanDim)
		x = MiniGoogLeNet.inception_module(x, 48, 96, chanDim)
		x = MiniGoogLeNet.downsample_module(x, 96, chanDim)

		x = MiniGoogLeNet.inception_module(x, 176, 160, chanDim)
		x = MiniGoogLeNet.inception_module(x, 176, 160, chanDim)
		x = AveragePooling2D((7, 7))(x)
		x = Dropout(0.5)(x)

		x = Flatten()(x)
		x = Dense(classes)(x)
		x = Activation("softmax")(x)

		model = Model(inputs, x, name="googlenet")

		return model
        
        
        
        
        
        
        

class LearningRateDecay:
	def plot(self, epochs, title="Learning Rate Schedule"):
		lrs = [self(i) for i in epochs]

		plt.style.use("ggplot")
		plt.figure()
		plt.plot(epochs, lrs)
		plt.title(title)
		plt.ylabel("Learning Rate")

class StepDecay(LearningRateDecay):
	def __init__(self, initAlpha=0.01, factor=0.25, dropEvery=10):
		self.initAlpha = initAlpha
		self.factor = factor
		self.dropEvery = dropEvery

	def __call__(self, epoch):
		exp = np.floor((1 + epoch) / self.dropEvery)
		alpha = self.initAlpha * (self.factor ** exp)

		return float(alpha)

class PolynomialDecay(LearningRateDecay):
	def __init__(self, maxEpochs=100, initAlpha=0.01, power=1.0):
		self.maxEpochs = maxEpochs
		self.initAlpha = initAlpha
		self.power = power

	def __call__(self, epoch):
		decay = (1 - (epoch / float(self.maxEpochs))) ** self.power
		alpha = self.initAlpha * decay

		return float(alpha)





NUM_EPOCHS = 1      
INIT_LR = 5e-3
BATCH_SIZE = 64
smoothing=0.1


labelNames = ["airplane", "automobile", "bird", "cat\\", "deer", "dog",
	"frog", "horse", "ship", "truck"]

print("[INFO] loading CIFAR-10 data...")
((trainX, trainY), (testX, testY)) = cifar10.load_data()
trainX = trainX.astype("float")
testX = testX.astype("float")

mean = np.mean(trainX, axis=0)
trainX -= mean
testX -= mean

lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.transform(testY)
trainY = trainY.astype("float")
testY = testY.astype("float")
smoothing=0.1
print("[INFO] before smoothing: {}".format(trainY[0]))
trainY = smooth_labels(trainY,smoothing)
print("[INFO] after smoothing: {}".format(trainY[0]))

aug = ImageDataGenerator(
	width_shift_range=0.1,
	height_shift_range=0.1,
	horizontal_flip=True,
	fill_mode="nearest")

schedule = PolynomialDecay(maxEpochs=NUM_EPOCHS, initAlpha=INIT_LR,
	power=1.0)
callbacks = [LearningRateScheduler(schedule)]

print("[INFO] compiling model...")
opt = SGD(lr=INIT_LR, momentum=0.9)
model = MiniGoogLeNet.build(width=32, height=32, depth=3, classes=10)
model.compile(loss="categorical_crossentropy", optimizer=opt,
	metrics=["accuracy"])

print("[INFO] training network...")
H = model.fit_generator(
	aug.flow(trainX, trainY, batch_size=BATCH_SIZE),
	validation_data=(testX, testY),
	steps_per_epoch=len(trainX) // BATCH_SIZE,
	epochs=NUM_EPOCHS,
	callbacks=callbacks,
	verbose=1)

print("[INFO] evaluating network...")
predictions = model.predict(testX, batch_size=BATCH_SIZE)
print(classification_report(testY.argmax(axis=1),
predictions.argmax(axis=1), target_names=labelNames))

N = np.arange(0, NUM_EPOCHS)
plt.style.use("ggplot")
plt.figure()
plt.plot(N, H.history["loss"], label="train_loss")
plt.plot(N, H.history["val_loss"], label="val_loss")
plt.plot(N, H.history["accuracy"], label="train_acc")
plt.plot(N, H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig("plot.png")
sd = StepDecay(initAlpha=0.01, factor=0.5, dropEvery=25)
sd.plot(range(0, 100), title="Step-based Decay")
plt.show()

pd = PolynomialDecay(power=1)
pd.plot(range(0, 100), title="Linear Decay (p=1)")
plt.show()

pd = PolynomialDecay(power=5)
pd.plot(range(0, 100), title="Polynomial Decay (p=5)")
plt.show()




    
