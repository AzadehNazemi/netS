from imutils.paths import list_images
from sklearn.model_selection import train_test_split    
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Reshape
from tensorflow.keras.layers import Input
from tensorflow.keras import backend as K
import imutils
import sys,os,cv2
from imutils import build_montages
import numpy as np
import pickle
from imutils import paths 

class ConvAutoencoder:
	@staticmethod   
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
		latent = Dense(latentDim, name="encoded")(x)

		x = Dense(np.prod(volumeSize[1:]))(latent)
		x = Reshape((volumeSize[1], volumeSize[2], volumeSize[3]))(x)

		for f in filters[::-1]:
			x = Conv2DTranspose(f, (3, 3), strides=2,
				padding="same")(x)
			x = LeakyReLU(alpha=0.2)(x)
			x = BatchNormalization(axis=chanDim)(x)

		x = Conv2DTranspose(depth, (3, 3), padding="same")(x)
		outputs = Activation("sigmoid", name="decoded")(x)

		autoencoder = Model(inputs, outputs, name="autoencoder")

		return autoencoder
        
        

def euclidean(a, b):
	return np.linalg.norm(a - b)

def perform_search(queryFeatures, index, maxResults=64):
	results = []

	for i in range(0, len(index["features"])):
		d = euclidean(queryFeatures, index["features"][i])
		results.append((d, i))

	results = sorted(results)[:maxResults]

	return results
     

def load_dataset(datasetPath,w,h):
    imagePaths = list(paths.list_images(datasetPath))
    data = []
    for imagePath in imagePaths:
        image = cv2.imread(imagePath)
        image=cv2.resize(image,(w,h))
        data.append(image)

    return np.array(data)
   
def visualize_predictions(decoded, gt, samples=10):
	outputs = None

	for i in range(0, samples):
		original = (gt[i] * 255).astype("uint8")
		output = (decoded[i] * 255).astype("uint8")


		if outputs is None:
			outputs = output

		else:
			outputs = np.vstack([outputs, output])

	return outputs
index = {}
EPOCHS = 20
INIT_LR = 1e-3
BS = 32


data=load_dataset(sys.argv[1],32,64)
print("[INFO] loading ")
(trainX,testX,trainX,testX) = train_test_split(data,data, test_size=0.25, random_state=42)

    

print("[INFO] building autoencoder...")
trainX = np.expand_dims(trainX, axis=-1)
testX = np.expand_dims(testX, axis=-1)
trainX = trainX.astype("float32") / 255.0
testX = testX.astype("float32") / 255.0
 
print("[INFO] building autoencoder...")
autoencoder = ConvAutoencoder.build(32, 64, 3)
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
autoencoder.compile(loss="mse", optimizer=opt)

H = autoencoder.fit(
	trainX, trainX,
	validation_data=(testX, testX),
	epochs=EPOCHS,
	batch_size=BS)

print("[INFO] making predictions...")
decoded = autoencoder.predict(testX)
print("[INFO] saving autoencoder...")
autoencoder.save("autoencoder.h5", save_format="h5")
autoencoder = load_model("autoencoder.h5")

encoder = Model(inputs=autoencoder.input,
	outputs=autoencoder.get_layer("encoded").output)

print("[INFO] encoding images...")
features = encoder.predict(trainX)

indexes = list(range(0, trainX.shape[0]))
data = {"indexes": indexes, "features": features}

print("[INFO] saving index...")
f = open("index.pickle", "wb")
f.write(pickle.dumps(data))
f.close()


print("[INFO] loading autoencoder and index...")
autoencoder = load_model("autoencoder.h5")
index = pickle.loads(open("index.pickle", "rb").read())
encoder = Model(inputs=autoencoder.input,
	outputs=autoencoder.get_layer("encoded").output)
features = encoder.predict(testX)
queryIdxs = list(range(0, testX.shape[0]))
queryIdxs = np.random.choice(queryIdxs, size=(testX.shape[0]),
	replace=False)
for i in queryIdxs:
	queryFeatures = features[i]
	results = perform_search(queryFeatures, index, maxResults=225)
	images = []
	for (d, j) in results:
		image = (trainX[j] * 255).astype("uint8")
		image = np.dstack([image] * 3)
		images.append(image)
	query = (testX[i] * 255).astype("uint8")
print("[INFO] making predictions...")
decoded = autoencoder.predict(testX)
vis = visualize_predictions(decoded, testX)
cv2.imwrite("vis.jpg", vis)
