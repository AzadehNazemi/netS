
# in "INPUT_DATABASE" each class must be locatred @  seperated folders 
# python trainMakeDataMakeModelResNet.py INPUT_DATABASE'
'''

from keras.engine import training
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import LearningRateScheduler
from keras.optimizers import SGD
from keras_preprocessing import image
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import AveragePooling2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.convolutional import ZeroPadding2D
from keras.layers.core import Activation
from keras.layers.core import Dense
from keras.layers import Flatten
from keras.layers import Input
from keras.models import Model
from keras.layers import add
from keras.regularizers import l2
from keras import backend as K
from imutils import paths
import numpy as np
from keras import save
import random
import shutil
import os
import sys


class ResNet:

    def residual_module(data, K, stride, chanDim, red=False,
                        reg=0.0001, bnEps=2e-5, bnMom=0.9):
        shortcut = data

        bn1 = BatchNormalization(axis=chanDim, epsilon=bnEps,
                                 momentum=bnMom)(data)
        act1 = Activation("relu")(bn1)
        conv1 = Conv2D(int(K * 0.25), (1, 1), use_bias=False,
                       kernel_regularizer=l2(reg))(act1)

        bn2 = BatchNormalization(axis=chanDim, epsilon=bnEps,
                                 momentum=bnMom)(conv1)
        act2 = Activation("relu")(bn2)
        conv2 = Conv2D(int(K * 0.25), (3, 3), strides=stride,
                       padding="same", use_bias=False,
                       kernel_regularizer=l2(reg))(act2)

        bn3 = BatchNormalization(axis=chanDim, epsilon=bnEps,
                                 momentum=bnMom)(conv2)
        act3 = Activation("relu")(bn3)
        conv3 = Conv2D(K, (1, 1), use_bias=False,
                       kernel_regularizer=l2(reg))(act3)

        if red:
            shortcut = Conv2D(K, (1, 1), strides=stride,
                              use_bias=False, kernel_regularizer=l2(reg))(act1)

        x = add([conv3, shortcut])

        return x
    def build(width, height, depth, classes, stages, filters,
              reg=0.0001, bnEps=2e-5, bnMom=0.9):
        inputShape = (height, width, depth)
        chanDim = -1

        if K.image_data_format() == "channels_first":
            inputShape = (depth, height, width)
            chanDim = 1

        inputs = Input(shape=inputShape)
        x = BatchNormalization(axis=chanDim, epsilon=bnEps,
                               momentum=bnMom)(inputs)

        x = Conv2D(filters[0], (5, 5), use_bias=False,
                   padding="same", kernel_regularizer=l2(reg))(x)
        x = BatchNormalization(axis=chanDim, epsilon=bnEps,
                               momentum=bnMom)(x)
        x = Activation("relu")(x)
        x = ZeroPadding2D((1, 1))(x)
        x = MaxPooling2D((3, 3), strides=(2, 2))(x)

        for i in range(0, len(stages)):
            stride = (1, 1) if i == 0 else (2, 2)
            x = ResNet.residual_module(x, filters[i + 1], stride,
                                       chanDim, red=True, bnEps=bnEps, bnMom=bnMom)

            for j in range(0, stages[i] - 1):
                x = ResNet.residual_module(x, filters[i + 1],
                                           (1, 1), chanDim, bnEps=bnEps, bnMom=bnMom)

        x = BatchNormalization(axis=chanDim, epsilon=bnEps,
                               momentum=bnMom)(x)
        x = Activation("relu")(x)
        x = AveragePooling2D((8, 8))(x)

        x = Flatten()(x)
        x = Dense(classes, kernel_regularizer=l2(reg))(x)
        x = Activation("softmax")(x)

        model = Model(inputs, x, name="resnet")

        return model


if os.path.exists("training") == False:
    os.makedirs("training")


if os.path.exists("testing") == False:
    os.makedirs("testing")

if os.path.exists("validation") == False:
    os.makedirs("validation")

INPUT_DATASET = sys.argv[1]
imagePaths = list(paths.list_images(INPUT_DATASET))
TRAIN_PATH = "training"
VAL_PATH = "validation"
TEST_PATH = "testing"
TRAIN_SPLIT = 0.8
VAL_SPLIT = 0.1
random.seed(42)
random.shuffle(imagePaths)


for l in range(len(imagePaths)):
    inputPath = imagePaths[l]
    filename = inputPath.split(os.path.sep)[-1]
    labelPath = inputPath.split(os.path.sep)[1]

    if not os.path.exists("training\\"+labelPath):
        os.makedirs("training\\"+labelPath)
    if not os.path.exists("testing\\"+labelPath):
        os.makedirs("testing\\"+labelPath)
    if not os.path.exists("validation\\"+labelPath):
        os.makedirs("validation\\"+labelPath)

    if l < 0.8 * len(imagePaths):
        shutil.copy2(inputPath, "training\\"+labelPath)
    else:
        shutil.copy2(inputPath, "validation\\"+labelPath)
        shutil.copy2(inputPath, "testing\\"+labelPath)


NUM_EPOCHS = 20
INIT_LR = 1e-1
BS = 32


def poly_decay(epoch):
    maxEpochs = NUM_EPOCHS
    baseLR = INIT_LR
    power = 1.0

    alpha = baseLR * (1 - (epoch / float(maxEpochs))) ** power

    return alpha


totalTrain = len(list(paths.list_images(TRAIN_PATH)))
totalVal = len(list(paths.list_images(VAL_PATH)))
totalTest = len(list(paths.list_images(TEST_PATH)))

trainAug = ImageDataGenerator(
    rescale=1 / 255.0,
    rotation_range=20,
    zoom_range=0.05,
    width_shift_range=0.05,
    height_shift_range=0.05,
    shear_range=0.05,
    horizontal_flip=True,
    fill_mode="nearest")

valAug = ImageDataGenerator(rescale=1 / 255.0)

trainGen = trainAug.flow_from_directory(
    TRAIN_PATH,
    class_mode="categorical",
    target_size=(64, 64),
    color_mode="rgb",
    shuffle=True,
    batch_size=BS)

valGen = valAug.flow_from_directory(
    VAL_PATH,
    class_mode="categorical",
    target_size=(64, 64),
    color_mode="rgb",
    shuffle=False,
    batch_size=BS)

testGen = valAug.flow_from_directory(
    TEST_PATH,
    class_mode="categorical",
    target_size=(64, 64),
    color_mode="rgb",
    shuffle=False,
    batch_size=BS)

model = ResNet.build(64, 64, 3, 2, (3, 4, 6),
                     (64, 128, 256, 512), reg=0.0005)
opt = SGD(lr=INIT_LR, momentum=0.9)
model.compile(loss="binary_crossentropy", optimizer=opt,
              metrics=["accuracy"])

callbacks = [LearningRateScheduler(poly_decay)]
H = model.fit_generator(
    trainGen,
    steps_per_epoch=totalTrain // BS,
    validation_data=valGen,
    validation_steps=totalVal // BS,
    epochs=NUM_EPOCHS,
    callbacks=callbacks)
model.save("resnet.h5")
#----------predict------

# testlResNet.py     "sample.jpg"
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import img_to_array
from keras.utils import to_categorical
from keras.models import load_model
import numpy as np
import random
import cv2
import os
import sys


model = load_model("resnet.h5")
testGen = cv2.imread(sys.argv[1])
image = cv2.resize(testGen, (64, 64))
image = image.astype("float") / 255.0
image = img_to_array(image)
image = np.expand_dims(image, axis=0)
model.predict(image)[0]
predication = np.argmax(model.predict(image)[0])
print(predication)
