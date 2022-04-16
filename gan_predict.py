import tensorflow as tf
#from tf import *
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from numpy import load
from numpy import expand_dims
import numpy as np
import cv2
import os
import sys
from os import listdir
from PIL import Image


#import matplotlib.pyplot as plt
# if os.path.exists('predict_pix2pix_gan') == False:
#     os.mkdir("predict_pix2pix_gan")
for root, dirs, files in os.walk(sys.argv[1]):
    for filename in files:
        exten = filename[filename.rfind("."):]
        fn = os.path.join(root, filename)
        if filename.count("_") > 3:


            drill = filename.split('_')[1]
            print(drill)
            if os.path.exists(drill) == False:
                os.mkdir(drill)
                break




model = load_model("model_013900.h5")




def load_image(filename, size=(256, 256)):
    # load image with the preferred size
    pixels = load_img(filename, target_size=size)
    # convert to numpy array
    pixels = img_to_array(pixels)
    # scale from [0,255] to [-1,1]
    pixels = (pixels - 127.5) / 127.5
    # reshape to 1 sample
    pixels = expand_dims(pixels, 0)
    return pixels
def connect(fn, filename):
        src_image = load_image(fn)
        imgcv = cv2.imread(fn)
        ho, wo = imgcv.shape[:2]
        print('Loaded', src_image.shape)
        gen_image = model.predict(src_image)
        gen_image = (gen_image + 1) / 2.0
        img = gen_image[0]
        img = cv2.convertScaleAbs(img, alpha=(255.0))


        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


        img = cv2.resize(gray, (wo, ho))
        thresholdValue = np.mean(img)
        xDim, yDim = img.shape[:2]
        mask = np.zeros(img.shape[:2], dtype="uint8")


        for i in range(xDim):
            for j in range(yDim):
                if (img[i][j] > thresholdValue):
                    mask[i][j] = 255
                else:
                    mask[i][j] = 0
        print(thresholdValue)
        cv2.imwrite(filename, mask)




for root, dirs, files in os.walk(sys.argv[1]):
    for filename in files:
        exten = filename[filename.rfind("."):]
        fn = os.path.join(root, filename)
        connect(fn, filename)

