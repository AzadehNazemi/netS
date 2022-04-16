from os import listdir
from numpy import asarray
from numpy import vstack
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
from numpy import savez_compressed
import sys
import os
import cv2


def load_images_src_tar():
    target_size = (256, 512)
    src_list, tar_list = list(), list()
    sn = []
    tn = []
    for filename in listdir("train/image4"):
        sn.append("train/image4/"+filename)
    for filename in listdir("train/label4"):
        tn.append("train/label4/"+filename)

    for i in range(len(tn)):

        t = load_img(tn[i], target_size=(256, 256), interpolation="nearest")
        s = load_img(sn[i],
                     target_size=(256, 256), interpolation="nearest")
        pixels1 = img_to_array(s)
        pixels2 = img_to_array(t)
        src_list.append(pixels1)
        tar_list.append(pixels2)

    return [asarray(src_list), asarray(tar_list)]


[src_images, tar_images] = load_images_src_tar()
print('Loaded: ', src_images.shape, tar_images.shape)
filename = 'dental_256.npz'
savez_compressed(filename, src_images, tar_images)
print('Saved dataset: ', filename)


