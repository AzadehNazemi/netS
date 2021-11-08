
'''
vnirawir to png
'''
from PIL import Image
import cv2
import sys
import os
import envi
from spectral import *


def convert(fl):
    sf = ("%.2f" % fl)
    return sf


for root, dirs, files in os.walk(sys.argv[1]):
    for filename in files:
        exten = filename[filename.rfind("."):]
        fn = os.path.join(root, filename)
        print(exten)
        if fn.count("vnirswir") > 0 and (exten == ".img" or exten == '.IMG'):

            hfn = fn.replace('img', 'hdr')
            with open(hfn, "a") as header:
                header.write("\n")
                header.write("byte order=0")
                header.write("\n")
            header.close()
            outpng= filename.replace('img', 'png')
            img = envi.open(hfn, fn)
            save_rgb(outpng, img)
