
'''
jp2 TO JPG |png  or  jpg
'''
from PIL import Image
import cv2
import os
import sys 
for root, dirs, files in os.walk(sys.argv[1]):
    for filename in files:
        ext = filename[filename.rfind("."):].lower()
        if ext == ".jp2":
            fn = os.path.join(root, filename)
            print(sys.argv[1])
            imagePath = fn
            img = Image.open(imagePath)
            outjpg = filename.replace("jp2", "jpg")
            img.save(outjpg)
