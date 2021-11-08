
'''
Convert bin plus header hdr file to png
bin|raw to png
'''
import cv2
import sys
import os
import envi


if os.path.exists("pngs") == False:
    os.mkdir("pngs")
fileMode = "png"

for root, dirs, files in os.walk(sys.argv[1]):
    for filename in files:
        exten = filename[filename.rfind("."):]
        fn = os.path.join(root, filename)
        outpng= "pngs\\" + filename.replace('bin', 'png')
        if (exten == ".BIN" or exten == '.bin'):
            if os.path.exists(outpng) == False:
                hfn = fn.replace('bin', 'hdr')
                if os.path.exists(hfn):
                    print(hfn, fn)
                    x = open(hfn)
                    headerfile = x.readlines()
                    if headerfile is not None:
                        with open(hfn, "a") as header:
                            if header != None:
                                header.write("\n")
                                header.write("byte order=0")
                                header.write("\n")
                                header.close()
                                img = envi.open(hfn, fn) 
                                save_rgb(outpng ,img)
                           
