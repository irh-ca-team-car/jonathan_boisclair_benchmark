from os import listdir
from PIL import Image
for filename in listdir('data/FLIR_CONVERTED/images'):
    if filename.endswith('jpg'):
        print(filename)
        try:
            img = Image.open('data/FLIR_CONVERTED/images/'+filename) # open the image file.
            img.verify() # verify that it is, in fact an image.
        except:
            print(filename,"is corrupted")