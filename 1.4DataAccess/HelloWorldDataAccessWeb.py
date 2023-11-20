import six.moves.urllib.request as urllib
import os
import PIL.Image as PImage
import matplotlib.pyplot as plt

folder = "../Files"
if not os.path.isfile(os.path.join(folder, "STOP_sign.jpg")):
    urllib.urlretrieve("https://upload.wikimedia.org/wikipedia/commons/f/f9/STOP_sign.jpg",
                       os.path.join(folder, "STOP_sign.jpg"))
image = PImage.open(os.path.join(folder, "STOP_sign.jpg"))
plt.imshow(image, cmap=plt.cm.gray)
plt.show()

