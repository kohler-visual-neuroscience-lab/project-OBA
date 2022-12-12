"""
The aim here is to create several versions of an input image with different
orientations

Mohammad Shams
m.shams.ahmar@gmail.com
last modification: 2022-11-2
"""

from PIL import Image, ImageEnhance
import numpy as np

# read the image
im = Image.open("house2.png")

factors = np.arange(0, 2, .2)

for count, factor in enumerate(factors):
    im_output = im.rotate(factor)
    im_output.save(f'house_orient{count}.png')
