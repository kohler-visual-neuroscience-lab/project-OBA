"""
The aim here is to create several versions of an input image with different contrast levels

Mohammad Shams
m.shams.ahmar@gmail.com
last modification: 2022-10-31
"""

from PIL import Image, ImageEnhance
import numpy as np

# read the image
im = Image.open("house2.png")

# image brightness enhancer
enhancer = ImageEnhance.Contrast(im)

factors = np.arange(.7, 1.3, .05)

for count, factor in enumerate(factors):
    im_output = enhancer.enhance(factor)
    im_output.save(f'house_C{count}.png')
