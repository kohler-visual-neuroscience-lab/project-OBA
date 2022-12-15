"""
The aim here is to create several versions of an input image with different
orientations
Mohammad Shams <m.shams.ahmar@gmail.com>
last modification: 2022-12-07
"""

from PIL import Image
import numpy as np
import os

# set the source image path
source_path = os.path.join("images", "source", "face.png")
# set the save image path
save_path = os.path.join("images")
# read the image
im = Image.open(source_path)
# define the span of tilts (degrees)
ch_min = 0
ch_max = 3
ch_step = 0.1
factors = np.arange(ch_min, ch_max, ch_step)

for count, factor in enumerate(factors):
    im_output = im.rotate(factor)
    im_output.save(os.path.join(save_path, f'face_tilt{count}.png'))
