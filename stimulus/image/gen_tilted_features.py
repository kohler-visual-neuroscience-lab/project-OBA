"""
The aim here is to create several versions of an input image with different
orientations

    Mo Shams <MShamsCBR@gmail.com>
    May 07, 2023
"""

import os
import numpy as np
from PIL import Image

# set the save image path
save_path = os.path.join("", "image_set_exp01")

# face or house categories
cats = ['red', 'blue']
# examples
exms = [1]

for cat in cats:
    for exm in exms:
        # set the source image path
        source_path = os.path.join("", "source", "exp01",
                                   f"{cat}{exm}.png")
        # read the image
        im = Image.open(source_path)
        # define the span of tilts (degrees)
        min_tilt = 0
        max_tilt = 10
        step_tilt = 0.1
        mags = np.arange(min_tilt, max_tilt, step_tilt)
        for imag, mag in enumerate(mags):
            im_output = im.rotate(mag)
            if imag == 0:
                im_output.save(os.path.join(save_path,
                                            f'{cat}{exm}_tilt{imag}.png'))
            else:
                im_output.save(os.path.join(save_path,
                                            f'{cat}{exm}_tilt{imag}_CCW.png'))
        min_tilt = 0
        max_tilt = -10
        step_tilt = -0.1
        mags = np.arange(min_tilt, max_tilt, step_tilt)
        for imag, mag in enumerate(mags):
            im_output = im.rotate(mag)
            if imag == 0:
                im_output.save(os.path.join(save_path, f'{cat}{exm}_tilt'
                                                       f'{imag}.png'))
            else:
                im_output.save(os.path.join(save_path, f'{cat}{exm}_tilt'
                                                       f'{imag}_CW.png'))
