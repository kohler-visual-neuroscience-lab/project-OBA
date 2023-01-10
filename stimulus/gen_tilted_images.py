"""
The aim here is to create several versions of an input image with different
orientations

Mo Shams <MShamsCBR@gmail.com>
last modification: 2023-01-10
"""

import os
import numpy as np
from PIL import Image

# face or house
cats = ['face', 'house']

for cat in cats:
    # set the source image path
    source_path = os.path.join("image", "source", f"{cat}_cropped.png")
    # set the save image path
    save_path = os.path.join("image")
    # read the image
    im = Image.open(source_path)
    # define the span of tilts (degrees)
    min_tilt = 0
    max_tilt = 5
    step_tilt = 0.1
    mags = np.arange(min_tilt, max_tilt, step_tilt)
    for imag, mag in enumerate(mags):
        im_output = im.rotate(mag)
        if imag == 0:
            im_output.save(os.path.join(save_path,
                                        f'{cat}_tilt{imag}.png'))
        else:
            im_output.save(os.path.join(save_path,
                                        f'{cat}_tilt{imag}_CCW.png'))
    min_tilt = 0
    max_tilt = -5
    step_tilt = -0.1
    mags = np.arange(min_tilt, max_tilt, step_tilt)
    for imag, mag in enumerate(mags):
        im_output = im.rotate(mag)
        if imag == 0:
            im_output.save(os.path.join(save_path, f'{cat}_tilt{imag}.png'))
        else:
            im_output.save(os.path.join(save_path, f'{cat}_tilt'
                                                   f'{imag}_CW.png'))
