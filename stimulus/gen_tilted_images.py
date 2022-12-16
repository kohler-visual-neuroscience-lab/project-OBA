"""
The aim here is to create several versions of an input image with different
orientations
Mohammad Shams <m.shams.ahmar@gmail.com>
last modification: 2022-12-07
"""

from PIL import Image
import numpy as np
import os

# face or house
cats = ['face', 'house']

for cat in cats:
    # set the source image path
    source_path = os.path.join("images", "source", f"{cat}.png")
    # set the save image path
    save_path = os.path.join("images")
    # read the image
    im = Image.open(source_path)
    # define the span of tilts (degrees)
    min_tilt = 0
    max_tilt = 5
    step_tilt = 0.1
    factors = np.arange(min_tilt, max_tilt, step_tilt)
    for count, factor in enumerate(factors):
        im_output = im.rotate(factor)
        if count == 0:
            im_output.save(os.path.join(save_path, f'{cat}_tilt{count}.png'))
        else:
            im_output.save(os.path.join(save_path, f'{cat}_tilt'
                                                   f'{count}_CCW.png'))
    min_tilt = 0
    max_tilt = -5
    step_tilt = -0.1
    factors = np.arange(min_tilt, max_tilt, step_tilt)
    for count, factor in enumerate(factors):
        im_output = im.rotate(factor)
        if count == 0:
            im_output.save(os.path.join(save_path, f'{cat}_tilt{count}.png'))
        else:
            im_output.save(os.path.join(save_path, f'{cat}_tilt'
                                                   f'{count}_CW.png'))
