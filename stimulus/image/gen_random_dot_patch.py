"""
This generates two transparent patches of random dots w/ these parameters:
- color A
- color B
- number of dots per color
- dot radius
- patch inner radius
- patch outer radius

Mo Shams
m.shams.ahmar@gmail.com
Oct 2024
"""

import numpy as np
import matplotlib.pyplot as plt

# Parameters
patch_inner_radius = 100  # pixels
patch_outer_radius = 300  # pixels
dot_radius = 3  # pixels
ndots_perColor = 300
# colors = [[0, 60, 255, 255], [170, 0, 0, 255]]
colors = [[0, 153, 255, 255], [255, 50, 50, 255]]

image_size = 2 * patch_outer_radius + 100  # pixels

for i_image in [0, 1]:

    # Generate random dot coordinates within circular patch
    dot_locs = []
    while len(dot_locs) < ndots_perColor:
        dotX = np.random.uniform(-patch_outer_radius, patch_outer_radius)
        dotY = np.random.uniform(-patch_outer_radius, patch_outer_radius)
        dotR = np.sqrt(dotX ** 2 + dotY ** 2)
        if patch_inner_radius <= dotR <= patch_outer_radius:
            dot_locs.append([dotX, dotY])

    # Create an array representing the image
    image = np.zeros((image_size, image_size, 4), dtype=np.uint8)  # RGBA
    # Draw dots on the image
    for (x, y) in dot_locs:
        dotX_mat, dotY_mat = int(x + image_size / 2), int(y + image_size / 2)
        # Draw a circle for each dot
        for i in range(dotY_mat - dot_radius, dotY_mat + dot_radius):
            for j in range(dotX_mat - dot_radius, dotX_mat + dot_radius):
                current_R2 = (i - dotY_mat) ** 2 + (j - dotX_mat) ** 2
                if current_R2 <= (dot_radius ** 2):
                    image[i, j, :] = colors[i_image]

    # Save the image as a transparent PNG file using matplotlib
    plt.imsave(f"random_dots/patch{i_image + 1}.png",
               image, format="png", cmap="gray",
               vmin=0, vmax=255, origin='upper')
