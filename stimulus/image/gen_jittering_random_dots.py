"""
This function will generate a random dot patch of certain number of dots

Mo Shams
m.shams.ahmar@gmail.com
March 2024
"""

import numpy as np
import matplotlib.pyplot as plt

# Parameters
patch_radius = 300  # pixels
dot_radius = 4  # pixels
number_of_dots = 250
fix_radius = 30  # pixels
motion_step = dot_radius/3  # pixels

blue_rgb = [0, 60, 255]
red_rgb = [170, 0, 0]

color_arr = np.repeat(['r', 'b'], number_of_dots // 2)
np.random.shuffle(color_arr)

# Generate random dot coordinates within circular patch
dot_coords = []
while len(dot_coords) < number_of_dots:
    x = np.random.uniform(-patch_radius + dot_radius,
                          patch_radius - dot_radius)
    y = np.random.uniform(-patch_radius + dot_radius,
                          patch_radius - dot_radius)
    dot_rad = np.sqrt(x ** 2 + y ** 2)
    if fix_radius <= dot_rad <= (patch_radius - dot_radius):
        dot_coords.append([x, y])

image_size = 2 * patch_radius  # pixels

for i_image in range(10):
    # Create an array representing the image
    image = np.zeros((image_size, image_size, 3), dtype=np.uint8)  # RGBA
    # Draw dots on the image
    for dot_ind, (x, y) in enumerate(dot_coords):
        dot_x, dot_y = int(x + image_size / 2), \
                       int(y + image_size / 2)
        dot_center = (dot_x, dot_y)
        # Draw a circle for each dot
        for i in range(dot_y - dot_radius,
                       dot_y + dot_radius):
            for j in range(dot_x - dot_radius,
                           dot_x + dot_radius):
                if (i - dot_y) ** 2 + (j - dot_x) ** 2 <= dot_radius ** 2:
                    if color_arr[dot_ind] == 'b':
                        image[i, j, :] = blue_rgb
                    if color_arr[dot_ind] == 'r':
                        image[i, j, :] = red_rgb
        # update coordinates of each dot
        change_axis = np.random.choice(['x', 'y'])
        change_direction = np.random.choice([-1, 1])
        if change_axis == 'x':
            dot_xnew = x + change_direction * motion_step
        else:
            dot_xnew = x
        if change_axis == 'y':
            dot_ynew = y + change_direction * motion_step
        else:
            dot_ynew = y
        if np.sqrt(dot_xnew ** 2 + dot_ynew ** 2) > patch_radius:
            flag_add = True
            while flag_add:
                dot_xnew = np.random.uniform(-patch_radius + dot_radius,
                                             patch_radius - dot_radius)
                dot_ynew = np.random.uniform(-patch_radius + dot_radius,
                                             patch_radius - dot_radius)
                dot_new_rad = np.sqrt(dot_xnew ** 2 + dot_ynew ** 2)
                if fix_radius <= dot_new_rad <= (patch_radius - dot_radius):
                    flag_add = False
        dot_coords[dot_ind] = [dot_xnew, dot_ynew]

    # cover the screen with the full sheet with an apperture

    # Save the image as a transparent PNG file using matplotlib
    plt.imsave(f"random_dots/random_dot_patch_{i_image}.png",
               image, format="png", cmap="gray",
               vmin=0, vmax=255, origin='upper')
