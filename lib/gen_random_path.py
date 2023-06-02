"""
This funcion generates a 2D centripetal brownian motion path
Transcribed from its previous MATLAB version 2022-02-16

Mohammad Shams
m.shams.ahmar@gmail.com
last modification: 2022-10-15
"""

import numpy as np


def brownian_2d(n_samples=10, distribution_sigma=10, max_step=10):
    # constants
    distribution_n = 100000
    distribution_mu = 0

    # generate the Gaussian distribution
    distribution = np.random.normal(distribution_mu,
                                    distribution_sigma,
                                    distribution_n)

    # set the starting point
    x = np.array([0])
    y = np.array([0])

    for isample in range(n_samples - 1):
        # crop the distribution within a "max_step" from the last sample
        ind_samples_x = ((x[-1] - max_step) < distribution) & \
                        (distribution < (x[-1] + max_step))
        ind_samples_y = ((y[-1] - max_step) < distribution) & \
                        (distribution < (y[-1] + max_step))
        data_x = distribution[ind_samples_x]
        data_y = distribution[ind_samples_y]
        # sample from the cropped distribution
        x_new = np.random.choice(data_x, 1)
        y_new = np.random.choice(data_y, 1)
        # append the sampled data to the existing path data
        x = np.append(x, x_new)
        y = np.append(y, y_new)

    return x
