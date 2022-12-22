"""
Randomly selects events from a given list (start,stop), keeping a minimum
distance between each selected item.

Mohammad Shams <m.shams.ahmar@gmail.com>
initiated on: December 20, 2022
"""

import random
import numpy as np


def gen_ditributed_event(start, stop,
                         sampling_start, sampling_stop,
                         min_distance):
    step = 1
    original_array = list(range(sampling_start, sampling_stop + 1, step))
    sample_arr = [start, stop]
    output_samples = [None]
    cont_flag = True
    while cont_flag:
        curr_sample = random.choice(original_array)
        sample_arr.append(curr_sample)
        sample_arr.sort()
        cont_flag = (all(np.diff(sample_arr) >= min_distance))
        if cont_flag:
            output_samples.append(curr_sample)
    output_samples.pop(0)
    output_samples.sort()
    return output_samples
