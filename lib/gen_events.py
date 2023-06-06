"""
Divides a 10 sec window into three segments of 2 sec and gap segments of 1 sec.
Then randomly decides how many events, and in which segments should appear.

Mohammad Shams <m.shams.ahmar@gmail.com>
initiated on: December 21, 2022
"""

import random
import numpy as np


def gen_events(ref_rate):
    segment_numbers = np.array([1, 2, 3])
    segment_times = [1 * ref_rate,
                     4 * ref_rate,
                     7 * ref_rate]
    n_events = np.random.choice(segment_numbers, p=[.1, .4, .5])
    np.random.shuffle(segment_numbers)
    event_frames = np.full((n_events,), np.nan)
    for ievent in range(n_events):
        event_frames[ievent] = random.choice(range(2 * ref_rate)) + \
                              segment_times[segment_numbers[ievent] - 1]
    return event_frames


def gen_events2(ref_rate):
    segment_numbers = np.array([1, 2])
    segment_times = [1 * ref_rate,
                     4 * ref_rate]
    n_events = np.random.choice(segment_numbers, p=[.3, .7])
    np.random.shuffle(segment_numbers)
    event_frames = np.full((n_events,), np.nan)
    for ievent in range(n_events):
        event_frames[ievent] = random.choice(range(2 * ref_rate)) + \
                              segment_times[segment_numbers[ievent] - 1]
    return event_frames
