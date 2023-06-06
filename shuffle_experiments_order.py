"""
TO GENERATE SHUFFLED EXPERIMENTS ORDER AND FREQUENCY TAGS
    Mo Shams <MShamsCBR@gmail.com>
    May 07, 2023
"""

import numpy

experiment_list = ["OBA-Central", "OBA-Peripheral",
                   "FBA-Central", "FBA-Peripheral"]
experiment_list = numpy.array(experiment_list)
numpy.random.shuffle(experiment_list)

freq_list = [7.5, 12]
freq_list = numpy.array(freq_list)
numpy.random.shuffle(freq_list)

print("===========================")
print("Shuffled experiments order:")
for index, item in enumerate(experiment_list):
    print(f"\t{index+1}: {item}.py")
print("...........................")
print("Shuffled frequency tags:")
for index, item in enumerate(freq_list):
    print(f"\tFreq{index+1}: {item} Hz")
print("===========================")
