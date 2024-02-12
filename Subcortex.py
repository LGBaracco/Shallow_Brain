import numpy as np
import scipy


class Subcortex:

    def __init__(self):

        self.weights = np.array([0.2, 0.2, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8])
        self.neurons = np.array([0.0, 0.0, 0.0, 0.0, 0.0])

    def saccade(self, image, cue_response=1):  # by default, let's say we do have PS

        left_brightness = image[6, 5]
        right_brightness = image[6, 25]

        self.neurons[0, 3] += [left_brightness*self.weights[0], right_brightness*self.weights[1], cue_response*self.weights[2]]

        for i, r in enumerate(self.neurons):
            r = -r