import numpy as np
import scipy

import torch.nn as nn
import torch
from utilfuncs import *


class SubcortexWW:
    """model of the subcortical pathway based on Wong-wang model"""

    def __init__(self):

        self.weights = np.array([0.2, 0.2, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8])
        self.neurons = np.array([0.0, 0.0, 0.0, 0.0, 0.0])

    def saccade(self, image, cue_response=1):  # by default, let's say we do have PS

        left_brightness = image[6, 5]
        right_brightness = image[6, 25]

        self.neurons[0, 3] += [left_brightness*self.weights[0], right_brightness*self.weights[1], cue_response*self.weights[2]]

        for i, r in enumerate(self.neurons):
            r = r


class SubcortexMLP(nn.Module):
    """Model of the subcortical pathway based on a simple one-hidden-layer Perceptron"""
    def __init__(self):
        super(SubcortexMLP, self).__init__()

        self.fc1 = nn.Linear(2, 16)
        self.fc2 = nn.Linear(16, 2, bias=False)

    def forward(self, x):
        while type(x) is list:  # dirty trick: not yet sure why it's necessary
            x = x[0]

        x = stimuli_extractor(x).to(next(self.parameters()).device)
        x = torch.relu(self.fc1(x))
        x = torch.log_softmax(self.fc2(x), dim=1)

        return x

    def ahead(self, x, t):

        match t:
            case 0:
                return stimuli_extractor(x).to(next(self.parameters()).device)
            case 1:
                return torch.relu(self.fc1(x))
            case 2:
                return torch.log_softmax(self.fc2(x), dim=1)
