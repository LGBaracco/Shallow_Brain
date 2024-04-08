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
    def __init__(self, r_initial=0.1, tau=0.01, dt=0.001):
        super(SubcortexMLP, self).__init__()

        self.fc1 = nn.Linear(2, 16)
        self.fc2 = nn.Linear(16, 2, bias=False)

        self.r_initial = r_initial
        self.tau = tau
        self.dt = dt

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

    def time_evolution(self, x, total_timesteps, inhibition=1):
        x = stimuli_extractor(x).to(next(self.parameters()).device)

        r0_out = r1_out = r2_out = self.r_initial

        r0 = torch.full((x.size()), self.r_initial, device=x.device)
        r0_values = torch.zeros((total_timesteps+1,)+x.size())
        r0_values[0] = r0
        y = self.fc1(x)  # calculation is performed just to get the shape of the output tensor
        r1 = torch.full((y.size()), self.r_initial, device=x.device)
        r1_values = torch.zeros((total_timesteps+1,) + y.size())
        r1_values[0] = r1
        y = self.fc2(y)
        r2 = torch.full((y.size()), self.r_initial, device=x.device)
        r2_values = torch.zeros((total_timesteps+1,) + y.size())
        r2_values[0] = r2

        for i in range(total_timesteps):
            r0_out += (self.dt / self.tau) * (-r0 + x)
            r0_values[i+1] = r0_out

            f1 = torch.relu(self.fc1(r0))
            r1_out += (self.dt / self.tau) * (-r1 + f1)
            r1_values[i+1] = r1_out

            f2 = torch.softmax(self.fc2(r1), dim=1)
            r2_out += (self.dt / self.tau) * (-r2 + f2) * inhibition
            r2_values[i+1] = r2_out

            r0 = r0_out
            r1 = r1_out
            r2 = r2_out

        return r0_values, r1_values, r2_values
