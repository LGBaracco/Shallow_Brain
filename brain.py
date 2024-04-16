import torch.nn as nn
import torch
import itertools
from utilfuncs import *


class ANNBrain(nn.Module):
    """Full brain network combining cortical and subcortical pathways"""
    def __init__(self, cortex, subcortex):
        super(ANNBrain, self).__init__()

        self.cortex = cortex
        self.subcortex = subcortex

        self.inhibition = 99999.9  # Values are already negative, so inhibitory weight needs to be positive

    def forward(self, x, cue=None):
        if cue is None:
            return self.cortex(x)  # or subcortex?

        _, saccade = torch.max(self.cortex.classify(cue), 1)

        x_subcortex = self.subcortex(x)

        x_subcortex *= self.inhibition * (saccade.item() - 2)

        x_cortex = self.cortex(x, cue)

        # x = torch.cat([x_cortex, x_subcortex], dim=1)

        return x_subcortex + x_cortex

    def forward_step(self, x, cue=None):
        while type(x) is list:  # dirty trick: not yet sure why it's necessary (might be due to 1-n batches)
            x = x[0]
        if cue is None:
            return self.cortex(x)  # or subcortex?

        saccade = self.cortex.classify(cue)

        m = torch.zeros((7, 4))

        x_subcortex = None
        x_cortex = None
        for i in range(7):

            match i:
                case 0:
                    x_subcortex = self.subcortex.ahead(x, i)
                    x_cortex = self.cortex.ahead(x, i, saccade)
                case 1:
                    _, cues = torch.max(saccade, 1)
                    x_subcortex = self.subcortex.ahead(x_subcortex, i) * (- (cues - 3))  # if antisaccade, multiply by 0
                    x_cortex = self.cortex.ahead(x_cortex, i, saccade)
                case 2:  # subcortex output
                    x_subcortex = self.subcortex.ahead(x_subcortex, i)
                    m[i, 2:4] = x_subcortex
                    x_cortex = self.cortex.ahead(x_cortex, i, saccade)
                case 3:
                    m[i, 2:4] = m[i-1, 2:4]
                    x_cortex = self.cortex.ahead(x_cortex, i, saccade)
                case 4:
                    m[i, 2:4] = m[i-1, 2:4]
                    outputs, x_cortex = self.cortex.ahead(x_cortex, i, saccade)
                    #m[i, 0:2] = outputs[0, 2]
                case 5:
                    m[i, 2:4] = m[i-1, 2:4]
                    #m[i, 0:2] = m[i-1, 2:4]
                    x_cortex = self.cortex.ahead(x_cortex, i, saccade)
                case _:
                    m[i, 2:4] = m[i-1, 2:4]
                    m[i, 0:2] = self.cortex.ahead(x_cortex, i, saccade)

        return m

    def time_evolution(self, x, total_timesteps, cue):

        _, cue_output = torch.max(self.cortex.classify(cue), 1)
        inhibition = (- (cue_output - 3))  # 1 if prosaccade, 0 if antisaccade

        cortex_activations = self.cortex.time_evolution(x, total_timesteps, cue)

        subcortex_activations = self.subcortex.time_evolution(x, total_timesteps, inhibition)

        final_activation = cortex_activations[-1] + subcortex_activations[-1]

        return cortex_activations, subcortex_activations, final_activation
