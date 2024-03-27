import torch.nn as nn
import torch
from collections import OrderedDict


class ConvolutionalClassifier(nn.Module):
    def __init__(self, r_initial=0.1, tau=0.01, dt=0.001):
        super(ConvolutionalClassifier, self).__init__()

        self.conv1 = nn.Conv2d(1, 4, 4)
        self.pool = nn.MaxPool2d(3, 2)
        self.conv2 = nn.Conv2d(4, 6, 4)
        self.fc1 = nn.Linear(30, 4)  # identify stimuli

        self.fc2 = nn.Linear(4, 32)
        self.fc3 = nn.Linear(32, 2)  # Motor output with cue

        self.r_initial = r_initial
        self.tau = tau
        self.dt = dt

    def classify(self, x):

        while type(x) is list:  # dirty trick: not yet sure why it's necessary (might be due to 1-n batches)
            x = x[0]

        x = x.unsqueeze(1)
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(x.shape[0], -1)
        x = torch.log_softmax(self.fc1(x), dim=1)

        return x

    def forward(self, x, cue=None):

        if cue is not None:

            cue_outputs = self.classify(cue)
            #_, cues = torch.max(cue_outputs, 1)
            #cues = cues.unsqueeze(1)

            x = self.classify(x)
            #_, stimuli = torch.max(stimuli_outputs, 1)
            #stimuli = stimuli.unsqueeze(1)

            #x = torch.cat([stimuli, cues], dim=1).float()

            x[:, 2:4] = cue_outputs[:, 2:4]

            x = torch.relu(self.fc2(x))
            x = torch.log_softmax(self.fc3(x), dim=1)
        else:
            x = self.classify(x)

        return x

    def ahead(self, x, t, cue=None):

        match t:
            case 0:
                x = x.unsqueeze(1)
                return torch.relu(self.conv1(x))
            case 1:
                return self.pool(x)
            case 2:
                return torch.relu(self.conv2(x))
            case 3:
                return self.pool(x)
            case 4:
                x = x.view(x.shape[0], -1)
                x = torch.log_softmax(self.fc1(x), dim=1)

                if cue is not None:
                    #_, stimuli = torch.max(x, 1)
                    #stimuli = stimuli.unsqueeze(1)
                    #cue = cue.unsqueeze(1)
                    #x# = x.unsqueeze(1)
                    #y = torch.cat([x, cue], dim=1).float()
                    y = x
                    y[:, 2:4] = cue[:, 2:4]

                    return x, y
                else:
                    return x
            case 5:
                return torch.relu(self.fc2(x))
            case 6:
                return torch.log_softmax(self.fc3(x), dim=1)

    def time_evolution(self, x, total_timesteps, cue):

        r0 = r1 = r2 = r3 = r4 = r5 = self.r_initial

        y = x.unsqueeze(1)
        r0_values = torch.zeros((total_timesteps+1,) + y.size())
        r0_values[0] = r0
        y = self.pool(self.conv1(y))  # calculation is performed just to get the shape of the output tensor
        r1_values = torch.zeros((total_timesteps+1,) + y.size())
        r1_values[0] = r1
        y = self.pool(self.conv2(y))
        r2_values = torch.zeros((total_timesteps+1,) + y.size())
        r2_values[0] = r2
        y = self.fc1(y.view(y.shape[0], -1))
        r3_values = torch.zeros((total_timesteps+1,) + y.size())
        r3_values[0] = r3
        y = self.fc2(y)
        r4_values = torch.zeros((total_timesteps+1,) + y.size())
        r4_values[0] = r4
        y = self.fc3(y)
        r5_values = torch.zeros((total_timesteps+1,) + y.size())
        r5_values[0] = r5

        x = x.unsqueeze(1)
        for i in range(total_timesteps):
            r0 += (self.dt / self.tau) * (-r0 + x)
            r0_values[i+1] = r0

            f1 = self.pool(torch.relu(self.conv1(r0)))
            r1 += (self.dt / self.tau) * (-r1 + f1)
            r1_values[i+1] = r1

            f2 = self.pool(torch.relu(self.conv2(r1)))
            r2 += (self.dt / self.tau) * (-r2 + f2)
            r2_values[i+1] = r2

            r2_flat = r2.view(r2.shape[0], -1)
            f3 = torch.log_softmax(self.fc1(r2_flat), dim=1)
            cue_outputs = self.classify(cue)
            f3[:, 2:4] = cue_outputs[:, 2:4]
            r3 += (self.dt / self.tau) * (-r3 + f3)  # Choice: integrate only after merging cue and stimulus
            r3_values[i+1] = r3

            f4 = torch.relu(self.fc2(r3))
            r4 += (self.dt / self.tau) * (-r4 + f4)
            r4_values[i+1] = r4

            f5 = torch.log_softmax(self.fc3(r4), dim=1)
            r5 += (self.dt / self.tau) * (-r5 + f5)
            r5_values[i+1] = r5

        return r0_values, r1_values, r2_values, r3_values, r4_values, r5_values
