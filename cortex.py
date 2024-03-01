import torch.nn as nn
import torch


class MLPClassifier(nn.Module):
    """MLP classifier, probably not useful anymore"""
    def __init__(self):
        super(MLPClassifier, self).__init__()

        self.fc1 = nn.Linear(16*32, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 4)

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.log_softmax(self.fc3(x), dim=1)

        return x


class ConvolutionalClassifier(nn.Module):
    def __init__(self):
        super(ConvolutionalClassifier, self).__init__()

        self.conv1 = nn.Conv2d(1, 4, 4)
        self.pool = nn.MaxPool2d(3, 2)
        self.conv2 = nn.Conv2d(4, 6, 4)
        self.fc1 = nn.Linear(30, 4)  # identify stimuli

        self.fc2 = nn.Linear(2, 32)
        self.fc3 = nn.Linear(32, 2)  # Motor output with cue

    def classifier(self, x):

        while type(x) is list:  # dirty trick: not yet sure why it's necessary
            x = x[0]

        x = x.unsqueeze(1)
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(x.shape[0], -1)
        x = self.fc1(x)

        return x

    def forward(self, x, cue=None):

        if cue is not None:

            cue_outputs = self.classifier(cue)
            _, cues = torch.max(cue_outputs, 1)
            cues = cues.unsqueeze(1)

            stimuli_outputs = self.classifier(x)
            _, stimuli = torch.max(stimuli_outputs, 1)
            stimuli = stimuli.unsqueeze(1)

            x = torch.cat([stimuli, cues], dim=1).float()

            # x[:, 2:4] = cue_labels[:, 2:4]

            x = torch.relu(self.fc2(x))
            x = self.fc3(x)
        else:
            x = self.classifier(x)

        return x
