import torch.nn as nn
import torch


# TODO: maybe get rid of logsoftmax (probably not), do the vague tests

class MLPClassifier(nn.Module):
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

        self.conv1 = nn.Conv2d(1, 6, 4)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 12, 4)
        self.fc1 = nn.Linear(60, 4)  # identify stimuli

        self.fc2 = nn.Linear(4, 16)
        self.fc3 = nn.Linear(16, 2)  # Motor output with cue

    def classifier(self, x):

        while type(x) is list:  # dirty trick: not yet sure why it's necessary
            x = x[0]

        x = x.unsqueeze(1)
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(x.shape[0], -1)
        x = torch.log_softmax(self.fc1(x), dim=1)

        return x

    def forward(self, x, cue=None):

        if cue is not None:
            cue_labels = self.classifier(cue)

            x = self.classifier(x)
            x[:, 2:4] = cue_labels[:, 2:4]

            x = torch.relu(self.fc2(x))
            x = torch.log_softmax(self.fc3(x), dim=1)
        else:
            x = self.classifier(x)

        return x
