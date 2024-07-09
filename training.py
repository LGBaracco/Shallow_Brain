from cortex import *
from subcortex import *
from utilfuncs import *
from typing import List
import numpy.typing as npt
import torch
from torch import nn, optim
from torch.utils.data import TensorDataset, DataLoader


def training(data: npt.NDArray, labels: List, batch_size: int, lr: float, epochs: int, device: torch.device) \
        -> (torch.nn.Module, float):
    """Train single stimuli or cues classifier (either MLP or convolutional)"""

    data = torch.from_numpy(data).float().to(device)
    labels = torch.tensor(labels).to(device)
    dataset = TensorDataset(data, labels)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    network = ConvolutionalClassifier().to(device)
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(network.parameters(), lr=lr)

    for name, param in network.named_parameters():
        if name == 'fc2.weight' or name == 'fc2.bias' or name == 'fc3.weight' or name == 'fc3.bias':
            param.requires_grad = False

    for epoch in range(epochs):
        running_loss = 0
        for images, labels in dataloader:
            optimizer.zero_grad()

            log_ps = network(images)

            loss = criterion(log_ps, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        if epoch % 50 == 0:
            print(f"Epoch #{epoch+1} Training loss: {running_loss / len(dataloader)}")

    # accuracy
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in dataloader:

            outputs = network(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print('accuracy: ' + str(accuracy))

    return network, accuracy


def fine_tuning(network, cues, stimuli, labels, batch_size: int, lr: float, epochs: int, device: torch.device) \
        -> (torch.nn.Module, float):
    """Train motor layer (stimuli with cue)"""

    network = network.to(device)
    cues = torch.from_numpy(cues).float().to(device)
    stimuli = torch.from_numpy(stimuli).float().to(device)
    labels = torch.tensor(labels).to(device)

    stimuli_set = TensorDataset(cues, stimuli, labels)
    stimuli_loader = DataLoader(stimuli_set, batch_size=batch_size, shuffle=True)

    criterion = nn.NLLLoss()
    optimizer = optim.Adam(network.parameters(), lr=lr)

    for name, param in network.named_parameters():
        if name != 'fc2.weight' and name != 'fc2.bias' and name != 'fc3.weight' and name != 'fc3.bias':
            param.requires_grad = False
        else:
            param.requires_grad = True

    for epoch in range(epochs):
        running_loss = 0
        for cues, stimuli, labels in stimuli_loader:
            optimizer.zero_grad()

            log_ps = network(stimuli, cues)
    
            loss = criterion(log_ps, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        if epoch % 50 == 0:
            print(f"Epoch #{epoch+1} Training loss: {running_loss / len(stimuli_loader)}")

    # accuracy
    correct = 0
    total = 0
    with torch.no_grad():
        for cues, stimuli, labels in stimuli_loader:
            outputs = network(stimuli, cues)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print('accuracy: ' + str(accuracy))

    if accuracy == 100.0:
        torch.save(network.state_dict(), 'data/conv.pth')
        print('saved!')

    return network, accuracy


def train_subcortex(data: npt.NDArray, labels: List, batch_size: int, lr: float, epochs: int, device: torch.device):
    """Train MLP implementation of subcortical pathway"""

    data = torch.from_numpy(data).float().to(device)
    labels = torch.tensor(labels).to(device)
    dataset = TensorDataset(data, labels)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    network = SubcortexMLP().to(device)
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(network.parameters(), lr=lr)

    for epoch in range(epochs):
        running_loss = 0
        for images, labels in dataloader:
            optimizer.zero_grad()

            log_ps = network(images)

            loss = criterion(log_ps, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        if epoch % 50 == 0:
            print(f"Epoch #{epoch+1} Training loss: {running_loss / len(dataloader)}")

    # accuracy
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in dataloader:

            outputs = network(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print('accuracy: ' + str(accuracy))

    # Do not save model if accuracy is not 100%
    if accuracy == 100.0:
        torch.save(network.state_dict(), 'data/subc.pth')
        print('saved!')

    return network, accuracy
