from network import *
from typing import List
import numpy.typing as npt
import torch
from torch import nn, optim
from torch.utils.data import TensorDataset, DataLoader


def training(data: npt.NDArray, labels: List, batch_size: int, lr: float, epochs: int, device: torch.device):
    """Train single stimuli or cues classifier (either MLP or convolutional)"""

    data = torch.from_numpy(data).float().to(device)
    labels = torch.tensor(labels).to(device)
    dataset = TensorDataset(data, labels)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    network = ConvolutionalClassifier().to(device)
    criterion = nn.CrossEntropyLoss()
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

    if accuracy != 100.0:
        return  # Do not save model if accuracy is not 100%

    if type(network) is ConvolutionalClassifier:
        torch.save(network.state_dict(), './conv.pth')
        print('saved!')
    elif type(network) is MLPClassifier:
        torch.save(network.state_dict(), './mlp.pth')
        print('saved!')

    return network


def fine_tuning(network, cues, stimuli, labels, batch_size: int, lr: float, epochs: int, device: torch.device):
    """Train motor layer (stimulus with cue)"""

    network = network.to(device)
    cues = torch.from_numpy(cues).float().to(device)
    stimuli = torch.from_numpy(stimuli).float().to(device)
    labels = torch.tensor(labels).to(device)

    stimuliset = TensorDataset(cues, stimuli, labels)
    stimuliloader = DataLoader(stimuliset, batch_size=batch_size, shuffle=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(network.parameters(), lr=lr)

    for name, param in network.named_parameters():
        if name != 'fc2.weight' and name != 'fc2.bias' and name != 'fc3.weight' and name != 'fc3.bias':
            param.requires_grad = False
        else:
            param.requires_grad = True

    for epoch in range(epochs):
        running_loss = 0
        for cues, stimuli, labels in stimuliloader:
            optimizer.zero_grad()

            log_ps = network(stimuli, cues)
    
            loss = criterion(log_ps, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        print(f"Epoch #{epoch+1} Training loss: {running_loss / len(stimuliloader)}")

    # accuracy
    correct = 0
    total = 0
    with torch.no_grad():
        for cues, stimuli, labels in stimuliloader:
            outputs = network(stimuli, cues)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print('accuracy: ' + str(accuracy))

    if accuracy != 100.0:
        return  # Do not save model if accuracy is not 100%

    if type(network) is ConvolutionalClassifier:
        torch.save(network.state_dict(), './conv.pth')
        print('saved!')
    elif type(network) is MLPClassifier:
        torch.save(network.state_dict(), './mlp.pth')
        print('saved!')
