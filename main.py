from image_generator import *
from network import *
from typing import List
import numpy.typing as npt
import torch
from torch import nn, optim
from torch.utils.data import TensorDataset, DataLoader


def training(data: npt.NDArray, labels: List, batch_size: int, lr: int, epochs: int):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data = torch.from_numpy(data).float().to(device)
    labels = torch.tensor(labels).to(device)
    dataset = TensorDataset(data, labels)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    network = ConvolutionalClassifier().to(device)
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
        print(f"Training loss: {running_loss / len(dataloader)}")

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

    if type(network) is ConvolutionalClassifier():
        torch.save(network.state_dict(), './conv.pth')
    elif type(network) is MLPClassifier():
        torch.save(network.state_dict(), './mlp.pth')


def main():

    # Hyperparameters
    BATCH_SIZE = 32
    LR = 0.003
    EPOCHS = 30
    TRAINING = False  # whether we are training a network or performing the experiment

    cues, cue_labels = generate_cues()
    stimuli, stimuli_labels = generate_stimuli()

    if TRAINING:
        data = np.concatenate((cues, stimuli), axis=0)
        labels = cue_labels + stimuli_labels
        training(data, labels, BATCH_SIZE, LR, EPOCHS)
    else:
        test_data, test_labels = interleave(cues, cue_labels, stimuli, stimuli_labels)
        network = torch.load('./conv.pth')


if __name__ == '__main__':
    main()
