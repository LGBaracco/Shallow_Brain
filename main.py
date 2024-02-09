from image_generator import *
from network import *
from typing import List
import numpy.typing as npt
import torch
from torch import nn, optim
from torch.utils.data import TensorDataset, DataLoader


def training(data: npt.NDArray, labels: List, batch_size: int, lr: float, epochs: int, device: torch.device):

    data = torch.from_numpy(data).float().to(device)
    labels = torch.tensor(labels).to(device)
    dataset = TensorDataset(data, labels)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    network = ConvolutionalClassifier().to(device)
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(network.parameters(), lr=lr)

    for name, param in network.named_parameters():
        if name == 'fc2.weight' and name == 'fc2.bias':
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


def transfer_learning(network, cues, stimuli, labels, batch_size: int, lr: float, epochs: int, device: torch.device):

    network = network.to(device)
    cues = torch.from_numpy(cues).float().to(device)
    stimuli = torch.from_numpy(stimuli).float().to(device)
    labels = torch.tensor(labels).to(device)

    cueset = TensorDataset(cues)
    cueloader = DataLoader(cueset, batch_size=batch_size, shuffle=False)
    stimuliset = TensorDataset(stimuli, labels)
    stimuliloader = DataLoader(stimuliset, batch_size=batch_size, shuffle=False)

    criterion = nn.NLLLoss()
    optimizer = optim.Adam(network.parameters(), lr=lr)

    for name, param in network.named_parameters():
        if name != 'fc2.weight' and name != 'fc2.bias':
            param.requires_grad = False
        else:
            param.requires_grad = True

    for epoch in range(epochs):
        running_loss = 0
        for cues, (stimuli, labels) in zip(cueloader, stimuliloader):
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
        for cues, (stimuli, labels) in zip(cueloader, stimuliloader):
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


def main():

    # Hyperparameters
    BATCH_SIZE = 32
    LR = 0.002
    EPOCHS = 100
    TRAINING = 0  # -1, training, 0 fine-tuning motor areas

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cues, cue_labels = generate_cues()
    stimuli, stimuli_labels = generate_stimuli()

    if TRAINING == -1:
        data = np.concatenate((cues, stimuli), axis=0)
        labels = cue_labels + stimuli_labels
        training(data, labels, BATCH_SIZE, LR, EPOCHS, device)
    elif TRAINING == 0:

        # match the size of cues and stimuli by repeating
        cue_labels = [next(itertools.cycle(cue_labels)) for _ in range(len(stimuli_labels))]
        cues = np.array([next(itertools.cycle(cues)) for _ in range(stimuli.shape[0])])

        # manual shuffling, as dataloader shuffling would not match cues and stimuli
        '''shuffler = np.random.permutation(stimuli.shape[0])
        cues = cues[shuffler, :, :]
        stimuli = stimuli[shuffler, :, :]
        for i in cue_labels[:]:
            cue_labels[i] = cue_labels[shuffler[i]]
            stimuli_labels[i] = stimuli_labels[shuffler[i]]'''

        motor_labels = generate_motor_labels(cue_labels, stimuli_labels)

        network = ConvolutionalClassifier().to(device)
        state_dict = torch.load('./conv.pth')
        network.load_state_dict(state_dict)

        transfer_learning(network, cues, stimuli, motor_labels, BATCH_SIZE, LR, EPOCHS, device)


if __name__ == '__main__':
    main()
