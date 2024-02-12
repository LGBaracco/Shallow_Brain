from image_generator import *
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
    """Train motor layer (stimulus with cue)"""

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


def simple_test(network, cues, stimuli, device):
    """Check whether every possible cue/stimuli pair label is correct and output in natural language"""

    network = network.to(device)
    cues = torch.from_numpy(cues).float().to(device)
    stimuli = torch.from_numpy(stimuli).float().to(device)

    cueset = TensorDataset(cues)
    cueloader = DataLoader(cueset, batch_size=1, shuffle=False)
    stimuliset = TensorDataset(stimuli)
    stimuliloader = DataLoader(stimuliset, batch_size=1, shuffle=False)

    outputs = []
    with torch.no_grad():
        for cue in cueloader:
            for stimulus in stimuliloader:

                value = network(cue)
                _, predicted = torch.max(value, 1)
                if predicted[0] == 2:
                    output = 'Prosaccade, '
                elif predicted[0] == 3:
                    output = 'Antisaccade, '

                value = network(stimulus, cue)
                _, predicted = torch.max(value, 1)
                if predicted[0] == 0:
                    output += 'Brighter left, '
                elif predicted[0] == 1:
                    output += 'Brighter right, '

                value = network(stimulus, cue)
                _, predicted = torch.max(value, 1)
                if predicted[0] == 0:
                    output += 'Eye movement left'
                else:
                    output += 'Eye movement right'

                outputs.append(outputs)
                print(output)


def vague_test():
    pass


def main():

    # Hyperparameters
    BATCH_SIZE = 32
    LR = 0.002
    EPOCHS = 100
    TRAINING = 0  # -1, training, 0 fine-tuning motor areas, 1 cue-stimuli matching test

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
        # Incorrect! got 100% accuracy regardless though
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

    elif TRAINING == 1:  # quick test
        cues, stimuli = generate_motor_test()

        network = ConvolutionalClassifier().to(device)
        state_dict = torch.load('./conv.pth')
        network.load_state_dict(state_dict)

        simple_test(network, cues, stimuli, device)

    elif TRAINING == 2:
        cues, _ = generate_motor_test()
        stimuli = generate_vague_stimuli()


if __name__ == '__main__':
    main()
