import torch
from torch.utils.data import TensorDataset, DataLoader


def simple_test(network, cues, stimuli, device):
    """Check whether every possible cue/stimuli pair label is correct and output in natural language
    NB: no quantitative measures are used (yet)"""

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
                if predicted == 2:
                    output = 'Prosaccade, '
                elif predicted == 3:
                    output = 'Antisaccade, '

                value = network(stimulus)
                _, predicted = torch.max(value, 1)
                if predicted == 0:
                    output += 'Brighter left, '
                elif predicted == 1:
                    output += 'Brighter right, '

                value = network(stimulus, cue)
                _, predicted = torch.max(value, 1)
                if predicted == 0:
                    output += 'Eye movement left'
                else:
                    output += 'Eye movement right'

                outputs.append(outputs)
                print(output)


def test(network, cues, stimuli, brightness_labels, device):
    """Test both the classification accuracy of both the stimuli and motor response using test set stimuli"""

    network = network.to(device)
    cues = torch.from_numpy(cues).float().to(device)
    stimuli = torch.from_numpy(stimuli).float().to(device)
    brightness_labels = torch.tensor(brightness_labels).float().to(device)

    dataset = TensorDataset(cues, stimuli, brightness_labels)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    correct_brightness = 0
    total = 0
    correct_motor = 0
    with torch.no_grad():
        for cues, stimuli, brightness_label in loader:

            # Classify stimuli
            outputs = network(stimuli)
            _, predicted_stimulus = torch.max(outputs, 1)
            total += brightness_label.size(0)
            correct_brightness += (predicted_stimulus == brightness_label).sum().item()

            # generate motor label
            outputs = network(cues)
            _, cue_label = torch.max(outputs, 1)
            if (cue_label == 1) ^ (brightness_label == 1):  # the cue/label match is an XOR
                motor_label = torch.tensor(1)
            else:
                motor_label = torch.tensor(0)

            outputs = network(stimuli, cues)
            _, predicted = torch.max(outputs, 1)
            correct_motor += (predicted == motor_label).sum().item()

    accuracy = 100 * correct_brightness / total
    print('brightness accuracy: ' + str(accuracy))
    accuracy = 100 * correct_motor / total
    print('motor accuracy: ' + str(accuracy))

    #ratio =
