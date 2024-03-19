import torch
from torch.utils.data import TensorDataset, DataLoader
from utilfuncs import *


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

                outputs.append(output)
                print(output)


@torch.no_grad()
def test(network, stimuli, brightness_labels, device) -> float:
    """Test both the classification accuracy of both the stimuli and motor response using test set stimuli"""

    network = network.to(device)
    stimuli = torch.from_numpy(stimuli).float().to(device)
    brightness_labels = torch.tensor(brightness_labels).float().to(device)

    stimuli_dataset = TensorDataset(stimuli, brightness_labels)
    stimuli_loader = DataLoader(stimuli_dataset, batch_size=1, shuffle=False)

    correct_brightness = 0
    total = 0

    for stimuli, brightness_label in stimuli_loader:

        # Classify stimuli
        outputs = network(stimuli)
        _, predicted_stimulus = torch.max(outputs, 1)
        total += brightness_label.size(0)
        correct_brightness += (predicted_stimulus == brightness_label).sum().item()

        '''if (cue_label == 3) ^ (brightness_label == 1):  # the cue/label match is an XOR
            motor_label = torch.tensor(1)
        else:
            motor_label = torch.tensor(0)'''

    brightness_accuracy = 100 * correct_brightness / total
    print('brightness accuracy: ' + str(brightness_accuracy))

    return brightness_accuracy


@torch.no_grad()
def test_vague_cues(network, slanted_cues, device) -> float:
    """Test ambiguous input: slanted bars instead of cues
        output is prosaccade/antisaccade ratio"""

    network = network.to(device)
    slanted_cues = torch.from_numpy(slanted_cues).float().to(device)

    # test first using slanted cue bars and normal sampled stimuli
    cue_dataset = TensorDataset(slanted_cues)
    cue_loader = DataLoader(cue_dataset, batch_size=1, shuffle=False)
    prosaccades = 0  # Arbitrary: both prosaccade or antisaccade would have been fine
    total = 0
    for cues in cue_loader:

        outputs = network(cues)
        _, predicted_cues = torch.max(outputs, 1)

        total += predicted_cues.size(0)
        prosaccades += (predicted_cues == 2).sum().item()

    cue_ratio = 100 * prosaccades / total
    print('Prosaccade ratio: ' + str(cue_ratio))

    return cue_ratio


@torch.no_grad()
def test_vague_stimuli(network, equal_stimuli, device) -> (float, float):
    """Test ambiguous input: pairs of equal brightness
        output is left/right ratio"""

    network = network.to(device)
    equal_stimuli = torch.from_numpy(equal_stimuli).float().to(device)
    stimuli_dataset = TensorDataset(equal_stimuli)
    stimuli_loader = DataLoader(stimuli_dataset, batch_size=1, shuffle=False)

    left_detection = 0  # also arbitrary (as opposed to right)
    total = 0
    predicted_stimulis = []
    for stimuli in stimuli_loader:

        outputs = network(stimuli)
        _, predicted_stimuli = torch.max(outputs, 1)

        total += predicted_stimuli.size(0)
        left_detection += (predicted_stimuli == 0).sum().item()
        predicted_stimulis.append(predicted_stimuli.item())

    stimuli_ratio = 100 * left_detection / total
    print('Left-detection ratio: ' + str(stimuli_ratio))

    return stimuli_ratio, predicted_stimulis


@torch.no_grad()
def test_brain(network, cues, cue_labels, stimuli, stimuli_labels, device):
    """Testing the whole network combining cortex and subcortex. Check classification accuracy (should be 100%), and subcortex ratio"""
    network = network.to(device)

    cues = torch.from_numpy(cues).float().to(device)
    cue_labels = torch.tensor(cue_labels).to(device)
    stimuli = torch.from_numpy(stimuli).float().to(device)
    stimuli_labels = torch.tensor(stimuli_labels).to(device)

    cue_set = TensorDataset(cues, cue_labels)
    cue_loader = DataLoader(cue_set, batch_size=1, shuffle=False)
    stimuli_set = TensorDataset(stimuli, stimuli_labels)
    stimuli_loader = DataLoader(stimuli_set, batch_size=1, shuffle=False)

    correct_classified = 0
    total = 0
    total_subcortex = 0
    used_subcortex = 0
    for cue, cue_label in cue_loader:
        for stimulus, stimulus_label in stimuli_loader:

            value = network(stimulus, cue)
            _, predicted = torch.max(value, 1)

            labels = get_possible_labels(cue_label, stimulus_label)

            total += predicted.size(0)
            correct_classified += 1 if predicted in labels else 0

            if cue_label == 2:
                total_subcortex += predicted.size(0)
                if predicted.item() > 1:
                    used_subcortex += predicted.size(0)

    classification_accuracy = 100 * correct_classified / total
    subcortex_ratio = 100 * used_subcortex / total_subcortex

    print('Classification accuracy: ' + str(classification_accuracy))
    print('Subcortex ratio: ' + str(subcortex_ratio))

    return classification_accuracy, subcortex_ratio


def get_outputs(network, cues, stimuli):
    outputs = network(cues)
    _, predicted_cues = torch.max(outputs, 1)

    outputs = network(stimuli)
    _, predicted_stimulus = torch.max(outputs, 1)

    outputs = network(stimuli, cues)
    _, predicted_motor = torch.max(outputs, 1)

    return predicted_cues, predicted_stimulus, predicted_motor


@torch.no_grad()
def test_step_wise(network, cues, cue_labels, stimuli, stimuli_labels, device):
    """Testing the whole network combining cortex and subcortex using a time component."""
    network = network.to(device)

    cues = torch.from_numpy(cues).float().to(device)
    cue_labels = torch.tensor(cue_labels).to(device)
    stimuli = torch.from_numpy(stimuli).float().to(device)
    stimuli_labels = torch.tensor(stimuli_labels).to(device)

    cue_set = TensorDataset(cues, cue_labels)
    cue_loader = DataLoader(cue_set, batch_size=1, shuffle=False)
    stimuli_set = TensorDataset(stimuli, stimuli_labels)
    stimuli_loader = DataLoader(stimuli_set, batch_size=1, shuffle=False)

    correct_classified = 0
    total = 0
    measures = torch.zeros((len(stimuli_set)*len(cue_set), 7, 4))
    for cue, cue_label in cue_loader:
        for stimulus, stimulus_label in stimuli_loader:

            value = network.forward_step(stimulus, cue)
            _, predicted = torch.max(value[-1, :], 0)
            measures[total, :, :] = value

            labels = get_possible_labels(cue_label, stimulus_label)

            total += 1
            correct_classified += 1 if predicted in labels else 0

    classification_accuracy = 100 * correct_classified / total

    print('Classification accuracy: ' + str(classification_accuracy))

    return classification_accuracy, measures


