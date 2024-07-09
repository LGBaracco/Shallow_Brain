import torch
from typing import List


def stimuli_extractor(batch):
    """Extracts stimuli from batches of inputs"""

    stimuli = torch.zeros(len(batch), 2)
    stimuli[:, 0] = batch[:, 7, 6]  # left
    stimuli[:, 1] = batch[:, 7, 25]  # right

    return stimuli


def get_possible_labels(cue_label, stimulus_label):
    """For full brain testing: get the motor labels that correspond to chances"""

    motor_label = []
    match (cue_label, stimulus_label):
        case (2, 0):  # PS, left brighter: left
            motor_label = [0, 2]
        case (2, 1):  # PS, right brighter: right
            motor_label = [1, 3]
        case (3, 0):  # AS, left brighter: right
            motor_label = [1]
        case (3, 1):  # AS, right brighter: left
            motor_label = [0]

    return motor_label


def generate_motor_labels(cue_labels, stimuli_labels) -> List:

    motor_labels = []
    for cue_label, stimulus_label in zip(cue_labels, stimuli_labels):
        match (cue_label, stimulus_label):
            case (2, 0):  # PS, left brighter: left
                motor_labels.append(0)
            case (2, 1):  # PS, right brighter: right
                motor_labels.append(1)
            case (3, 0):  # AS, left brighter: right
                motor_labels.append(1)
            case (3, 1):  # AS, right brighter: left
                motor_labels.append(0)

    return motor_labels


def get_decision_threshold(rates, threshold, dt):

    # convergence = rates[-1] * threshold
    convergence = threshold
    for i in range(rates.size(0)):

        if rates[i] >= convergence:
            return i * dt

    return None
