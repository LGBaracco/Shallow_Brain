import numpy as np

from image_generator import *
from network import *
from training import *
from testing import *
import torch


# TODO:
#  fabricate test with all outputs and see the issue
#  test network bias with equal brightness training/test sets (left/right brightness ratio as a measure)
#  test network bias with slanted cues and unequal training/test sets (prosaccade/antisaccade ratio as a measure)
#  quantify brightness and motor output accuracy (all cases)
#  visualizations (both methods and results)


def main():

    # Hyperparameters
    BATCH_SIZE = 32
    LR = 0.0003
    EPOCHS = 3000
    TRAINING = 0  # -1, training, 0 fine-tuning motor areas, 1 sanity check, 2 other tests

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cues, cue_labels = generate_cues()
    stimuli, stimuli_labels = generate_stimuli()

    if TRAINING == -1:
        data = np.concatenate((cues, stimuli), axis=0)
        labels = cue_labels + stimuli_labels
        training(data, labels, BATCH_SIZE, LR, EPOCHS, device)
    elif TRAINING == 0:

        stimuli = np.concatenate((stimuli, np.flip(stimuli, axis=0)), axis=0)
        stimuli_labels.extend(stimuli_labels[::-1])

        # match the size of cues and stimuli by repeating
        cycler = itertools.cycle(cue_labels)
        cue_labels = [next(cycler) for _ in range(len(stimuli_labels))]
        cycler = itertools.cycle(cues)
        cues = np.array([next(cycler) for _ in range(stimuli.shape[0])])

        motor_labels = generate_motor_labels(cue_labels, stimuli_labels)

        network = ConvolutionalClassifier().to(device)
        state_dict = torch.load('./conv.pth')
        network.load_state_dict(state_dict)

        fine_tuning(network, cues, stimuli, motor_labels, BATCH_SIZE, LR, EPOCHS, device)

    elif TRAINING == 1:  # quick sanity check
        cues, stimuli = generate_sanity_check()

        network = ConvolutionalClassifier().to(device)
        state_dict = torch.load('./conv.pth')
        network.load_state_dict(state_dict)

        simple_test(network, cues, stimuli, device)

    elif TRAINING == 2:  # quick QUALITATIVE analysis of vague stimuli

        cues, _ = generate_vague()
        _, stimuli = generate_sanity_check()

        network = ConvolutionalClassifier().to(device)
        state_dict = torch.load('./conv.pth')
        network.load_state_dict(state_dict)

        simple_test(network, cues, stimuli, device)

    elif TRAINING == 3:
        stimuli, labels = generate_test_set()

        cycler = itertools.cycle(cues)
        cues = np.array([next(cycler) for _ in range(stimuli.shape[0])])

        network = ConvolutionalClassifier().to(device)
        state_dict = torch.load('./conv.pth')
        network.load_state_dict(state_dict)

        test(network, cues, stimuli, labels, device)


if __name__ == '__main__':
    main()
