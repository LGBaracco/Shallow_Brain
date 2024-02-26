import itertools
import pickle as pkl
import numpy as np

from image_generator import *
from network import *
from training import *
from testing import *
from plotting import *
import torch


def main():

    # Hyperparameters
    ITERATIONS = 40
    BATCH_SIZE = 32
    LR = 0.003
    EPOCHS = 150
    TRAINING = 4  # -1, training 0 sanity check, 1 ambiguity test, 2 validation set, 3 training and get data, 4 plots

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cues, cue_labels = generate_cues()
    stimuli, stimuli_labels = generate_stimuli()

    if TRAINING == -1:

        # Training classifier
        data = np.concatenate((cues, stimuli), axis=0)
        labels = cue_labels + stimuli_labels

        network = training(data, labels, BATCH_SIZE, LR, EPOCHS, device)
        if network is None:
            main()  # Restart in case accuracy is not 100%

        # Fine-tuning motor areas
        stimuli = np.concatenate((stimuli, np.flip(stimuli, axis=0)), axis=0)
        stimuli_labels.extend(stimuli_labels[::-1])

        # match the size of cues and stimuli by repeating
        cycler = itertools.cycle(cue_labels)
        cue_labels = [next(cycler) for _ in range(len(stimuli_labels))]
        cycler = itertools.cycle(cues)
        cues = np.array([next(cycler) for _ in range(stimuli.shape[0])])

        motor_labels = generate_motor_labels(cue_labels, stimuli_labels)

        fine_tuning(network, cues, stimuli, motor_labels, BATCH_SIZE, LR, EPOCHS, device)

    elif TRAINING == 0:  # quick sanity check
        cues, stimuli = generate_sanity_check()

        network = ConvolutionalClassifier().to(device)
        state_dict = torch.load('./conv.pth')
        network.load_state_dict(state_dict)

        simple_test(network, cues, stimuli, device)

    elif TRAINING == 1:  # Get classification ratios of vague images ( both cues and stimuli)

        cues, stimuli = generate_vague()
        # _, stimuli = generate_sanity_check()

        network = ConvolutionalClassifier().to(device)
        state_dict = torch.load('./conv.pth')
        network.load_state_dict(state_dict)

        test_vague(network, cues, stimuli, device)

    elif TRAINING == 2:
        stimuli, labels = generate_test_set()

        network = ConvolutionalClassifier().to(device)
        state_dict = torch.load('./conv.pth')
        network.load_state_dict(state_dict)

        test(network, cues, stimuli, labels, device)

    elif TRAINING == 3:  # get NDArray with all final quantitative measures

        measures = np.zeros((ITERATIONS, 6))  # get all the necessary measures for each iteration
        vague_predictions = np.zeros((ITERATIONS, 11))
        for i in range(ITERATIONS):
            # Training classifier
            cues, cue_labels = generate_cues()
            stimuli, stimuli_labels = generate_stimuli()
            data = np.concatenate((cues, stimuli), axis=0)
            labels = cue_labels + stimuli_labels
            network, measure = training(data, labels, BATCH_SIZE, LR, EPOCHS, device)
            measures[i, 0] = measure

            # Fine-tuning motor areas
            stimuli = np.concatenate((stimuli, np.flip(stimuli, axis=0)), axis=0)
            stimuli_labels.extend(stimuli_labels[::-1])
            # match the size of cues and stimuli by repeating
            cycler = itertools.cycle(cue_labels)
            cue_labels = [next(cycler) for _ in range(len(stimuli_labels))]
            cycler = itertools.cycle(cues)
            cues = np.array([next(cycler) for _ in range(stimuli.shape[0])])
            motor_labels = generate_motor_labels(cue_labels, stimuli_labels)
            network, measure = fine_tuning(network, cues, stimuli, motor_labels, BATCH_SIZE, LR, EPOCHS*2, device)
            measures[i, 1] = measure

            # 2f brightness test set
            cues, _ = generate_cues()  # cues array has been messed with, so regenerate
            stimuli, labels = generate_test_set()
            brightness_accuracy, motor_accuracy = test(network, cues, stimuli, labels, device)
            measures[i, 2:4] = [brightness_accuracy, motor_accuracy]

            # Ambiguous test
            cues, stimuli = generate_vague()
            cue_ratio, stimuli_ratio, predicted_stimulis = test_vague(network, cues, stimuli, device)
            measures[i, 4:6] = [cue_ratio, stimuli_ratio]
            vague_predictions[i, :] = predicted_stimulis

        print(measures)
        with open('measures.npy', 'wb') as f:
            np.save(f, measures)  # save measures in an external file
        with open('brightnesses.npy', 'wb') as f:
            np.save(f, vague_predictions)  # save measures in an external file

    elif TRAINING == 4:
        measures = np.load('measures.npy')
        brightnesses = np.load('brightnesses.npy')
        plot_accuracy(measures[:, :4], False, True)
        plot_ratio(measures[:, 4:], False, True)
        plot_equal_brightness(brightnesses, False, True)


if __name__ == '__main__':
    main()
