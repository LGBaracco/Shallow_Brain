import itertools
from image_generator import *
from network import *
from training import *
from testing import *
import torch


# TODO:
#  quantify brightness and motor output accuracy (all cases)
#  visualizations (both methods and results)


def main():

    # Hyperparameters
    ITERATIONS = 20
    BATCH_SIZE = 32
    LR = 0.002
    EPOCHS = 150
    TRAINING = -1  # -1, training 0 sanity check, 1 ambiguity test, 2 validation set

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

        cycler = itertools.cycle(cues)
        cues = np.array([next(cycler) for _ in range(stimuli.shape[0])])

        network = ConvolutionalClassifier().to(device)
        state_dict = torch.load('./conv.pth')
        network.load_state_dict(state_dict)

        test(network, cues, stimuli, labels, device)


if __name__ == '__main__':
    main()
