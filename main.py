import itertools
import numpy as np

from image_generator import *
from utilfuncs import *
from cortex import *
from subcortex import *
from brain import *
from training import *
from testing import *
from plotting import *
import torch


def main():

    # Hyperparameters
    # Training parameters
    ITERATIONS = 30  # re-training iterations
    BATCH_SIZE = 32
    LR = 0.003
    EPOCHS = 150

    # Temporal parameters
    TIMESTEPS = 120
    DT = 0.01
    TAU = 0.1
    R_INITIAL = 0.0
    THRESHOLD = 0.7  # convergence decision threshold

    # Change this parameter to decide what action to perform
    # -1 training subcortex 0 training cortex, 1 and 2 training/testing subcortex and cortex, 3 plotting subcort and cort,
    # 4 test accuracy full brain, 5 step-wise analysis,
    # 6 continuous time analysis (sanity check), 7 full continuous time analysis,
    # 8 full ct analysis across training sessions (data for paper plots) # 9 plotting full ct analysis (get data from 8)
    TRAINING = 8

    # Plotting parameters
    VIEW = True
    SAVE = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cues, cue_labels = generate_cues()
    stimuli, stimuli_labels = generate_stimuli()

    if TRAINING == -1:
        network, _ = train_subcortex(stimuli, stimuli_labels, BATCH_SIZE, LR, EPOCHS, device)
        if network is None:
            main()  # Restart in case accuracy is not 100%

    elif TRAINING == 0:

        # Training classifier
        data = np.concatenate((cues, stimuli), axis=0)
        labels = cue_labels + stimuli_labels

        network, _ = training(data, labels, BATCH_SIZE, LR, EPOCHS, device)
        if network is None:
            main()  # Restart in case accuracy is not 100%

        # Train motor area
        fine_tune_network(cues, cue_labels, stimuli, stimuli_labels, network, BATCH_SIZE, LR, EPOCHS, device)

    elif TRAINING == 1:

        measures = np.zeros((ITERATIONS, 3))  # get all the necessary measures for each iteration
        vague_predictions = np.zeros((ITERATIONS, 11))
        for i in range(ITERATIONS):
            torch.cuda.seed()

            stimuli, stimuli_labels = generate_stimuli()
            network, measure = train_subcortex(stimuli, stimuli_labels, BATCH_SIZE, LR, EPOCHS, device)
            measures[i, 0] = measure

            # 2f brightness test set
            stimuli, labels = generate_test_set()
            brightness_accuracy = test_subcortex(network, stimuli, labels, device)
            measures[i, 1] = brightness_accuracy

            # Ambiguous test
            _, stimuli = generate_vague()
            stimuli_ratio, predicted_stimulis = test_vague_stimuli(network, stimuli, device)
            measures[i, 2] = stimuli_ratio
            vague_predictions[i, :] = predicted_stimulis

            print(measures)
            with open('data/measures_subcortex.npy', 'wb') as f:
                np.save(f, measures)  # save measures in an external file
            with open('data/brightnesses_subcortex.npy', 'wb') as f:
                np.save(f, vague_predictions)  # save measures in an external file

    elif TRAINING == 2:  # get NDArray with all final quantitative measures

        measures = np.zeros((ITERATIONS, 6))  # get all the necessary measures for each iteration
        vague_predictions = np.zeros((ITERATIONS, 11))
        for i in range(ITERATIONS):
            print('Iteration #' + str(i+1))
            torch.cuda.seed()
            # Training classifier
            cues, cue_labels = generate_cues()
            stimuli, stimuli_labels = generate_stimuli()
            data = np.concatenate((cues, stimuli), axis=0)
            labels = cue_labels + stimuli_labels
            network, measure = training(data, labels, BATCH_SIZE, LR, EPOCHS, device)
            measures[i, 0] = measure

            # Fine-tuning motor pathway
            network, measure = fine_tune_network(cues, cue_labels, stimuli, stimuli_labels, network, BATCH_SIZE, LR, EPOCHS*2, device)
            measures[i, 1] = measure

            # 2f brightness test set
            stimuli, labels = generate_test_set()
            brightness_accuracy, motor_accuracy = test_cortex(network, stimuli, labels, cues, device)
            measures[i, 2:4] = brightness_accuracy

            # Ambiguous test
            cues, stimuli = generate_vague()
            cue_ratio = test_vague_cues(network, cues, device)
            stimuli_ratio, predicted_stimulis = test_vague_stimuli(network, stimuli, device)
            measures[i, 4:6] = [cue_ratio, stimuli_ratio]
            vague_predictions[i, :] = predicted_stimulis

        print(measures)
        with open('data/measures.npy', 'wb') as f:
            np.save(f, measures)  # save measures in an external file
        with open('data/brightnesses.npy', 'wb') as f:
            np.save(f, vague_predictions)  # save measures in an external file

    elif TRAINING == 3:
        # Subcortex
        measures = np.load('data/measures_subcortex.npy')
        brightnesses = np.load('data/brightnesses_subcortex.npy')
        plot_accuracy(measures[:, :2], VIEW, SAVE)
        plot_ratio(measures[:, 2:], VIEW, SAVE)
        plot_equal_brightness(brightnesses, VIEW, SAVE, True)

        # Cortex
        measures = np.load('data/measures.npy')
        brightnesses = np.load('data/brightnesses.npy')
        plot_accuracy(measures[:, :4], VIEW, SAVE)
        plot_ratio(measures[:, 4:], VIEW, SAVE)
        plot_equal_brightness(brightnesses, VIEW, SAVE, False)

    elif TRAINING == 4:

        cortex = ConvolutionalClassifier().to(device)
        state_dict = torch.load('data/conv.pth')
        cortex.load_state_dict(state_dict)
        subcortex = SubcortexMLP().to(device)
        state_dict = torch.load('data/subc.pth')
        subcortex.load_state_dict(state_dict)

        network = ANNBrain(cortex, subcortex)

        accuracy = test_brain(network, cues, cue_labels, stimuli, stimuli_labels, device)
        print(accuracy)

        # plot_brain(measures, False, False) Deprecated: plots subcortex ratio when outputs are concatenated

    elif TRAINING == 5:

        cortex = ConvolutionalClassifier().to(device)
        state_dict = torch.load('data/conv.pth')
        cortex.load_state_dict(state_dict)
        subcortex = SubcortexMLP().to(device)
        state_dict = torch.load('data/subc.pth')
        subcortex.load_state_dict(state_dict)

        network = ANNBrain(cortex, subcortex)

        accuracy, measures = test_step_wise(network, cues, cue_labels, stimuli, stimuli_labels, device)

        print(accuracy, measures)
        plot_heatmap(measures, VIEW, SAVE)

        # plot_brain(measures, False, False)

    elif TRAINING == 6:
        # Subcortex
        network = SubcortexMLP(dt=DT, r_initial=R_INITIAL).to(device)
        state_dict = torch.load('data/subc.pth')
        network.load_state_dict(state_dict)

        data = torch.from_numpy(stimuli).float().to(device)
        with torch.no_grad():
            activations = network.time_evolution(data[0:1, :, :], TIMESTEPS)
        plot_evolution(activations, TIMESTEPS, DT, VIEW, SAVE)

        # Cortex
        network = ConvolutionalClassifier(dt=DT, r_initial=R_INITIAL).to(device)
        state_dict = torch.load('data/conv.pth')
        network.load_state_dict(state_dict)

        data = torch.from_numpy(stimuli).float().to(device)
        cues = torch.from_numpy(cues).float().to(device)
        # result = network(data[0:1, :, :], cues[0:1, :, :])
        with torch.no_grad():
            activations = network.time_evolution(data[0:1, :, :], TIMESTEPS, cues[0:1, :, :])

        plot_evolution(activations, TIMESTEPS, DT, VIEW, SAVE)

    elif TRAINING == 7:

        cortex = ConvolutionalClassifier(dt=DT, tau=TAU, r_initial=R_INITIAL).to(device)
        state_dict = torch.load('data/conv.pth')
        cortex.load_state_dict(state_dict)
        subcortex = SubcortexMLP(dt=DT, tau=TAU, r_initial=R_INITIAL).to(device)
        state_dict = torch.load('data/subc.pth')
        subcortex.load_state_dict(state_dict)
        network = ANNBrain(cortex, subcortex)

        stimuli, stimuli_labels = generate_test_set()
        # stimuli += np.abs(np.random.normal(loc=0.0, scale=0.2, size=stimuli.shape))  # Gaussian noise
        data = torch.from_numpy(stimuli).float().to(torch.device('cuda'))
        cues = torch.from_numpy(cues).float().to(torch.device('cuda'))

        with torch.no_grad():
            (_, _, _, _, _, cortex_measures_pro), (_, _, subcortex_measures_pro), finals_pro = network.time_evolution(data, TIMESTEPS, cues[0:1, :, :])
            (_, _, _, _, _, cortex_measures_anti), (_, _, subcortex_measures_anti), finals_anti = network.time_evolution(data, TIMESTEPS, cues[-1:, :, :])

        plot_decision_evolution(cortex_measures_pro, subcortex_measures_pro, cortex_measures_anti,
                                subcortex_measures_anti,
                                TIMESTEPS, DT, stimuli_labels, THRESHOLD, VIEW, SAVE)
        plot_rt_histogram(cortex_measures_pro, subcortex_measures_pro, cortex_measures_anti,
                          DT, stimuli_labels, THRESHOLD, VIEW, SAVE)
        plot_rt_mixed(cortex_measures_pro, subcortex_measures_pro, cortex_measures_anti,
                      DT, stimuli_labels, THRESHOLD, VIEW, SAVE)

    elif TRAINING == 8:

        measures = torch.zeros((ITERATIONS, 4, TIMESTEPS+1, len(stimuli_labels), 2))
        for i in range(ITERATIONS):
            print('Iteration #' + str(i + 1))
            torch.cuda.seed()

            cues, cue_labels = generate_cues()
            stimuli, stimuli_labels = generate_stimuli()
            data = np.concatenate((cues, stimuli), axis=0)
            labels = cue_labels + stimuli_labels
            subcortex, _ = train_subcortex(stimuli, stimuli_labels, BATCH_SIZE, LR, EPOCHS, device)
            cortex, _ = training(data, labels, BATCH_SIZE, LR, EPOCHS, device)
            cortex, _ = fine_tune_network(cues, cue_labels, stimuli, stimuli_labels, cortex, BATCH_SIZE, LR, EPOCHS * 2, device)

            network = ANNBrain(cortex, subcortex)

            cues, cue_labels = generate_cues()
            stimuli, stimuli_labels = generate_stimuli()
            # stimuli += np.abs(np.random.normal(loc=0.0, scale=0.2, size=stimuli.shape)) # Gaussian noise
            stimuli = torch.from_numpy(stimuli).float().to(device)
            cues = torch.from_numpy(cues).float().to(device)

            with torch.no_grad():
                (_, _, _, _, _, cortex_measures_pro), (_, _, subcortex_measures_pro), finals_pro = network.time_evolution(stimuli,
                                                                                                                          TIMESTEPS,
                                                                                                                          cues[0:1, :, :])
                (_, _, _, _, _, cortex_measures_anti), (_, _, subcortex_measures_anti), finals_anti = network.time_evolution(stimuli,
                                                                                                                             TIMESTEPS,
                                                                                                                             cues[-1:, :, :])

            measures[i, 0, :, :, :] = cortex_measures_pro
            measures[i, 1, :, :, :] = subcortex_measures_pro
            measures[i, 2, :, :, :] = cortex_measures_anti
            measures[i, 3, :, :, :] = subcortex_measures_anti

        torch.save(measures, 'data/time.pth')  # save measures in an external file

    elif TRAINING == 9:
        measures = torch.load('data/time.pth')

        plot_decision_evolution(measures[:, 0, :, :, :], measures[:, 1, :, :, :], measures[:, 2, :, :, :], measures[:, 3, :, :, :],
                                TIMESTEPS, DT, stimuli_labels, THRESHOLD, VIEW, SAVE)
        plot_rt_histogram(measures[:, 0, :, :, :], measures[:, 1, :, :, :], measures[:, 2, :, :, :],
                          DT, stimuli_labels, THRESHOLD, VIEW, SAVE)
        plot_rt_mixed(measures[:, 0, :, :, :], measures[:, 1, :, :, :], measures[:, 2, :, :, :],
                      DT, stimuli_labels, THRESHOLD, VIEW, SAVE)
        plot_decision_layer(measures[:, 0, :, :, :], measures[:, 1, :, :, :], measures[:, 2, :, :, :], measures[:, 3, :, :, :],
                            TIMESTEPS, DT, stimuli_labels, VIEW, SAVE)


def fine_tune_network(cues, cue_labels, stimuli, stimuli_labels, network, batch_size, lr, epochs, device):
    """Prepare the training of the motor circuit of the cortical network"""

    stimuli = np.concatenate((stimuli, np.flip(stimuli, axis=0)), axis=0)
    stimuli_labels.extend(stimuli_labels[::-1])

    # match the size of cues and stimuli by repeating
    cycler = itertools.cycle(cue_labels)
    cue_labels = [next(cycler) for _ in range(len(stimuli_labels))]
    cycler = itertools.cycle(cues)
    cues = np.array([next(cycler) for _ in range(stimuli.shape[0])])

    motor_labels = generate_motor_labels(cue_labels, stimuli_labels)

    return fine_tuning(network, cues, stimuli, motor_labels, batch_size, lr, epochs, device)


if __name__ == '__main__':
    main()
