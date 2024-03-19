from PIL import Image
import random
import numpy as np
from typing import List
import numpy.typing as npt
import matplotlib.pyplot as plt


def generate_stimuli() -> (npt.NDArray, List):

    images = np.zeros((110, 16, 32))
    labels = []

    index = 0
    for i in range(11):
        for j in range(11):

            left = i / 10.0
            right = j / 10.0

            if left == right:
                continue
            elif left > right:
                labels.append(0)
            else:
                labels.append(1)

            images[index, 6:10, 5:9] = left
            images[index, 6:10, 23:27] = right

            index += 1

    # img = Image.fromarray(np.uint8(images[:,:,25] * 255))
    # img.show()

    return images, labels


def generate_cues() -> (npt.NDArray, List):
    images = np.zeros((16, 16, 32))
    labels = []

    # Vertical (Ps)
    index = 0
    for i in range(4):
        for j in range(2):
            labels.append(2)
            images[index, 6+j:9+j, 14+i] = 1.0

            index += 1

    # Horizontal (As)
    for i in range(4):
        for j in range(2):
            labels.append(3)
            images[index, 6 + i, 14 + j:17 + j] = 1.0

            index += 1

    return images, labels


def interleave(cues, cue_labels, stimuli, stimuli_labels) -> (npt.NDArray, List):
    """Might be useless: interleave cues with stimuli. maybe useful for final testing"""

    images = np.zeros((126, 16, 32))
    labels = []

    for i in range(137):
        if i % 2 == 0:  # cues
            images[i, :, :] = cues[i//2 % 16, :, :]
            labels.append(cue_labels[i//2 % 16])
        else:  # stimuli
            images[i, :, :] = stimuli[i-16, :, :]
            labels.append(stimuli_labels[i-16])

    return images, labels


def generate_motor_labels(cue_labels, stimuli_labels) -> List:  # TODO: move this into utilfuncs

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


def generate_test_set() -> (npt.NDArray, List):
    """Generate test set using stimuli with 2 decimal positions (training set only uses 1 decimal position)"""

    stimuli = np.zeros((28, 16, 32))
    labels = []

    for i in range(28):  # arbitrary number

        a = round(random.uniform(0, 1), 2)
        b = round(random.uniform(0, 1), 2)

        if a == b:
            generate_test_set()  # restart generation in case stimuli are equal

        if i % 2 == 0:

            stimuli[i, 6:10, 5:9] = max(a, b)
            stimuli[i, 6:10, 23:27] = min(a, b)

            labels.append(0)
        else:
            stimuli[i, 6:10, 5:9] = min(a, b)
            stimuli[i, 6:10, 23:27] = max(a, b)

            labels.append(1)

    return stimuli, labels


def generate_sanity_check() -> (np.ndarray, np.ndarray):
    cues, _ = generate_cues()
    cues = np.array([cues[3, :, :], cues[-3, :, :]])
    stimuli, _ = generate_stimuli()
    stimuli = np.array([stimuli[15, :, :], stimuli[-17, :, :]])

    return cues, stimuli


def generate_vague() -> (np.ndarray, np.ndarray):
    """Generate ambiguous cues: 45° slanted bars in all four directions
    Generate ambiguous stimuli: both boxes of perfectly equal brightness"""
    stimuli = np.zeros((11, 16, 32))

    cues = np.zeros((4, 16, 32))

    cues[0, 6, 14] = 1.0
    cues[0, 7, 15] = 1.0
    cues[0, 8, 16] = 1.0

    cues[1, 6, 16] = 1.0
    cues[1, 7, 15] = 1.0
    cues[1, 8, 14] = 1.0

    cues[2, 8, 14] = 1.0
    cues[2, 7, 15] = 1.0
    cues[2, 6, 16] = 1.0

    cues[3, 8, 16] = 1.0
    cues[3, 7, 15] = 1.0
    cues[3, 6, 14] = 1.0

    # plt.imshow(cues[1, :, :], cmap='gray')
    # plt.show()

    for i in range(11):
        stimuli[i, 6:10, 5:9] = i/10.0
        stimuli[i, 6:10, 23:27] = i/10.0

    '''for i, cue in enumerate(cues):
        plt.imshow(cue, cmap='gray')
        filename = 'Pictures/vague_stimuli_cues/cue_' + str(i)
        plt.savefig(filename)
        plt.clf()
    for i, stimulus in enumerate(stimuli):
        plt.imshow(stimulus, cmap='gray')
        filename = 'Pictures/vague_stimuli_cues/stimulus_' + str(i)
        plt.savefig(filename)
        plt.clf()'''

    return cues, stimuli
