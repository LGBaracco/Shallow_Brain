from PIL import Image
import numpy as np
from typing import List
import numpy.typing as npt


def generate_stimuli() -> (npt.NDArray, List):

    images = np.zeros((121, 16, 32))
    labels = []

    index = 0
    for i in range(11):
        for j in range(11):

            left = i / 10.0
            right = j / 10.0

            labels.append(0) if left > right else labels.append(1)

            images[index, 6:10, 5:9] = left
            images[index, 6:10, 23:27] = right

            index += 1

    #img = Image.fromarray(np.uint8(images[:,:,25] * 255))
    #img.show()

    return images, labels


def generate_cues() -> (npt.NDArray, List):
    images = np.zeros((16, 16, 32))
    labels = []

    # Vertical
    index = 0
    for i in range(4):
        for j in range(2):
            labels.append(2)
            images[index, 6+j:9+j, 14+i] = 1.0

            index += 1

    # Horizontal
    for i in range(4):
        for j in range(2):
            labels.append(3)
            images[index, 6 + i, 14 + j:17 + j] = 1.0

            index += 1

    #img = Image.fromarray(np.uint8(images[10,:,:] * 255))
    #img.show()

    return images, labels


def interleave(cues, cue_labels, stimuli, stimuli_labels) -> (npt.NDArray, List):

    images = np.zeros((137, 16, 32))
    labels = []

    for i in range(137):
        if i % 2 == 0:
            images[i, :, :] = cues[i//2 % 16, :, :]
            labels.append(cue_labels[i//2 % 16])
        else:
            images[i, :, :] = stimuli[i-16, :, :]
            labels.append(stimuli_labels[i-16])

    return images, labels
