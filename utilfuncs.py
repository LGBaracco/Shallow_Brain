import torch


def stimuli_extractor(batch):
    """Extracts stimuli from batches of inputs"""

    stimuli = torch.zeros(len(batch), 2)
    stimuli[:, 0] = batch[:, 7, 6]  # left
    stimuli[:, 1] = batch[:, 7, 25]  # right

    return stimuli
