import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch


def plot_accuracy(measures, view=False, save=True):

    means = np.mean(measures, axis=0)
    std_devs = np.std(measures, axis=0)

    labels = np.array(['Classification training accuracy', 'Classification test accuracy'])  # 'Prosaccade ratio (slanted cues)', 'Left ratio (equal brightness)'
    colors = ['blue', 'green']
    if measures.shape[1] == 3:
        labels = np.append(labels, 'Motor training accuracy')
        colors[1] = 'blue'
        colors.append('green')
        filename = 'Pictures/accuracy_cortex'
    else:
        filename = 'Pictures/accuracy_subcortex'

    plt.bar(labels, means, yerr=std_devs, align='center', alpha=0.5, ecolor='black', capsize=10, color=colors)
    plt.ylim(bottom=0, top=105)
    plt.ylabel('Value')
    plt.xlabel('Measure')
    plt.title('Mean and Standard Deviation of accuracies')
    plt.xticks(labels, rotation=45, ha='right')
    plt.tight_layout()

    if save:
        plt.savefig(filename)
    if view:
        plt.show()

    plt.clf()


def plot_ratio(measures, view=False, save=True):

    means = np.mean(measures, axis=0)
    std_devs = np.std(measures, axis=0)

    labels = np.array(['Left ratio (equal brightness)'])

    if measures.shape[1] == 2:
        labels = np.insert(labels, 0, 'Prosaccade ratio (slanted cues)')
        filename = 'Pictures/vague_ratio_cortex'
    else:
        filename = 'Pictures/vague_ratio_subcortex'

    plt.bar(labels, means, yerr=std_devs, align='center', ecolor='black', alpha=0.5, capsize=10)
    plt.ylim(bottom=0, top=100)
    plt.ylabel('Value')
    plt.xlabel('Metric')
    plt.title('Mean and Standard Deviation of ratios')
    #plt.xticks(labels, rotation=45, ha='right')
    plt.tight_layout()

    if save:
        plt.savefig(filename)
    if view:
        plt.show()

    plt.clf()


def plot_equal_brightness(measures, view=False, save=True, subcortex=False):

    values = np.sum(measures == 0, axis=0)
    values = 100 * values / measures.shape[0]

    plt.bar(np.arange(measures.shape[1]), values, align='center', alpha=0.5)
    plt.ylim(bottom=0, top=100)
    plt.ylabel('Ratio')
    plt.xlabel('Brightness')
    plt.title('left ratios per brightness value')
    plt.tight_layout()

    if save:
        if subcortex:
            filename = 'Pictures/ratio_equal_brightness_subcortex'
        else:
            filename = 'Pictures/ratio_equal_brightness_cortex'

        plt.savefig(filename)
    if view:
        plt.show()

    plt.clf()


def plot_brain(measures, view=False, save=True):

    labels = np.array(['Accuracy', 'Subcortex ratio (prosaccade)'])
    colors = ['blue', 'green']

    plt.bar(labels, measures, align='center', alpha=0.5, color=colors)
    plt.ylim(bottom=0, top=100)
    plt.ylabel('Value')
    plt.xlabel('Metric')
    plt.title('Accuracy and subcortex use ratio of full brain')
    plt.tight_layout()

    if save:
        plt.savefig('Pictures/full_brain_measures')
    if view:
        plt.show()

    plt.clf()


def plot_heatmap(measures, view=False, save=True):

    measures[measures == 0] = measures.min()
    measures -= measures.min()
    measures = (measures - measures.min()) / (measures.max() - measures.min())

    pss = measures[:110, :, :]
    ass = measures[-110:, :, :]
    values = np.concatenate([pss, ass])

    saccade_tick = [110]
    right_tick = np.arange(0, 220, 10)
    label_tick = right_tick / 100.0
    label_tick[11:] = label_tick[:11]
    label_tick = list(map(lambda e: "L brightness: {:.1f}".format(e), label_tick))

    # Plot 1
    difference = values[:, :, 0] - values[:, :, 1]
    fig, axs = plt.subplots(1, 1)
    im = axs.imshow(difference, interpolation="none", cmap='gray_r')
    axs.set_aspect('auto')
    plt.ylabel('Stimuli')
    plt.xlabel('Time')
    plt.title('Difference of left and right activation in cortex across stimuli over time')
    plt.xticks([5.5], ['Decision threshold'])
    plt.yticks(saccade_tick, ['saccade type cut-off'])
    plt.yticks(right_tick, label_tick, minor=True)
    plt.grid(axis='y', which='minor', linestyle='--', linewidth=1, color='blue', alpha=0.5)
    plt.grid(axis='y', which='major', linestyle='-', linewidth=2, color='red', alpha=0.7)
    plt.grid(axis='x', which='major', linestyle='-', linewidth=2, color='green', alpha=0.7)
    plt.tight_layout()
    fig.colorbar(im)

    if save:
        plt.savefig('Pictures/Cortex_heatmap')
    if view:
        fig.show()

    plt.clf()

    # Plot 2
    difference_sub = values[:, :, 2] - values[:, :, 3]
    fig, axs = plt.subplots(1, 1)
    im = axs.imshow(difference_sub, interpolation="none", cmap='gray_r')
    plt.ylabel('Stimuli')
    plt.xlabel('Time')
    plt.title('Difference of left and right activation in subcortex across stimuli over time')
    axs.set_aspect('auto')
    plt.xticks([1.5], ['Decision threshold'])
    plt.yticks(saccade_tick, ['saccade type cut-off'])
    plt.yticks(right_tick, label_tick, minor=True)
    plt.grid(axis='y', which='minor', linestyle='--', linewidth=1, color='blue', alpha=0.5)
    plt.grid(axis='y', which='major', linestyle='-', linewidth=2, color='red', alpha=0.7)
    plt.grid(axis='x', which='major', linestyle='-', linewidth=2, color='green', alpha=0.7)
    plt.tight_layout()
    fig.colorbar(im)

    if save:
        plt.savefig('Pictures/Subcortex_heatmap')
    if view:
        fig.show()

    plt.clf()

    # Plot 3
    full = difference + difference_sub
    fig, axs = plt.subplots(1, 1)
    im = axs.imshow(full, interpolation="none", cmap='gray_r')
    plt.ylabel('Stimuli')
    plt.xlabel('Time')
    plt.title('Difference of left and right activation across stimuli over time')
    axs.set_aspect('auto')
    plt.xticks([1.5, 5.5], ['Subcortex reaction time', 'Cortex reaction time'])
    plt.yticks(saccade_tick, ['saccade type cut-off'])
    plt.yticks(right_tick, label_tick, minor=True)
    plt.grid(axis='y', which='minor', linestyle='--', linewidth=1, color='blue', alpha=0.5)
    plt.grid(axis='y', which='major', linestyle='-', linewidth=2, color='red', alpha=0.7)
    plt.grid(axis='x', which='major', linestyle='-', linewidth=2, color='green', alpha=0.7)
    plt.tight_layout()
    fig.colorbar(im)

    if save:
        plt.savefig('Pictures/Full_heatmap')
    if view:
        fig.show()

    plt.clf()


def plot_continuous_subcortex(r0, r1, r2, total_time, dt, view=False, save=True):

    time = np.arange(0, total_time*dt, dt)

    plt.ylabel('Activation')
    plt.xlabel('Time (s)')
    plt.title('Difference of left and right activation across stimuli over time')
    plt.plot(time, r0[:, 0, 0], label='r0')
    plt.plot(time, r1[:, 0, 0], label='r1')
    plt.plot(time, r2[:, 0, 0], label='r2')
    plt.legend()
    plt.show()
