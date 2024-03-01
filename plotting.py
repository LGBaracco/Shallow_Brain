import numpy as np
import matplotlib.pyplot as plt


def plot_accuracy(measures, view=False, save=True):

    means = np.mean(measures, axis=0)
    std_devs = np.std(measures, axis=0)

    labels = np.array(['Classification training accuracy', 'Motor training accuracy'])  # 'Prosaccade ratio (slanted cues)', 'Left ratio (equal brightness)'
    colors = ['blue', 'blue']
    if measures.shape[1] == 3:
        labels = np.append(labels, 'Classification test accuracy')
        colors = np.append(colors, 'green')
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
    plt.xlabel('Measure')
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
    # plt.xticks(labels, rotation=45, ha='right')
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


