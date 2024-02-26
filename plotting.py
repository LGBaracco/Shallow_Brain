import numpy as np
import matplotlib.pyplot as plt


def plot_accuracy(measures, view=False, save=True):

    means = np.mean(measures, axis=0)
    std_devs = np.std(measures, axis=0)

    labels = np.array(['Classification training accuracy', 'Motor training accuracy',
                       'Classification test accuracy', 'Motor test accuracy'])  # 'Prosaccade ratio (slanted cues)', 'Left ratio (equal brightness)'

    colors = ['blue', 'blue', 'green', 'green']

    plt.bar(labels, means, yerr=std_devs, align='center', alpha=0.5, ecolor='black', capsize=10, color=colors)
    plt.ylim(bottom=0, top=105)
    plt.ylabel('Value')
    plt.xlabel('Measure')
    plt.title('Mean and Standard Deviation of accuracies')
    plt.xticks(labels, rotation=45, ha='right')
    plt.tight_layout()

    if save:
        plt.savefig('Pictures/accuracy')
    if view:
        plt.show()

    plt.clf()


def plot_ratio(measures, view=False, save=True):

    means = np.mean(measures, axis=0)
    std_devs = np.std(measures, axis=0)

    labels = np.array(['Prosaccade ratio (slanted cues)', 'Left ratio (equal brightness)'])  #

    plt.bar(labels, means, yerr=std_devs, align='center', ecolor='black', alpha=0.5, capsize=10)
    plt.ylim(bottom=0, top=100)
    plt.ylabel('Value')
    plt.xlabel('Measure')
    plt.title('Mean and Standard Deviation of ratios')
    #plt.xticks(labels, rotation=45, ha='right')
    plt.tight_layout()

    if save:
        plt.savefig('Pictures/vague_ratio')
    if view:
        plt.show()

    plt.clf()


def plot_equal_brightness(measures, view=False, save=True):

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
        plt.savefig('Pictures/ratio_equal_brightness')
    if view:
        plt.show()

    plt.clf()


