import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch


def plot_accuracy(measures, view=False, save=True):

    means = np.mean(measures, axis=0)
    std_devs = np.std(measures, axis=0)

    labels = np.array(['Classification training accuracy', 'Classification test accuracy'])
    colors = ['blue', 'green']
    if measures.shape[1] == 4:
        labels = np.append(labels, 'Motor training accuracy')
        labels = np.append(labels, 'Motor test accuracy')
        colors[1] = 'blue'
        colors.extend(['green', 'green'])
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


def plot_evolution(activations, total_time, dt, view=False, save=True):

    time = np.arange(0, (total_time*dt)+dt, dt)

    plt.ylabel('Firing rate (a.u.)')
    plt.xlabel('Time (s)')
    plt.title('rate of sampled neuron from each subcortical layer over time')
    for i, r in enumerate(activations):
        if len(r.size()) == 5:
            plt.plot(time, r[:, 0, 0, 0, 0], label=f"r{i}")
            plt.title('activation of sampled neuron from each cortical layer over time')
        else:
            plt.plot(time, r[:, 0, 0], label=f"r{i}")
    plt.legend()
    plt.xlim(0, time[-1])

    if save:
        if len(activations) > 3:
            plt.savefig('Pictures/time/cortex_time_evolution')
        else:
            plt.savefig('Pictures/time/subcortex_time_evolution')
    if view:
        plt.show()
    plt.clf()


def plot_decision_evolution(cortex_measures_pro, subcortex_measures_pro, cortex_measures_anti, subcortex_measures_anti, total_time, dt, labels, view=False, save=True):

    time = np.arange(0, (total_time*dt)+dt, dt)
    labels = np.array(labels)

    # Left and prosaccade
    activations_cortex = cortex_measures_pro[:, np.where(labels == 0)[0], :]
    means_cortex = torch.mean(activations_cortex, dim=1)
    std_cortex = torch.std(activations_cortex, dim=1)
    activations_subcortex = subcortex_measures_pro[:, np.where(labels == 0)[0], :]
    means_subcortex = torch.mean(activations_subcortex, dim=1)
    std_subcortex = torch.std(activations_subcortex, dim=1)

    plt.ylabel('Firing rate (a.u.)')
    plt.xlabel('Time (s)')
    plt.title('Mean rates: left brighter, prosaccade')
    #_, decision = torch.max(activations[-1, :, :], 1)
    # plt.title(f'L:{result[0,0]:.2f}, R:{result[0,1]:.2f}, D:{decision}')
    plt.plot(time, means_cortex[:, 0], label="L cortex rate", color='red')
    plt.fill_between(time, means_cortex[:, 0]-std_cortex[:, 0], means_cortex[:, 0]+std_cortex[:, 0], alpha=0.3, color='red')
    plt.plot(time, means_cortex[:, 1], label="R cortex rate", color='orange')
    plt.fill_between(time, means_cortex[:, 1]-std_cortex[:, 1], means_cortex[:, 1]+std_cortex[:, 1], alpha=0.3, color='orange')
    plt.plot(time, means_subcortex[:, 0], label="L subcortex rate", color='blue')
    plt.fill_between(time, means_subcortex[:, 0]-std_subcortex[:, 0], means_subcortex[:, 0]+std_subcortex[:, 0], alpha=0.3, color='blue')
    plt.plot(time, means_subcortex[:, 1], label="R subcortex rate", color='cyan')
    plt.fill_between(time, means_subcortex[:, 1]-std_subcortex[:, 1], means_subcortex[:, 1]+std_subcortex[:, 1], alpha=0.3, color='cyan')

    plt.legend()
    plt.xlim(0, time[-1])

    if save:
        plt.savefig('Pictures/time/prosaccade_left_time_evolution')
    if view:
        plt.show()

    plt.clf()

    # Right and prosaccade
    activations_cortex = cortex_measures_pro[:, np.where(labels == 1)[0], :]
    means_cortex = torch.mean(activations_cortex, dim=1)
    std_cortex = torch.std(activations_cortex, dim=1)
    activations_subcortex = subcortex_measures_pro[:, np.where(labels == 1)[0], :]
    means_subcortex = torch.mean(activations_subcortex, dim=1)
    std_subcortex = torch.std(activations_subcortex, dim=1)

    plt.ylabel('Firing rate (a.u.)')
    plt.xlabel('Time (s)')
    plt.title('Mean rates: right brighter, prosaccade')
    plt.plot(time, means_cortex[:, 0], label="L cortex rate", color='red')
    plt.fill_between(time, means_cortex[:, 0]-std_cortex[:, 0], means_cortex[:, 0]+std_cortex[:, 0], alpha=0.3, color='red')
    plt.plot(time, means_cortex[:, 1], label="R cortex rate", color='orange')
    plt.fill_between(time, means_cortex[:, 1]-std_cortex[:, 1], means_cortex[:, 1]+std_cortex[:, 1], alpha=0.3, color='orange')
    plt.plot(time, means_subcortex[:, 0], label="L subcortex rate", color='blue')
    plt.fill_between(time, means_subcortex[:, 0]-std_subcortex[:, 0], means_subcortex[:, 0]+std_subcortex[:, 0], alpha=0.3, color='blue')
    plt.plot(time, means_subcortex[:, 1], label="R subcortex rate", color='cyan')
    plt.fill_between(time, means_subcortex[:, 1]-std_subcortex[:, 1], means_subcortex[:, 1]+std_subcortex[:, 1], alpha=0.3, color='cyan')

    plt.legend()
    plt.xlim(0, time[-1])

    if save:
        plt.savefig('Pictures/time/prosaccade_right_time_evolution')
    if view:
        plt.show()

    plt.clf()

    # Left and antisaccade
    activations_cortex = cortex_measures_anti[:, np.where(labels == 0)[0], :]
    means_cortex = torch.mean(activations_cortex, dim=1)
    std_cortex = torch.std(activations_cortex, dim=1)
    activations_subcortex = subcortex_measures_anti[:, np.where(labels == 0)[0], :]
    means_subcortex = torch.mean(activations_subcortex, dim=1)
    std_subcortex = torch.std(activations_subcortex, dim=1)

    plt.ylabel('Firing rate (a.u.)')
    plt.xlabel('Time (s)')
    plt.title('Mean rates: left brighter, antisaccade')
    plt.plot(time, means_cortex[:, 0], label="L cortex rate", color='red')
    plt.fill_between(time, means_cortex[:, 0]-std_cortex[:, 0], means_cortex[:, 0]+std_cortex[:, 0], alpha=0.3, color='red')
    plt.plot(time, means_cortex[:, 1], label="R cortex rate", color='orange')
    plt.fill_between(time, means_cortex[:, 1]-std_cortex[:, 1], means_cortex[:, 1]+std_cortex[:, 1], alpha=0.3, color='orange')
    plt.plot(time, means_subcortex[:, 0], label="L subcortex rate", color='blue')
    plt.fill_between(time, means_subcortex[:, 0]-std_subcortex[:, 0], means_subcortex[:, 0]+std_subcortex[:, 0], alpha=0.3, color='blue')
    plt.plot(time, means_subcortex[:, 1], label="R subcortex rate", color='cyan')
    plt.fill_between(time, means_subcortex[:, 1]-std_subcortex[:, 1], means_subcortex[:, 1]+std_subcortex[:, 1], alpha=0.3, color='cyan')

    plt.legend()
    plt.xlim(0, time[-1])

    if save:
        plt.savefig('Pictures/time/antisaccade_left_time_evolution')
    if view:
        plt.show()

    plt.clf()

    # Right and antisaccade
    activations_cortex = cortex_measures_anti[:, np.where(labels == 1)[0], :]
    means_cortex = torch.mean(activations_cortex, dim=1)
    activations_subcortex = subcortex_measures_anti[:, np.where(labels == 1)[0], :]
    means_subcortex = torch.mean(activations_subcortex, dim=1)

    plt.ylabel('Firing rate (a.u.)')
    plt.xlabel('Time (s)')
    plt.title('Mean rates: right brighter, antisaccade')
    plt.plot(time, means_cortex[:, 0], label="L cortex rate", color='red')
    plt.fill_between(time, means_cortex[:, 0]-std_cortex[:, 0], means_cortex[:, 0]+std_cortex[:, 0], alpha=0.3, color='red')
    plt.plot(time, means_cortex[:, 1], label="R cortex rate", color='orange')
    plt.fill_between(time, means_cortex[:, 1]-std_cortex[:, 1], means_cortex[:, 1]+std_cortex[:, 1], alpha=0.3, color='orange')
    plt.plot(time, means_subcortex[:, 0], label="L subcortex rate", color='blue')
    plt.fill_between(time, means_subcortex[:, 0]-std_subcortex[:, 0], means_subcortex[:, 0]+std_subcortex[:, 0], alpha=0.3, color='blue')
    plt.plot(time, means_subcortex[:, 1], label="R subcortex rate", color='cyan')
    plt.fill_between(time, means_subcortex[:, 1]-std_subcortex[:, 1], means_subcortex[:, 1]+std_subcortex[:, 1], alpha=0.3, color='cyan')

    plt.legend()
    plt.xlim(0, time[-1])

    if save:
        plt.savefig('Pictures/time/antisaccade_right_time_evolution')
    if view:
        plt.show()

    plt.clf()
