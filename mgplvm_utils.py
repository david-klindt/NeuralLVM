import matplotlib.pyplot as plt
import numpy as np

def add_suffix(rep, num_neuron_train, len_data_train, s):
    return f"{s}_{rep}_N{num_neuron_train}_T{len_data_train}"


def eval_lats(pred, target):
    errs = np.zeros((2, 1000))
    for i in range(2):
        for i_s, s in enumerate(np.linspace(0, 2 * np.pi, 1000)):
            newpred = 2 * (0.5 - i) * pred + s
            errs[i, i_s] = np.mean(np.arccos(np.cos(newpred - target)))
    return np.amin(errs)

def plot_data(y_train, len_data_train, num_neuron_tot, fname):
    plt.figure()
    plt.imshow(y_train, cmap="Greys", aspect="auto")
    plt.xlabel("time")
    plt.ylabel("neuron")
    plt.xlim(0, len_data_train)
    plt.ylim(0, num_neuron_tot)
    plt.xticks([])
    plt.yticks([])
    plt.savefig(fname)

def cb(mod, i, loss):
    """here we construct an (optional) function that helps us keep track of the training"""
    if i in [0, 50, 100, 150, 300, 500, 1000, 1500,
                1999]:  #iterations to plot
        X = mod.lat_dist.prms[0].detach().cpu().numpy()[0, ...]
        X = X % (2 * np.pi)
        plt.figure(figsize=(4, 4))
        plt.xlim(0, 2 * np.pi)
        plt.ylim(0, 2 * np.pi)
        plt.xlabel("true latents")
        plt.ylabel("model latents")
        plt.scatter(z_tot[:, 0], X[:, 0], color='k')
        plt.title('iter = ' + str(i))
        plt.savefig(
            suffix(f"{FLAGS.results_dir}/latents{i}") + ".png")


