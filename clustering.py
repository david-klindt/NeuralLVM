import numpy as np
from utils import torch_circular_gp
from data import data_generation
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.feature_selection import mutual_info_regression
import torch
from itertools import permutations



def get_data(
        num_neuron_train,
        num_neuron_test,
        len_data_train,
        len_data_test,
        index,
        global_seed,
        num_ensemble,
        dim
):
    """Just a faster, multi-ensemble version for the ensemble benchmark."""
    num_neuron = num_neuron_train + num_neuron_test
    data = data_generation(
        len_data=len_data_train + len_data_test,
        dim=dim,
        num_neuron=num_neuron,
        poisson_noise=True,
        bump_placement='random',
        seed=global_seed + index,
        num_ensemble=num_ensemble,
        periodicity=True,
        kernel_type='exp',
        kernel_sdev=5.0,
        kernel_scale=50.0,
        peak_firing=0.5,
        back_firing=0.005,
        tuning_width=1.2,
        snr_scale=1.0,
    )

    #print("Generating latents\n")
    # very slow, used in other experiments:
    # z = data.generate_z()
    # iid, gives better results, but unrealistic
    # z = np.random.uniform(0, 2 * np.pi, (len_data_train + len_data_test, dim * num_ensemble))
    # fast GP:
    z_smoothness = 3
    torch.manual_seed(global_seed + index)
    z = torch_circular_gp(len_data_train + len_data_test, dim * num_ensemble, z_smoothness).numpy()
    z_train = z[:len_data_train, :]
    z_test = z[len_data_train:, :]

    #print("Generating receptive fields\n")
    rf = data.generate_receptive_fields(z)

    #print("Generating spikes")
    y_train = data.generate_spikes(z_train, rf)
    data.poisson_noise = False
    y_test = data.generate_spikes(z_test, rf)

    # select training and test neurons
    np.random.seed(global_seed + index)
    neurons_train_ind = np.zeros(num_neuron * num_ensemble, dtype=bool)
    for i in range(num_ensemble):
        ind = np.random.choice(num_neuron, num_neuron_train, replace=False)
        neurons_train_ind[ind + i * num_neuron] = True

    return y_train, z_train, y_test, z_test, rf, neurons_train_ind


def cluster(data, method, n_clusters, latents=None, dim=8, seed=235798):
    """data matrix is neurons x time"""
    if method == 'raw_pca':
        u, _, _ = np.linalg.svd(data)
        features = abs(u[:, :dim])
    elif method.startswith('cov'):
        covariance = np.cov(data)
        if method == 'cov_pca':
            u, _, _ = np.linalg.svd(covariance)
            features = abs(u[:, :dim])
        elif method == 'cov_agg':
            clustering = AgglomerativeClustering(n_clusters=n_clusters).fit(covariance)
            return clustering.labels_
    elif method == 'supervised':
        num_neuron = data.shape[0]
        num_latent = latents.shape[1]
        importance = np.zeros((num_neuron, num_latent))
        for i in range(num_neuron):
            for j in range(num_latent):
                importance[i, j] = mutual_info_regression(latents[:, j][:, None], data[i])
        features = importance
    else:
        raise ValueError("Clustering method not defined.")

    kmeans = KMeans(n_clusters=n_clusters, random_state=seed, n_init=100, max_iter=1000).fit(features)
    return kmeans.labels_


def get_accuracy(label, label_):
    """Find best permutation of cluster labels to true labels."""
    accs = []
    for perm in permutations(np.unique(label_)):
        label__ = label_.copy()
        for i, p in enumerate(perm):
            label__[label_ == i] = p
        accs.append(np.mean(label == label__))
    return np.max(accs)