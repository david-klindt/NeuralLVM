import torch
import matplotlib.pyplot as plt
import numpy as np
from sklearn.feature_selection import mutual_info_regression
from model import *


def torch_circular_gp(num_sample, num_dim, smoothness):
    z = torch.randn(num_sample, num_dim) / smoothness
    z = torch.cumsum(z, 0)
    return vector2angle(angle2vector(z))


def analysis(ensembler, model, trainer, z_test, do_inference=False):
    num_ensemble = ensembler.num_ensemble
    latent_dim = ensembler.latent_dim
    _, y_, z_, mu, logvar = ensembler(trainer.data_test[trainer.neurons_train_ind])

    if do_inference:
        z_ = inference(ensembler,
            trainer.data_test[trainer.neurons_train_ind],
            trainer.data_test[trainer.neurons_test_ind],
        )
        _, y_, _, mu, logvar = ensembler(
            trainer.data_test[trainer.neurons_train_ind], z=z_)

    z_ = z_.view(z_test.shape)
    logvar = logvar.view(z_test.shape)

    # plot ensemble_weights
    ensemble_weights = torch.nn.functional.softmax(
        ensembler.ensemble_weights_test, dim=1).detach().cpu().numpy()
    plt.plot(ensemble_weights)
    plt.legend(np.arange(num_ensemble))
    plt.title('Ensemble Weights')
    plt.xlabel('Neurons')
    plt.ylabel('Weights')
    plt.tight_layout()
    plt.show()

    # all latent comparisons
    vars = torch.exp(logvar).detach().cpu().numpy()
    plt.figure(figsize=(20, 20))
    for i in range(latent_dim * num_ensemble):
        for j in range(latent_dim * num_ensemble):
            plt.subplot(latent_dim * num_ensemble, latent_dim * num_ensemble,
                        i * latent_dim * num_ensemble + j + 1)
            x = z_test[:, i].detach().cpu().numpy()
            y = z_[:, j].detach().cpu().numpy()
            plt.scatter(x, y, c=vars[:, j], s=1)
            plt.title('MI=%.4f' % mutual_info_regression(x[:, None], y))
            plt.colorbar(label='variance')
            plt.xlabel('true z%s' % i)
            plt.ylabel('pred z%s' % j)
    plt.tight_layout()
    plt.show()

    # rate predictions
    plt.figure(figsize=(12, 12))
    for i in range(latent_dim * num_ensemble):
        plt.subplot(num_ensemble, latent_dim, i + 1)
        plt.plot(trainer.data_test[trainer.neurons_test_ind][i * 25, :200].detach().cpu().numpy())
        plt.plot(y_[i * 25, :200].detach().cpu().numpy())
        plt.legend(['true', 'predicted'])
        plt.title('Responses (test neurons)')
    plt.tight_layout()
    plt.show()

    # RF comparisons
    plt.figure(figsize=(18, 6))
    true_rfs = model.receptive_fields[trainer.neurons_test_ind].detach().cpu().numpy()
    learned_rfs = ensembler.receptive_fields_test.detach().cpu().numpy().reshape(
        true_rfs.shape[0], -1
    )
    for i in range(true_rfs.shape[1]):
        for j in range(learned_rfs.shape[1]):
            plt.subplot(
                true_rfs.shape[1],
                learned_rfs.shape[1],
                i * learned_rfs.shape[1] + j + 1
            )
            plt.scatter(
                true_rfs[:, i],
                np.mod(learned_rfs[:, j], 2 * np.pi),
                c=ensemble_weights.argmax(1)
            )
            plt.colorbar(label='ensembles')
            plt.xlabel('true')
            plt.ylabel('pred')
            plt.title('Receptive fields')
    plt.tight_layout()
    plt.show()
