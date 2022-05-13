import torch
import matplotlib.pyplot as plt
import numpy as np
from sklearn.feature_selection import mutual_info_regression
from NeuralLVM.code.inference import inference


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def angle2vector(angle):
    return torch.stack([torch.sin(angle), torch.cos(angle)], -1)


def vector2angle(vector):
    return torch.atan2(vector[..., 0], vector[..., 1])


def angle2vector_flat(angle):
    vector = []
    for i in range(angle.shape[1]):
        vector.append(torch.sin(angle[:, i]))
        vector.append(torch.cos(angle[:, i]))
    return torch.stack(vector, 1)


def vector2angle_flat(vector):
    angle = []
    for i in range(vector.shape[1] // 2):
        angle.append(torch.atan2(vector[:, i * 2], vector[:, i * 2 + 1]))
    return torch.stack(angle, 1)


def sum_pairs(x):
    sum = []
    for i in range(x.shape[1] // 2):
        sum.append(torch.sum(x[:, i * 2:(i + 1) * 2], dim=1))
    return torch.stack(sum, 1)


def compute_kld_to_normal(mu, logvar):
    """Computes the KL(q|p) between variational posterior q and standard
    normal p."""
    return torch.mean(-0.5 * (1 + logvar - mu ** 2 - logvar.exp()))


def compute_kld(q1_mu, q1_logvar, q0_mu, q0_logvar):
    """Computes the KL(q_t|q_{t-1}) between variational posterior q_t ("q1")
    and variational posterior q_{t-1} ("q0")."""
    KL = (q0_logvar - q1_logvar) / 2
    KL = KL + (q1_logvar.exp() + (q0_mu - q1_mu) ** 2) / (2 * q0_logvar.exp())
    KL = torch.mean(KL - 1 / 2)
    return KL


def compute_slowness_loss(mu):
    """compute squared difference over 2nd dimension, i.e., time."""
    return torch.mean((mu[:, 1:] - mu[:, :-1]) ** 2)


def compute_poisson_loss(y, y_):
    return torch.mean(y_ - y * torch.log(y_ + 1e-9))


def torch_normalize(y):
    normed = y - torch.mean(y, dim=1, keepdim=True)
    norm = torch.linalg.norm(normed, dim=1, keepdim=True) + 1e-6
    return normed / norm


def correlation_loss(y, y_):
    return torch.mean(
        torch.sum(torch_normalize(y) * torch_normalize(y_), dim=1))


def check_grad(model, log_file):
    for name, param in model.named_parameters():
        if param.grad is not None:
            if torch.any(torch.isnan(param.grad)):
                print('NaN in Gradient, skipping step', name, file=log_file)
                return False
            if torch.any(torch.isinf(param.grad)):
                print('inf in Gradient, skipping step', name, file=log_file)
                return False
    return True


def torch_circular_gp(num_sample, num_dim, smoothness):
    z = torch.randn(num_sample, num_dim) / smoothness
    z = torch.cumsum(z, 0)
    return vector2angle(angle2vector(z))


def get_grid(dim, num):
    """makes evenly sampled grid [-pi, pi] of num points in each dim."""
    single = np.linspace(-np.pi, np.pi, num)
    out = np.meshgrid(*[single] * dim)
    return np.stack(out, -1).reshape(num ** dim, dim)


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
