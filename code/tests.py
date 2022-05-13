import torch
import numpy as np
from NeuralLVM.code.utils import angle2vector_flat
from NeuralLVM.code.utils import sum_pairs
from NeuralLVM.code.utils import count_parameters
from NeuralLVM.code.utils import torch_circular_gp
from NeuralLVM.code.utils import analysis
from NeuralLVM.code.model import Model
from NeuralLVM.code.training import Trainer

device = "cuda" if torch.cuda.is_available() else "cpu"


### For first simulation experiments ###
class StochasticNeurons(torch.nn.Module):
    def __init__(
            self,
            N,
            num_ensemble=2,
            latent_dim=2,
            seed=304857,
            noise=False,
            tuning_width=10.0,
            scale=16.0,
    ):
        super(StochasticNeurons, self).__init__()
        self.num_ensemble = num_ensemble
        self.tuning_width = tuning_width
        self.scale = scale
        self.noise = noise
        self.latent_dim = latent_dim

        torch.manual_seed(seed)
        self.receptive_fields = torch.nn.Parameter(
            torch.rand(num_ensemble * N, latent_dim) * 2 * np.pi,
            requires_grad=False
        )
        ensemble_weights = np.zeros((N * num_ensemble, num_ensemble))
        for i in range(num_ensemble):
            ensemble_weights[i*N:(i+1)*N, i] = 1
        self.ensemble_weights = torch.nn.Parameter(
            torch.tensor(ensemble_weights, dtype=torch.float),
            requires_grad=False
        )
        selector = torch.stack([torch.eye(2 * latent_dim) for i in range(num_ensemble)], 0)
        self.selector = torch.nn.Parameter(selector, requires_grad=False)

    def forward(self, z):
        z_vector = angle2vector_flat(z)
        rf_vector = angle2vector_flat(self.receptive_fields)

        # early selection
        selector = self.ensemble_weights[..., None, None] * self.selector[None]
        selector = torch.concat(torch.split(selector, 1, dim=1), 3).view(
            -1, 2 * self.latent_dim, self.num_ensemble * 2 * self.latent_dim)
        selected = torch.matmul(selector, z_vector.T)
        dist = (rf_vector[..., None] - selected)**2
        pairs = sum_pairs(dist)
        if self.latent_dim == 2:
            pairs = sum_pairs(pairs)
        response = torch.exp(-pairs / self.tuning_width) * self.scale
        responses = response[:, 0]
        if self.noise:
            responses = torch.poisson(responses)
        responses = responses / self.scale

        return responses


def test_training(num_ensemble=2, num_neuron_train=50, num_neuron_test=50,
                  latent_dim=2, z_smoothness=3, num_basis=1,
                  num_sample=10000, num_test=1000):
    num_neuron = num_neuron_train + num_neuron_test
    neurons_train_ind = np.zeros(num_neuron * num_ensemble, dtype=bool)
    ind = np.random.choice(
        num_neuron * num_ensemble,
        num_neuron_train * num_ensemble,
        replace=False
    )
    neurons_train_ind[ind] = True
    simulator = StochasticNeurons(
        num_neuron, num_ensemble=num_ensemble, noise=True, latent_dim=latent_dim).to(device)
    ensembler = Model(
        num_neuron_train=num_neuron_train * num_ensemble,
        num_neuron_test=num_neuron_test * num_ensemble,
        kernel_size=9,
        num_hidden=256,
        latent_manifolds=('T2', 'T2'),
        feature_type=('gauss', 'gauss'),
        shared=(True, True),
        learn_coeff=(True, True),
        learn_mean=(False, False),
        learn_var=(True, True),
        isotropic=(True, True),
        num_basis=(num_basis, num_basis),
        seed=1293842,
    ).to(device)
    print('model', ensembler)
    print('number of trainable parameters in model:', (count_parameters(ensembler)))

    if z_smoothness > 0:  # gp latents
        z_train = torch_circular_gp(num_sample, latent_dim * num_ensemble, z_smoothness)
        z_test = torch_circular_gp(num_test, latent_dim * num_ensemble, z_smoothness)
    else:  # iid latents
        z_train = torch.rand(num_sample, latent_dim * num_ensemble) * 2 * np.pi
        z_test = torch.rand(num_test, latent_dim * num_ensemble).to(device) * 2 * np.pi

    z_train = z_train.to(device)
    z_test = z_test.to(device)
    data_train = simulator(z_train).detach()
    simulator.noise = False
    data_test = simulator(z_test).detach()
    simulator.noise = True

    trainer = Trainer(
        model=ensembler,
        data_train=data_train.cpu().numpy(),
        data_test=data_test.cpu().numpy(),
        neurons_train_ind=neurons_train_ind,
        mode='full',
        z_train=None,
        z_test=None,
        seed=923683,
    )
    trainer.train()
    analysis(ensembler, simulator, trainer, z_test)
    print("Repeat analysis with good inference:")
    analysis(ensembler, simulator, trainer, z_test, do_inference=True)