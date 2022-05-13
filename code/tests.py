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

def test_training(
    kernel_size=9,
    batch_length=64,
    num_hidden=256,
    weight_time=0,
    weight_entropy=0,
    num_worse=50,
    weight_kl=1e-6,
    learning_rate=3e-3,
    batch_size=16,
    num_ensemble=2,
    num_neuron_train=50,
    num_neuron_test=50,
    latent_dim=2,
    z_smoothness=3,
    num_basis=1,
    shared=True,
    learn_coeff=True,
    learn_mean=False,
    learn_var=True,
    isotropic=True,
    num_sample=10000,
    num_test=1000,
    feature_type="gauss",
    latent_manifolds="T2",
    seed=seed,
):
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
        kernel_size=kernel_size,
        num_hidden=num_hidden,
        latent_manifolds=(latent_manifolds,) * num_ensemble,
        feature_type=(feature_type) * num_ensemble,
        shared=(shared,) * num_ensemble,
        learn_coeff=(learn_coeff,) * num_ensemble,
        learn_mean=(learn_mean,) * num_ensemble,
        learn_var=(learn_var,) * num_ensemble,
        isotropic=(isotropic,) * num_ensemble,
        num_basis=(num_basis,) * num_ensemble,
        seed=seed,
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

    label_train = np.zeros(num_neuron_train * num_ensemble)
    label_test = np.zeros(num_neuron_test * num_ensemble)
    for i in range(num_ensemble):
        label_train[i * num_neuron_train: (i + 1) * num_neuron_train] = i
        label_test[i * num_neuron_test: (i + 1) * num_neuron_test] = i

    trainer = Trainer(
        model=ensembler,
        data_train=data_train.cpu().numpy(),
        data_test=data_test.cpu().numpy(),
        neurons_train_ind=neurons_train_ind,
        mode='full',
        z_train=None,
        z_test=None,
        label_train=label_train,
        label_test=label_test,
        num_steps=100000,
        num_log_step=100,
        batch_size=batch_size,
        batch_length=batch_length,
        learning_rate=learning_rate,
        num_worse=num_worse,  # if loss doesn't improve X times, stop.
        weight_kl=weight_kl,
        weight_time=weight_time,
        weight_entropy=weight_entropy,
        log_dir="model_ckpt",
        log_training=True,
        seed=seed,
    )
    output = trainer.train()

    # ToDo(pickle save output file in model_ckpt folder)

    # ToDo(fix analysis code for refactored model)
    #analysis(ensembler, simulator, trainer, z_test)
    #print("Repeat analysis with good inference:")
    #analysis(ensembler, simulator, trainer, z_test, do_inference=True))


if __name__ == "__main__":
    pass

"""
Hyperparameter Search:

### fix:

num_ensemble=2
latent_manifolds=('T2', 'T2')
num_neuron_train=50
num_neuron_test=50
latent_dim=2
feature_type=('gauss', 'gauss')
z_smoothness=3
num_test=1000
num_steps=100000
num_log_step=100


### search over:

# data (this is more to check)
num_sample: [1000, 10000, 100000]
num_neuron_train: [10, 50, 100]
seed: ...

# model
kernel_size: [1, 3, 9, 17, 33],
num_hidden: [16, 32, 64, 128, 256, 512]
shared: [(True, True), (False, False)]
learn_coeff: [(True, True), (False, False)]
learn_mean: [(True, True), (False, False)]
learn_var: [(True, True), (False, False)]
isotropic: [(True, True), (False, False)]
num_basis: [(1, 1), (2, 2), (4, 4), (8, 8)]
seed: ...

# training
batch_size: [1, 16, 32]
batch_length: [64, 128]
learning_rate: [1e-3, 3e-3, 1e-2, 3e-2]
num_worse: [10, 50, 100]
weight_kl: [0 or geomspace(1e-9, 1e0)]
weight_time: [0 or geomspace(1e-9, 1e0)]
weight_entropy: [0 or geomspace(1e-9, 1e0)]
seed: ...

"""
