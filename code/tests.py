import torch
import numpy as np
from code.utils import count_parameters
from code.utils import torch_circular_gp
from code.utils import analysis
from code.model import Model
from code.training import Trainer
from code.data import StochasticNeurons

device = "cuda" if torch.cuda.is_available() else "cpu"


def test_training(
        num_ensemble=2,
        num_neuron_train=50,
        num_neuron_test=50,
        latent_dim=2,
        z_smoothness=3,
        num_basis=1,
        num_sample=10000,
        num_test=1000
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
        nonlinearity=('exp', 'exp'),
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
        batch_size=16,
        batch_length=128,
        learning_rate=3e-3,
        num_worse=100,  # if loss doesn't improve X times, stop.
        weight_kl=1e-6,
        weight_time=0,
        weight_entropy=0,
        log_dir='model_ckpt',
        log_training=True,
        seed=923683,
    )
    output = trainer.train()

    # ToDo(fix analysis code for refactored model)
    #analysis(ensembler, simulator, trainer, z_test)
    #print("Repeat analysis with good inference:")
    #analysis(ensembler, simulator, trainer, z_test, do_inference=True)