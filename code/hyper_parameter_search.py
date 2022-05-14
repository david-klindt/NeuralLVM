import torch
import numpy as np
from NeuralLVM.code.utils import count_parameters
from NeuralLVM.code.utils import torch_circular_gp
from NeuralLVM.code.model import Model
from NeuralLVM.code.training import Trainer
from NeuralLVM.code.data import StochasticNeurons

device = "cuda" if torch.cuda.is_available() else "cpu"



def randint(low, high):
	return np.int(np.random.randint(low, high, 1)[0])


def uniform(low, high):
    return np.random.uniform(low, high, 1)[0]


def loguniform(low, high):
	return np.exp(np.random.uniform(np.log(low), np.log(high), 1))[0]


def choice(options):
    output = np.random.choice(options)
    if type(options[0]) == bool:
        output = bool(output)
    return output


def get_config(index, num_repetition=3, num_seeds=1000000, global_seed=42):
    config = dict()

    # generate list of seeds, and choose some for this run and repetitions
    all_seeds = np.arange(num_seeds * num_repetition)
    np.random.seed(global_seed)
    np.random.shuffle(all_seeds)
    all_seeds = all_seeds.reshape(num_seeds, num_repetition)
    np.random.seed(all_seeds[index][0])  # for draws below
    config['seeds'] = all_seeds[index]

    ### fixed:
    config['num_ensemble'] = 2
    config['latent_manifolds'] = 'T2'
    config['num_neuron_train'] = 50
    config['num_neuron_test'] = 50
    config['latent_dim'] = 2
    config['feature_type'] = 'gauss'
    config['z_smoothness'] = 3
    config['num_test'] = 1000
    config['num_steps'] = 100000
    config['num_log_step'] = 100
    config['num_sample'] = 10000  # [1000, 10000, 100000]
    config['num_neuron_train'] = 50  # [10, 50, 100]

    ### search over:
    # model
    config['kernel_size'] = choice([1, 9, 17])
    config['num_hidden'] = choice([16, 32, 64, 128, 256, 512])
    config['shared'] = choice([True, False])
    config['learn_coeff'] = choice([True, False])
    config['learn_mean'] = choice([True, False])
    config['learn_var'] = choice([True, False])
    config['isotropic'] = choice([True, False])
    config['num_basis'] = choice([1, 2, 4, 8])
    config['nonlinearity'] = choice(['exp', 'softplus'])
    # training
    config['batch_size'] = choice([1, 16, 32])
    config['batch_length'] = choice([64, 128])
    config['learning_rate'] = loguniform(1e-4, 1e-1)
    config['num_worse'] = choice([10, 50, 100])
    config['weight_kl'] = choice([0., loguniform(1e-9, 1e0)])
    config['weight_time'] = choice([0., loguniform(1e-9, 1e0)])
    config['weight_entropy'] = choice([0., loguniform(1e-9, 1e0)])

    return config


def run_experiment(config, repetition):

    # ToDo(pickle save config file in model_ckpt folder)

    seed = config['seeds'][repetition]
    np.random.seed(seed)
    torch.manual_seed(seed)

    num_neuron = config['num_neuron_train'] + config['num_neuron_test']
    neurons_train_ind = np.zeros(num_neuron * config['num_ensemble'], dtype=bool)
    ind = np.random.choice(
        num_neuron * config['num_ensemble'],
        config['num_neuron_train'] * config['num_ensemble'],
        replace=False
    )
    neurons_train_ind[ind] = True
    simulator = StochasticNeurons(
        num_neuron, num_ensemble=config['num_ensemble'], noise=True,
        latent_dim=config['latent_dim'], seed=seed).to(device)
    ensembler = Model(
        num_neuron_train=config['num_neuron_train'] * config['num_ensemble'],
        num_neuron_test=config['num_neuron_test'] * config['num_ensemble'],
        kernel_size=9,
        num_hidden=256,
        num_ensemble=config['num_ensemble'],
        latent_manifolds=config['latent_manifolds'],
        feature_type=config['feature_type'],
        shared=config['shared'],
        learn_coeff=config['learn_coeff'],
        learn_mean=config['learn_coeff'],
        learn_var=config['learn_var'],
        isotropic=config['isotropic'],
        num_basis=config['num_basis'],
        nonlinearity=config['nonlinearity'],
        seed=seed,
    ).to(device)
    print('model', ensembler)
    print('number of trainable parameters in model:', (count_parameters(ensembler)))

    if config['z_smoothness'] > 0:  # gp latents
        z_train = torch_circular_gp(
            config['num_sample'],  config['latent_dim'] * config['num_ensemble'], config['z_smoothness'])
        z_test = torch_circular_gp(
            config['num_test'], config['latent_dim'] * config['num_ensemble'], config['z_smoothness'])
    else:  # iid latents
        z_train = torch.rand(
            config['num_sample'], config['latent_dim'] * config['num_ensemble']) * 2 * np.pi
        z_test = torch.rand(
            config['num_test'], config['latent_dim'] * config['num_ensemble']).to(device) * 2 * np.pi

    z_train = z_train.to(device)
    z_test = z_test.to(device)
    data_train = simulator(z_train).detach()
    simulator.noise = False
    data_test = simulator(z_test).detach()
    simulator.noise = True

    label_train = np.zeros(config['num_neuron_train'] * config['num_ensemble'])
    label_test = np.zeros(config['num_neuron_test'] * config['num_ensemble'])
    for i in range(config['num_ensemble']):
        label_train[i * config['num_neuron_train']: (i + 1) * config['num_neuron_train']] = i
        label_test[i * config['num_neuron_test']: (i + 1) * config['num_neuron_test']] = i

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
        num_steps=config['num_steps'],
        num_log_step=config['num_steps'],
        batch_size=config['batch_size'],
        batch_length=config['batch_length'],
        learning_rate=config['learning_rate'],
        num_worse=config['num_worse'],
        weight_kl=config['weight_kl'],
        weight_time=config['weight_time'],
        weight_entropy=config['weight_entropy'],
        log_dir='model_ckpt',
        log_training=True,
        seed=seed
    )
    output = trainer.train()

    # ToDo(pickle save output file in model_ckpt folder)
    del output['q_z'], output['p_z']  # those cannot be pickled (pickle Rick!!!)