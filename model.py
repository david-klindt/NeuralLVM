import torch
import numpy as np
from utils import get_grid, angle2vector, vector2angle
from hyperspherical_vae import reparameterize

device = "cuda" if torch.cuda.is_available() else "cpu"
print('Running on', device)

"""
Dimension names used below:
    B=Batch
    L=Length
    N=Neurons
    H=num_basis
    E=num_ensemble
    D=latent_dim
    V=angle as vector (2D)
"""


class Encoder(torch.nn.Module):
    def __init__(
            self,
            in_channels,
            latent_manifolds=('T1', 'R2'),  # this gives S1 and R2 latent spaces
            kernel_size=9,
            num_hidden=256,
            seed=2938057
    ):
        super(Encoder, self).__init__()
        self.in_channels = in_channels
        self.latent_manifolds = latent_manifolds
        self.kernel_size = kernel_size
        self.num_hidden = num_hidden
        torch.manual_seed(seed)

        self.network = torch.nn.Sequential(
            torch.nn.Conv1d(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=kernel_size,
                padding='same',
                groups=in_channels
            ),
            torch.nn.Conv1d(
                in_channels=in_channels,
                out_channels=num_hidden,
                kernel_size=1,
                padding='same',
            ),
            torch.nn.ReLU(),
            torch.nn.Conv1d(
                in_channels=num_hidden,
                out_channels=num_hidden,
                kernel_size=1,
                padding='same',
            ),
            torch.nn.ReLU(),
        )

        self.mean_head = torch.nn.ModuleList()
        self.var_head = torch.nn.ModuleList()
        for i, m in enumerate(latent_manifolds):
            out_channels_mean = out_channels_var = int(m[1])  # latent_dim
            if m[0] == 'T':  # if latent is on sphere, output vector in 2D and normalize
                out_channels_mean *= 2
            self.mean_head.append(
                torch.nn.Conv1d(
                    in_channels=num_hidden,
                    out_channels=out_channels_mean,
                    kernel_size=1,
                    padding='same'
                )
            )
            self.var_head.append(
                torch.nn.Conv1d(
                    in_channels=num_hidden,
                    out_channels=out_channels_var,
                    kernel_size=1,
                    padding='same'
                )
            )

    def forward(self, x):
        batch_size, num_neuron_train, length = x.shape
        assert num_neuron_train == self.in_channels

        hidden_activations = self.network(x)

        output = {'mean': [], 'logvar': [], 'q_z': [], 'p_z': [], 'z': []}
        for i, m in enumerate(self.latent_manifolds):
            latent_dim = int(m[1])
            mean = self.mean_head[i](hidden_activations).permute(0, 2, 1)  # B x L x D(*2)
            logvar = self.var_head[i](hidden_activations).permute(0, 2, 1)  # B x L x D
            if m[0] == 'T':  # if latent is on sphere, output vector in 2D and normalize
                mean = mean.view(batch_size, length, latent_dim, 2)  # B x L x D x V
                mean = mean / mean.norm(dim=3, keepdim=True)
                kappa = torch.nn.functional.softplus(logvar) + 1  # the `+ 1` prevent collapsing behaviors (from sVAE)
                q_z, p_z = reparameterize('vmf', mean, kappa)
                z = q_z.rsample()  # B x L x D x V
                z = vector2angle(z)  # B x L x D
            else:
                var = torch.exp(logvar)
                q_z, p_z = reparameterize('normal', mean, var)
                z = q_z.rsample()  # B x L x D
            output['mean'].append(mean)
            output['logvar'].append(logvar)
            output['q_z'].append(q_z)
            output['p_z'].append(p_z)
            output['z'].append(z)

        return output


class FeatureBasis(torch.nn.Module):  # one for each latent manifold
    def __init__(
            self,
            num_neuron,
            feature_type='gauss',  # {'gauss', 'fourier'}
            shared=True,
            flexibility=(True, True, True),  # learn: coefficients, means, variances
            num_basis=1,
            latent_dim=2,
            manifold='torus',  # {'torus', 'euclidean'} eg. 'torus' + latent_dim=1 = S1; 'euclidean' + latent_dim=2 = R2
            nonlinearity='exp',
            seed=345978,
    ):
        super(FeatureBasis, self).__init__()
        self.num_neuron = num_neuron
        self.feature_type = feature_type
        self.shared = shared
        self.flexibility = flexibility
        self.num_basis = num_basis
        self.latent_dim = latent_dim
        self.manifold = manifold
        self.nonlinearity = nonlinearity
        torch.manual_seed(seed)

        coeff_shape = (1 if shared else num_neuron, num_basis)
        parameter_shape = (1 if shared else num_neuron, num_basis, latent_dim)

        # ToDo(Martin): what are good values here below?
        if feature_type == 'gauss':
            coeff_init = torch.randn(coeff_shape) * 1e-2
            # initialize randomly in [-pi, pi]
            # mean_init = torch.rand(parameter_shape) * 2 * np.pi - np.pi
            # initialize to evenly spread grid (produces a total of num_basis**latent_dim basis functions)
            mean_init = torch.tensor(get_grid(latent_dim, num_basis))
            log_var_init = torch.log(torch.ones(parameter_shape) * 10.)
            coeff_init[:, 0] += 1  # initialize close to single bump
        elif feature_type == 'fourier':
            raise ValueError("not yet fully implemented")
            coeff_shape = (num_basis * 2 + 1, 1 if shared else num_neuron)
            coeff_init = torch.randn(coeff_shape) * 1e-3
            mean_init = torch.zeros(1)  # not used
            log_var_init = torch.zeros(1)  # not used
            # initialize as DC + slowest cosine:
            coeff_init[0] += 1
            coeff_init[1] += 1
        else:
            raise ValueError("feature_type unknown")

        self.coeff = torch.nn.Parameter(coeff_init, requires_grad=flexibility[0])
        self.mean = torch.nn.Parameter(mean_init, requires_grad=flexibility[1])
        self.log_var = torch.nn.Parameter(log_var_init, requires_grad=flexibility[2])

    def forward(self, z, rf, test=False):
        """
        z: latents
        rf: receptive_field_centers
        test: if True, don't train feature basis
        """
        z_shape = z.shape  # B x L x D
        assert len(z_shape) == 3 and z_shape[2] == self.latent_dim
        rf_shape = rf.shape  # N x D
        assert rf_shape[1] == self.latent_dim

        coeff = self.coeff.detach() if test else self.coeff  # {N, 1} x H
        mean = self.mean.detach() if test else self.mean  # {N, 1} x H x D
        log_var = self.log_var.detach() if test else self.log_var  # {N, 1} x H x D

        if self.feature_type == 'gauss':
            if self.manifold == 'torus':  # add (x V) as last dimension, i.e., embedding angle in 2D
                z = angle2vector(z)  # B x L x D x V
                rf = angle2vector(rf)  # N x D x V
                mean = angle2vector(mean)  # {N, 1} x H x D x V
                sum_over = (4, 5)
            else:
                sum_over = 4
            mean = mean - rf[:, None]  # N x H x D (x V)
            dist = z[:, :, None, None] - mean[None, None]  # B x L x N x H x D (x V)
            dist = dist / torch.exp(log_var[None, None])
            dist = torch.sum(dist ** 2, dim=sum_over)  # B x L x N x H
            response = torch.sum(torch.exp(- dist) * coeff[None, None], dim=3)  # B x L x N
        elif self.feature_type == 'fourier':
            raise ValueError("not yet fully implemented")

        if self.nonlinearity == 'exp':
            response = torch.exp(response)
        elif self.nonlinearity == 'softplus':
            response = torch.log(torch.exp(response) + 1)

        return response


class Decoder(torch.nn.Module):
    def __init__(
            self,
            num_neuron_train,
            num_neuron_test,
            latent_manifolds=('T1', 'R2'),  # this gives S1 and R2 latent spaces
            feature_type=('gauss', 'gauss'),  # {'gauss', 'fourier'}
            shared=(True, True),
            flexibility=((True, True, True), (True, True, True)),  # learn: coefficients, means, variances
            num_basis=(1, 16),  # ignored for feature_type='bump'
            seed=2093857,
    ):
        super(Decoder, self).__init__()
        self.num_neuron_train = num_neuron_train  # used to infer latents and learn shared features
        self.num_neuron_test = num_neuron_test  # only for testing, learn RFs given fixed feature basis and latents.
        self.latent_manifolds = latent_manifolds
        self.feature_type = feature_type
        self.shared = shared
        self.flexibility = flexibility
        self.num_basis = num_basis
        torch.manual_seed(seed)

        # Get latent manifolds
        self.num_ensemble = len(latent_manifolds)
        self.feature_bases_train = torch.nn.ModuleList()
        self.feature_bases_test = torch.nn.ModuleList()
        self.receptive_fields_train = torch.nn.ParameterList()
        self.receptive_fields_test = torch.nn.ParameterList()

        for i, m in enumerate(latent_manifolds):
            latent_dim = int(m[1])
            if m[0] == 'R':
                manifold = 'euclidean'
                init_fn = lambda shape: torch.randn(shape)  # initialize randomly standard normal
            elif m[0] == 'T':
                manifold = 'torus'
                init_fn = lambda shape: torch.rand(shape) * 2 * np.pi - np.pi  # initialize uniformly in [-pi, pi]
            else:
                raise ValueError("latent manifold not yet implemented.")
            self.feature_bases_train.append(
                FeatureBasis(num_neuron_train, feature_type=feature_type[i],  shared=shared[i],
                             flexibility=flexibility[i], num_basis=num_basis[i], latent_dim=latent_dim,
                             manifold=manifold, seed=seed)
            )
            if not self.shared[i]:
                self.feature_bases_test.append(
                    FeatureBasis(num_neuron_test, feature_type=feature_type[i], shared=shared[i],
                                 flexibility=flexibility[i], num_basis=num_basis[i], latent_dim=latent_dim,
                                 manifold=manifold, seed=seed)
                )

            self.receptive_fields_train.append(
                torch.nn.Parameter(init_fn((num_neuron_train, latent_dim)), requires_grad=True)
            )
            self.receptive_fields_test.append(
                torch.nn.Parameter(init_fn((num_neuron_test, latent_dim)), requires_grad=True)
            )

        # Prepare final readout
        self.ensemble_weights_train = torch.nn.Parameter(
            torch.randn(num_neuron_train, self.num_ensemble), requires_grad=True
        )
        self.ensemble_weights_test = torch.nn.Parameter(
            torch.randn(num_neuron_test, self.num_ensemble), requires_grad=True
        )
        self.log_final_scale_train = torch.nn.Parameter(  # intialize constant at 1 (this goes through exp before use)
            torch.zeros(num_neuron_train), requires_grad=True
        )
        self.log_final_scale_test = torch.nn.Parameter(  # intialize constant at 1 (this goes through exp before use)
            torch.zeros(num_neuron_test), requires_grad=True
        )

    def forward(self, z):
        """z is list of latents for each manifold"""
        responses_train, responses_test = [], []
        for i in range(self.num_ensemble):
            responses_train.append(
                self.feature_bases_train[i](z[i], self.receptive_fields_train[i], test=False)
            )
            if self.shared[i]:
                responses_test.append(
                    self.feature_bases_train[i](z[i].detach(), self.receptive_fields_test[i], test=True)
                )
            else:
                responses_test.append(
                    self.feature_bases_test[i](z[i].detach(), self.receptive_fields_test[i], test=False)
                )
        # ToDo(add shared latents)
        responses_train = torch.stack(responses_train, dim=3)  # B x L x N x E
        responses_test = torch.stack(responses_test, dim=3)  # B x L x N x E
        ensemble_weights_train = torch.nn.functional.softmax(
            self.ensemble_weights_train, dim=1)[None, None]  # 1 x 1 x N x E
        ensemble_weights_test = torch.nn.functional.softmax(
            self.ensemble_weights_test, dim=1)[None, None]  # 1 x 1 x N x E
        responses_train = torch.sum(ensemble_weights_train * responses_train, dim=3)  # B x L x N
        responses_test = torch.sum(ensemble_weights_test * responses_test, dim=3)  # B x L x N
        responses_train = responses_train * torch.exp(self.log_final_scale_train[None, None])
        responses_test = responses_test * torch.exp(self.log_final_scale_test[None, None])
        responses_train = responses_train.permute(0, 2, 1)  # B x N x L
        responses_test = responses_test.permute(0, 2, 1)  # B x N x L
        return responses_train, responses_test


class Model(torch.nn.Module):
    def __init__(
            self,
            num_neuron_train,
            num_neuron_test,
            kernel_size=9,
            num_hidden=256,
            latent_manifolds=('T1', 'R2'),  # this gives S1 and R2 latent spaces
            feature_type=('gauss', 'gauss'),  # {'gauss', 'fourier'}
            shared=(True, True),
            flexibility=((True, True, True), (True, True, True)),  # learn: coefficients, means, variances
            num_basis=(1, 1),  # for 1 and 'gauss' = 'bump' model, careful this scales as num_basis**n (e.g. n=2 for R2)
            seed=1293842,
    ):
        super(Model, self).__init__()
        self.num_neuron_train = num_neuron_train  # used to infer latents and learn shared features
        self.num_neuron_test = num_neuron_test  # only for testing, learn RFs given fixed feature basis and latents.
        self.kernel_size = kernel_size
        self.num_hidden = num_hidden
        self.latent_manifolds = latent_manifolds
        self.feature_type = feature_type
        self.shared = shared
        self.flexibility = flexibility
        self.num_basis = num_basis
        self.seed = seed

        self.encoder = Encoder(
            in_channels=num_neuron_train, latent_manifolds=latent_manifolds, kernel_size=kernel_size,
            num_hidden=num_hidden, seed=seed
        )
        self.decoder = Decoder(
            num_neuron_train, num_neuron_test, latent_manifolds=latent_manifolds, feature_type=feature_type,
            shared=shared, flexibility=flexibility, num_basis=num_basis, seed=seed
        )

    def forward(self, x, z=None):
        # required input shape: (B x N x L)
        batch_size, num_neuron_train, length = x.shape
        assert num_neuron_train == self.num_neuron_train

        # run model
        output = self.encoder(x)
        if z is None:
            responses_train, responses_test = self.decoder(output['z'])
        else:   # use externally fed-in z
            for a, b in zip(z, output['z']):
                assert a.shape == b.shape
            responses_train, responses_test = self.decoder(z)
        output['responses_train'] = responses_train
        output['responses_test'] = responses_test

        return output
