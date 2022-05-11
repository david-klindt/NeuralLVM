import torch
import numpy as np
from utils import *

device = "cuda" if torch.cuda.is_available() else "cpu"
print('Running on', device)


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
        input_shape = x.shape
        batch_size, num_neuron_train, length = x.shape
        assert num_neuron_train == self.in_channels

        hidden_activations = self.network(x)


        x = self.encoder(x)
        mu = self.mean_head(x)  # B x E*D*V x L
        mu = mu.permute(0, 2, 1)  # B x L x E*D*V
        mu = mu.view(  # B x L x E x D x V
            batch_size, length, self.num_ensemble, self.latent_dim, 2)
        logvar = self.var_head(x)  # B x E*D x L
        logvar = logvar.permute(0, 2, 1)  # B x L x E*D
        logvar = logvar.view(  # B x L x E x D
            batch_size, length, self.num_ensemble, self.latent_dim)


        mu, logvar = []
        for i, m in enumerate(self.latent_manifolds):
            latent_dim = int(m[1])
            mean = self.mean_head[i](hidden_activations).permute(0, 2, 1)  # B x L x D(*2)
            logvar = self.var_head[i](hidden_activations).permute(0, 2, 1)  # B x L x D
            if m[0] == 'T':  # if latent is on sphere, output vector in 2D and normalize
                mean = mean.view(batch_size, length, latent_dim, 2)  # B x L x D x V
                mean = mean / mean.norm(dim=3, keepdim=True)
                var = torch.nn.functional.softplus(logvar) + 1  # the `+ 1` prevent collapsing behaviors (from hsVAE)
            else


            if self.latent_style == 'hyper':
                mu = mu / mu.norm(dim=-1, keepdim=True)

                logvar =
            elif self.normalize_encodings:
                mu = mu / mu.norm(dim=-1, keepdim=True)

            if z is None:
                if self.latent_style == 'hack':
                    z_angle = vector2angle(mu)  # B x L x E x D
                    z = reparameterize(z_angle, logvar)  # B x L x E x D
                elif self.latent_style == 'hyper':
                    q_z, p_z = reparameterize_vmf(mu, logvar)
                    z = q_z.rsample()  # B x L x E x D x V
                    z = vector2angle(z)  # B x L x E x D
            else:
                q_z, p_z = None, None


        return


class FeatureBasis(torch.nn.Module):  # one for each latent manifold
    def __init__(
            self,
            num_neuron,
            feature_type='bump',  # {'bump', 'gauss', 'fourier'}
            shared=True,
            flexibility=(True, True, True),  # learn: coefficients, means, variances
            num_basis=16,  # ignored for feature_type='bump'
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
        if feature_type == 'bump':
            coeff_init = torch.ones(coeff_shape)
            mean_init = torch.zeros(parameter_shape)
            log_var_init = torch.ones(parameter_shape) * np.log(10)
        elif feature_type == 'gauss':
            coeff_init = torch.randn(coeff_shape) * 1e-3
            mean_init = torch.rand(parameter_shape) * 2 * np.pi - np.pi  # initialize randomly in [-pi, pi]
            log_var_init = torch.ones(parameter_shape) * 2 * np.pi / num_basis
            # initialize as single bump:
            coeff_init[0] += 1
            mean_init[0] *= 0
            log_var_init[0] *= np.log(10)
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
        dimension names:    B=Batch, L=Length, N=Neurons, H=num_basis, E=num_ensemble,
                            D=latent_dim, V=angle as vector (2D)
        """
        z_shape = z.shape  # B x L x D
        assert len(z_shape) == 3 and z_shape[2] == self.latent_dim
        rf_shape = rf.shape  # N x D
        assert rf_shape == (self.num_neuron, self.latent_dim)

        coeff = self.coeff.detach() if test else self.coeff  # {N, 1} x H
        mean = self.mean.detach() if test else self.mean  # {N, 1} x H x D
        log_var = self.log_var.detach() if test else self.log_var  # {N, 1} x H x D

        if self.feature_type in ['bump', 'gauss']:
            if self.manifold == 'torus':  # add (x V) as last dimension, i.e., embedding angle in 2D
                z = angle2vector(z)  # B x L x D x V
                rf = angle2vector(rf)  # N x D x V
                mean = angle2vector(mean)  # {N, 1} x H x D x V
                sum_over = (4, 5)
            else:
                sum_over = 4
            mean = mean - rf[:, None]  # N x H x D (x V)
            dist = z[:, :, None, None] - mean[None, None]  # B x L x N x H x D (x V)
            dist = torch.sum(dist ** 2, dim=sum_over)  # B x L x N x H
            response = torch.exp(- dist / torch.exp(log_var))
            response = torch.sum(response * coeff[None, None], dim=3)  # B x L x N
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
            feature_type=('bump', 'gauss'),  # {'bump', 'gauss', 'fourier'}
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
                self.feature_bases_train(z[i], self.receptive_fields_train[i], test=False)
            )
            if self.shared[i]:
                responses_test.append(
                    self.feature_bases_train(z[i].detach(), self.receptive_fields_test[i], test=True)
                )
            else:
                responses_test.append(
                    self.feature_bases_test(z[i].detach(), self.receptive_fields_test[i], test=False)
                )
        responses_train = torch.stack(responses_train, dim=4)  # Batch x Length x Neuron x Latent
        responses_test = torch.stack(responses_test, dim=4)  # Batch x Length x Neuron x Latent
        ensemble_weights_train = torch.nn.functional.softmax(
            self.ensemble_weights_train, dim=1)[None, None]  # 1 x 1 x Neuron x Latent
        ensemble_weights_test = torch.nn.functional.softmax(
            self.ensemble_weights_test, dim=1)[None, None]  # 1 x 1 x Neuron x Latent
        responses_train = torch.sum(ensemble_weights_train * responses_train, dim=3)  # Batch x Length x Neuron
        responses_test = torch.sum(ensemble_weights_test * responses_test, dim=3)  # Batch x Length x Neuron
        responses_train = responses_train * torch.exp(self.log_final_scale_train[None, None])
        responses_test = responses_test * torch.exp(self.log_final_scale_test[None, None])
        responses_train = responses_train.permute(0, 2, 1)  # Batch x Neuron x Length
        responses_test = responses_test.permute(0, 2, 1)  # Batch x Neuron x Length
        return responses_train, responses_test


class Model(torch.nn.Module):
    def __init__(
            self,
            num_neuron_train,
            num_neuron_test,
            kernel_size=9,
            num_hidden=256,
            latent_manifolds=('T1', 'R2'),  # this gives S1 and R2 latent spaces
            feature_type=('bump', 'gauss'),  # {'bump', 'gauss', 'fourier'}
            shared=(True, True),
            flexibility=((True, True, True), (True, True, True)),  # learn: coefficients, means, variances
            num_basis=(1, 16),  # ignored for feature_type='bump'
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
            in_channels=num_neuron_train, kernel_size=kernel_size, num_hidden=num_hidden, seed=seed
        )
        self.decoder = Decoder(
            num_neuron_train, num_neuron_test, latent_manifolds=latent_manifolds, feature_type=feature_type,
            shared=shared, flexibility=flexibility, num_basis=num_basis, seed=seed
        )

    def forward(self, x):





############## PREVIOUS CODE ################



class LatentVariableModel(torch.nn.Module):
    def __init__(
            self,
            num_neuron_train,
            num_neuron_test,
            num_hidden=256,
            num_ensemble=2,
            latent_dim=2,
            seed=2093857,
            tuning_width=10.0,
            nonlinearity='exp',
            kernel_size=1,
            normalize_encodings=True,
            feature_type='bump',  # {'bump', 'shared', 'separate'}
            num_feature_basis=3,  # careful this scales badly with latent_dim!
    ):
        super(LatentVariableModel, self).__init__()
        self.num_neuron_train = num_neuron_train  # used to infer latents and learn shared features
        self.num_neuron_test = num_neuron_test  # only used for testing, learn their RFs given fixed feature basis and
        # inferred latents.
        self.num_ensemble = num_ensemble
        self.latent_dim = latent_dim
        self.nonlinearity = nonlinearity
        self.kernel_size = kernel_size
        self.normalize_encodings = normalize_encodings
        self.feature_type = feature_type

        torch.manual_seed(seed)
        self.receptive_fields_train = torch.nn.Parameter(
            # initialize randomly in [-pi, pi]
            - np.pi + 2 * np.pi * torch.rand(
                num_neuron_train, num_ensemble, latent_dim),
            requires_grad=True#not(feature_type.startswith('separate'))
        )
        self.receptive_fields_test = torch.nn.Parameter(
            # initialize randomly in [-pi, pi]
            - np.pi + 2 * np.pi * torch.rand(
                num_neuron_test, num_ensemble, latent_dim),
            requires_grad=True#not(feature_type.startswith('separate'))
        )
        self.ensemble_weights_train = torch.nn.Parameter(
            torch.randn(num_neuron_train, num_ensemble),
            requires_grad=True
        )
        self.ensemble_weights_test = torch.nn.Parameter(
            torch.randn(num_neuron_test, num_ensemble),
            requires_grad=True
        )
        self.log_final_scale_train = torch.nn.Parameter(
            # intialize constant at 1
            torch.zeros(num_neuron_train),
            requires_grad=True
        )
        self.log_final_scale_test = torch.nn.Parameter(
            # intialize constant at 1
            torch.zeros(num_neuron_test),
            requires_grad=True
        )
        self.encoder = torch.nn.Sequential(
            torch.nn.Conv1d(
                in_channels=num_neuron_train,
                out_channels=num_neuron_train,
                kernel_size=kernel_size,
                padding='same',
                groups=num_neuron_train
            ),
            torch.nn.Conv1d(
                in_channels=num_neuron_train,
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
        self.mean_head = torch.nn.Conv1d(
            in_channels=num_hidden,
            out_channels=num_ensemble * latent_dim * 2,
            kernel_size=1,
            padding='same'
        )
        self.var_head = torch.nn.Conv1d(
            in_channels=num_hidden,
            out_channels=num_ensemble * latent_dim,
            kernel_size=1,
            padding='same'
        )

        self.feature_basis = FeatureBasis(
            num_neuron_train,
            feature_type=feature_type,
            num_basis=num_feature_basis,
            latent_dim=latent_dim,
            tuning_width=tuning_width,
            nonlinearity=nonlinearity,
            variance=None,
            seed=seed,
        )
        if feature_type.startswith('separate'):  # make second for test if no sharing
            self.feature_basis_test = FeatureBasis(
                num_neuron_test,
                feature_type=feature_type,
                num_basis=num_feature_basis,
                latent_dim=latent_dim,
                tuning_width=tuning_width,
                nonlinearity=nonlinearity,
                variance=None,
                seed=seed,
            )

    def forward(self, x, z=None):
        # dimension names: B=Batch, L=Length, N=Neurons,
        # E=num_ensemble, D=latent_dim, V=angle as vector (2D)
        input_shape = x.shape
        if len(input_shape) == 2:
            # required input shape: B x N x L
            # if input is only N x L, prepend B dimension
            x = x[None]
        batch_size, num_neuron_train, length = x.shape
        assert num_neuron_train == self.num_neuron_train

        x = self.encoder(x)
        mu = self.mean_head(x)  # B x E*D*V x L
        mu = mu.permute(0, 2, 1)  # B x L x E*D*V
        mu = mu.view(  # B x L x E x D x V
            batch_size, length, self.num_ensemble, self.latent_dim, 2)
        logvar = self.var_head(x)  # B x E*D x L
        logvar = logvar.permute(0, 2, 1)  # B x L x E*D
        logvar = logvar.view(  # B x L x E x D
            batch_size, length, self.num_ensemble, self.latent_dim)

        if self.normalize_encodings:
            mu = mu / torch.sum(mu ** 2, dim=-1, keepdim=True) ** .5

        if z is None:
            z_angle = vector2angle(mu)  # B x L x E x D
            z = reparameterize(z_angle, logvar)  # B x L x E x D

        # Compute responses
        is_test = 0  # compute gradients for complete model
        response_train = self.compute_responses(
            self.ensemble_weights_train,
            self.log_final_scale_train,
            self.feature_basis(z, self.receptive_fields_train, is_test=is_test),
            input_shape
        )
        if self.feature_type.startswith('separate'):
            feature_basis_test = self.feature_basis_test
            is_test = 1  # 1 - grads only for decoder
        else:
            feature_basis_test = self.feature_basis
            is_test = 2  # 2 - no grads
        response_test = self.compute_responses(
            self.ensemble_weights_test,
            self.log_final_scale_test,
            feature_basis_test(z, self.receptive_fields_test, is_test=is_test),
            input_shape
        )

        return response_train, response_test, z, mu, logvar

    def compute_responses(self, ensemble_weights, log_final_scale, response, input_shape):
        ensemble_weights = torch.nn.functional.softmax(  # 1 x 1 x N x E
            ensemble_weights, dim=1)[None, None]
        responses = torch.sum(  # B x L x N
            ensemble_weights * response, dim=3)
        responses = responses * torch.exp(log_final_scale[None, None])
        responses = responses.permute(0, 2, 1)  # B x N x L
        if len(input_shape) == 2:
            # if input had no batch dimension, remove this again
            responses = responses[0]
        return responses


def inference(
        model,
        responses_train_neurons,
        responses_test_neurons,
        num_sample=10,
        num_iter=2000,
        learning_rate=1e-3,
    ):
    model.eval()
    y_train = responses_train_neurons
    y_test = responses_test_neurons

    # get latent samples
    latents = []
    for i in range(num_sample):
        _, _, z_, _, _ = model(y_train, z=None)
        z_opt = torch.clone(z_.detach())
        z_opt.requires_grad = True
        latents.append(z_opt)

    optimizer = torch.optim.Adam(latents, lr=learning_rate)

    for i in range(num_iter):
        optimizer.zero_grad()
        loss = 0
        losses = []
        for j in range(num_sample):
            y_train_, _, _, _, _ = model(y_train, z=latents[j])
            losses.append(compute_poisson_loss(y_train, y_train_))
            loss = loss + losses[-1]
        loss.backward()
        optimizer.step()

        if not (i % 100):
            train_loss = torch.min(torch.tensor(losses)).item()
            losses = []
            for j in range(num_sample):
                _, y_test_, _, _, _ = model(y_train, z=latents[j])
                losses.append(compute_poisson_loss(y_test, y_test_))
            print('INFERENCE: iter %s, negLLH(train): %s, negLLH(test): %s' % (
                i, train_loss, torch.min(torch.tensor(losses)).item()))

    # get best latents of all samples
    losses = []
    for j in range(num_sample):
        y_train_, _, _, _, _ = model(y_train, z=latents[j])
        losses.append(compute_poisson_loss(y_train, y_train_))
    best_latents = latents[torch.argmin(torch.tensor(losses))]
    return best_latents