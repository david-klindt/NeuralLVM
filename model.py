import torch
from torch.autograd import Variable
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"
print('Running on', device)


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
    angle.append(torch.atan2(vector[:, i*2], vector[:, i*2+1]))
  return torch.stack(angle, 1)


def sum_pairs(x):
  sum = []
  for i in range(x.shape[1] // 2):
    sum.append(torch.sum(x[:, i*2:(i+1)*2], dim=1))
  return torch.stack(sum, 1)


def reparameterize(mu, logvar):
    std = logvar.div(2).exp()
    eps = Variable(std.data.new(std.size()).normal_())
    return mu + std * eps


def compute_kld_to_normal(mu, logvar):
    """Computes the KL(q|p) between variational posterior q and standard
    normal p."""
    return torch.mean(-0.5 * (1 + logvar - mu ** 2 - logvar.exp()))


def compute_kld(q1_mu, q1_logvar, q0_mu, q0_logvar):
    """Computes the KL(q_t|q_{t-1}) between variational posterior q_t ("q1")
    and variational posterior q_{t-1} ("q0")."""
    KL = (q0_logvar - q1_logvar) / 2
    KL = KL + (q1_logvar.exp() + (q0_mu - q1_mu)**2) / (2 * q0_logvar.exp())
    KL = torch.mean(KL - 1 / 2)
    return KL


def compute_slowness_loss(mu):
    """compute squared difference over 2nd dimension, i.e., time."""
    return torch.mean((mu[:, 1:] - mu[:, -1])**2)


def compute_poisson_loss(y, y_):
    return torch.mean(y_ - y * torch.log(y_ + 1e-9))


def torch_normalize(y):
    normed = y - torch.mean(y, dim=1, keepdim=True)
    norm = torch.linalg.norm(normed, dim=1, keepdim=True) + 1e-6
    return normed / norm


def correlation_loss(y, y_):
    return torch.mean(
        torch.sum(torch_normalize(y) * torch_normalize(y_), dim=1))


class FeatureBasis(torch.nn.Module):
    def __init__(
            self,
            num_neuron,
            feature_type='bump',  # {'bump', 'shared', 'separate'}
            num_basis=3,
            latent_dim=2,
            tuning_width=10.0,
            nonlinearity='exp',
            variance=None,
            seed=345978,
    ):
        super(FeatureBasis, self).__init__()
        self.feature_type = feature_type
        self.latent_dim = latent_dim
        self.tuning_width = tuning_width  # of single bump
        self.nonlinearity = nonlinearity
        if variance is None:  # of feature basis
            variance = torch.ones(1) * 4 * np.pi / num_basis
        self.variance = torch.nn.Parameter(
            variance, requires_grad=False)#feature_type.endswith('flex'))
        torch.manual_seed(seed)

        if feature_type == 'bump':
            self.log_tuning_width = torch.nn.Parameter(
                torch.ones(1) * np.log(tuning_width),
                requires_grad=True
            )
        else:
            # build grid of num_basis**latent_dim centers
            means = torch.linspace(0, 2 * np.pi, num_basis + 1)[:-1]
            means = torch.meshgrid([means for _ in range(latent_dim)])
            means = torch.stack(means, 0).view(latent_dim, -1).T
            # shape: num_basis**latent_dim x latent_dim
            self.means = torch.nn.Parameter(
                means, requires_grad=feature_type.endswith('flex'))
            if feature_type.startswith('shared'):
                self.coeffs = torch.nn.Parameter(
                    torch.randn(num_basis ** latent_dim, 1),
                    requires_grad=True
                )
            elif feature_type.startswith('separate'):
                self.coeffs = torch.nn.Parameter(
                    torch.randn(num_basis ** latent_dim, num_neuron),
                    requires_grad=True
                )

    def forward(self, z, receptive_field_centers, is_test=False):
        # dimension names: B=Batch, L=Length, N=Neurons, H=num_basis**latent_dim
        # E=num_ensemble, D=latent_dim, V=angle as vector (2D)

        if is_test:
            z = z.detach()
            variance = self.variance.detach()
            if self.feature_type is not 'bump':
                coeffs = self.coeffs.detach()
                means = self.means.detach()
            else:
                log_tuning_width = self.log_tuning_width.detach()
        else:
            variance = self.variance
            if self.feature_type is not 'bump':
                coeffs = self.coeffs
                means = self.means
            else:
                log_tuning_width = self.log_tuning_width

        z_vector = angle2vector(z)  # B x L x E x D x V
        rf_vector = angle2vector(receptive_field_centers)  # N x E x D x V

        if self.feature_type == 'bump':
            z_vector = z_vector[:, :, None]  # B x L x 1 x E x D x V
            rf_vector = rf_vector[None, None]  # 1 x 1 x N x E x D x V
            dist = torch.sum(  # B x L x N x E
                (z_vector - rf_vector) ** 2, dim=(4, 5))
            response = - dist / torch.exp(log_tuning_width)
        else:
            means = angle2vector(means[:, None, None])  # H x 1 x 1 x D x V
            means_per_neuron = means - rf_vector[None]  # H x N x E x D x V
            # make dist: B x L x H x N x E x D x V
            dist = z_vector[:, :, None, None] - means_per_neuron[None, None]
            dist = torch.sum(dist ** 2, dim=(5, 6))  # B x L x H x N x E
            response = torch.exp(- dist / variance)
            # coeffs shape: 1 x 1 x H x {1 if shared, else N} x 1
            coeffs = coeffs[None, None, :, :, None]
            response = torch.sum(response * coeffs, dim=2)  # B x L x N x E

        if self.nonlinearity == 'exp':
            response = torch.exp(response)
        elif self.nonlinearity == 'softplus':
            response = torch.log(torch.exp(response) + 1)

        return response


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
            requires_grad=not(feature_type.startswith('separate'))
        )
        self.receptive_fields_test = torch.nn.Parameter(
            # initialize randomly in [-pi, pi]
            - np.pi + 2 * np.pi * torch.rand(
                num_neuron_test, num_ensemble, latent_dim),
            requires_grad=not(feature_type.startswith('separate'))
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
        response_train = self.compute_responses(
            self.ensemble_weights_train,
            self.log_final_scale_train,
            self.feature_basis(z, self.receptive_fields_train, is_test=False),
            input_shape
        )
        if self.feature_type.startswith('separate'):
            feature_basis_test = self.feature_basis_test
        else:
            feature_basis_test = self.feature_basis
        response_test = self.compute_responses(
            self.ensemble_weights_test,
            self.log_final_scale_test,
            feature_basis_test(z, self.receptive_fields_test, is_test=True),
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