import torch
from torch.autograd import Variable
import numpy as np
import time
from scipy.stats import pearsonr, spearmanr
import matplotlib.pyplot as plt
from sklearn.feature_selection import mutual_info_regression
from gtda.homology import VietorisRipsPersistence
from gtda.plotting import plot_diagram

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


def torch_circular_gp(num_sample, num_dim, smoothness):
    z = torch.randn(num_sample, num_dim) / smoothness
    z = torch.cumsum(z, 0)
    return vector2angle(angle2vector(z))


class LatentVariableModel(torch.nn.Module):
    def __init__(
            self,
            num_neuron,
            num_hidden=256,
            num_ensemble=2,
            latent_dim=2,
            seed=2093857,
            tuning_width=2.0,
            nonlinearity='exp',
            kernel_size=1,
            normalize_encodings=True,
            feature_type='bump',  # {'bump', 'shared', 'separate'}
            num_feature_basis=3,  # careful this scales badly with latent_dim!
    ):
        super(LatentVariableModel, self).__init__()
        self.num_neuron = num_neuron
        self.num_ensemble = num_ensemble
        self.latent_dim = latent_dim
        self.nonlinearity = nonlinearity
        self.kernel_size = kernel_size
        self.normalize_encodings = normalize_encodings
        self.feature_type = feature_type

        torch.manual_seed(seed)
        self.receptive_fields = torch.nn.Parameter(
            #torch.randn(num_neuron, num_ensemble, latent_dim),
            torch.zeros(num_neuron, num_ensemble, latent_dim),
            requires_grad=feature_type is not 'separate'
        )
        self.ensemble_weights = torch.nn.Parameter(
            torch.randn(num_neuron, num_ensemble),
            requires_grad=True
        )
        self.final_scale = torch.nn.Parameter(
            torch.randn(num_neuron),
            requires_grad=True
        )
        self.encoder = torch.nn.Sequential(
            torch.nn.Conv1d(
                in_channels=num_neuron,
                out_channels=num_neuron,
                kernel_size=kernel_size,
                padding='same',
                groups=num_neuron
            ),
            torch.nn.Conv1d(
                in_channels=num_neuron,
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
            num_neuron,
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
        batch_size, num_neuron, length = x.shape
        assert num_neuron == self.num_neuron

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
        response = self.feature_basis(z, self.receptive_fields)

        ensemble_weights = torch.nn.functional.softmax(  # 1 x 1 x N x E
            self.ensemble_weights, dim=1)[None, None]
        responses = torch.sum(  # B x L x N
            ensemble_weights * response, dim=3)
        responses = responses * torch.exp(self.final_scale[None, None])
        responses = responses.permute(0, 2, 1)  # B x N x L
        if len(input_shape) == 2:
            # if input had no batch dimension, remove this again
            responses = responses[0]

        return responses, z, mu, logvar


class FeatureBasis(torch.nn.Module):
    def __init__(
            self,
            num_neuron,
            feature_type='bump',  # {'bump', 'shared', 'separate'}
            num_basis=3,
            latent_dim=2,
            tuning_width=2.0,
            nonlinearity='exp',
            variance=None,
            seed=345978,
    ):
        super(FeatureBasis, self).__init__()
        self.feature_type = feature_type
        self.latent_dim = latent_dim
        self.tuning_width = tuning_width  # of single bump
        self.nonlinearity = nonlinearity
        if variance is None:
            variance = 4 * np.pi / num_basis
        self.variance = variance  # of feature basis
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
            self.means = torch.nn.Parameter(means, requires_grad=False)
            if feature_type == 'shared':
                self.coeffs = torch.nn.Parameter(
                    torch.randn(num_basis ** latent_dim, 1),
                    requires_grad=True
                )
            elif feature_type == 'separate':
                self.coeffs = torch.nn.Parameter(
                    torch.randn(num_basis ** latent_dim, num_neuron),
                    requires_grad=True
                )

    def forward(self, z, receptive_field_centers):
        # dimension names: B=Batch, L=Length, N=Neurons, H=num_basis**latent_dim
        # E=num_ensemble, D=latent_dim, V=angle as vector (2D)

        z_vector = angle2vector(z)  # B x L x E x D x V
        rf_vector = angle2vector(receptive_field_centers)  # N x E x D x V

        if self.feature_type == 'bump':
            z_vector = z_vector[:, :, None]  # B x L x 1 x E x D x V
            rf_vector = rf_vector[None, None]  # 1 x 1 x N x E x D x V
            dist = torch.sum(  # B x L x N x E
                (z_vector - rf_vector) ** 2, dim=(4, 5))
            response = - dist / torch.exp(self.log_tuning_width)
        else:
            means = angle2vector(self.means[:, None, None])  # H x 1 x 1 x D x V
            means_per_neuron = means - rf_vector[None]  # H x N x E x D x V
            # make dist: B x L x H x N x E x D x V
            dist = z_vector[:, :, None, None] - means_per_neuron[None, None]
            dist = torch.sum(dist ** 2, dim=(5, 6))  # B x L x H x N x E
            response = torch.exp(- dist / self.variance)
            # coeffs shape: 1 x 1 x H x {1 if shared, else N} x 1
            coeffs = self.coeffs[None, None, :, :, None]
            response = torch.sum(response * coeffs, dim=2)  # B x L x N x E

        if self.nonlinearity == 'exp':
            response = torch.exp(response)
        elif self.nonlinearity == 'softplus':
            response = torch.log(torch.exp(response) + 1)

        return response


class Trainer:
    def __init__(
            self,
            model,
            data_train,
            data_test,
            mode='full',
            z_train=None,
            z_test=None,
            num_steps=50000,
            num_log_step=1000,
            batch_size=1024,
            seed=23412521,
            learning_rate=1e-3,
            num_worse=5,  # if loss doesn't improve X times, stop.
            weight_kl=1e-2,
            weight_time=0,
            weight_entropy=0,
    ):
        torch.manual_seed(seed)
        self.model = model
        self.data_train = torch.Tensor(data_train).to(device)
        self.data_test = torch.Tensor(data_test).to(device)
        self.mode = mode
        if mode is not 'full':
            self.z_train = torch.Tensor(z_train).to(device)
            self.z_test = torch.Tensor(z_test).to(device)
        else:
            self.z_test = None
        self.num_neurons = data_train.shape[0]
        self.batch_size = batch_size
        self.seed = seed
        self.num_steps = num_steps
        self.num_log_step = num_log_step
        self.num_worse = num_worse
        self.weight_kl = weight_kl
        self.weight_time = weight_time
        self.weight_entropy = weight_entropy
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    def train(self):
        t0 = time.time()
        worse = 0  # counter for early stopping
        running_loss = 0.0
        loss_track = []
        for i in range(self.num_steps + 1):
            self.optimizer.zero_grad()
            np.random.seed(i + 1)
            ind = np.random.randint(self.data_train.shape[1])
            y = self.data_train[:, ind:ind + self.batch_size]
            if self.mode is not 'full':
                z = self.z_train[ind:ind + self.batch_size]
            else:
                z = None
            y_, z_, mu, logvar = self.model(y, z=z)
            slowness_loss = compute_slowness_loss(mu)
            if self.model.normalize_encodings:  # if normalized, take out of KL.
                mu = torch.zeros_like(logvar)
            kld_loss = compute_kld_to_normal(mu, logvar)
            poisson_loss = compute_poisson_loss(y, y_)
            ensemble_weights = torch.nn.functional.softmax(
                self.model.ensemble_weights, dim=1)
            entropy = - torch.mean(
                ensemble_weights * torch.log(ensemble_weights + 1e-6))
            if self.mode is 'encoder':
                loss = torch.sum((angle2vector(z) - angle2vector(z_)) ** 2)
            else:
                loss = (
                    poisson_loss +
                    self.weight_kl * kld_loss +
                    self.weight_time * slowness_loss +
                    self.weight_entropy * entropy
                )
            loss.backward()
            self.optimizer.step()
            running_loss += loss.item()

            if i > 0 and not (i % self.num_log_step):
                y_, z_, mu, logvar = self.model(self.data_test, z=self.z_test)
                slowness_loss = compute_slowness_loss(mu)
                if self.model.normalize_encodings:
                    mu = torch.zeros_like(logvar)
                kld_loss = compute_kld_to_normal(mu, logvar)
                poisson_loss = compute_poisson_loss(self.data_test, y_)
                if self.mode is not 'full':
                    encoder_loss = torch.sum(
                        (angle2vector(z) - angle2vector(z_)) ** 2)
                    print('encoder_loss=', encoder_loss)
                corrs = []
                for j in range(y_.shape[0]):
                    corrs.append(pearsonr(y_[j].detach().cpu().numpy(),
                                          self.data_test[j].detach().cpu().numpy())[0])
                print('run=%s, running_loss=%.4e, negLLH=%.4e, KL_normal=%.4e, '
                      'Slowness_loss=%.4e, corr=%.6f, H=%.4e, time=%.2f' % (
                    i, running_loss, poisson_loss.item(), kld_loss.item(),
                    slowness_loss.item(), np.nanmean(corrs), entropy.item(),
                    time.time() - t0
                ))

                # early stopping
                loss_track.append(running_loss)
                if loss_track[-1] > np.min(loss_track):
                    worse += 1
                    if worse > self.num_worse:
                        print('Early stopping at iteration', i)
                        break
                else:
                    worse = 0  # reset counter
                    # ToDo: save best weights
                    #self.weights = weights.detach().cpu().numpy().copy()
                    #np.save(self.save_file, self.weights)
                running_loss = 0.0


### Simulation Experiments ###
class StochasticNeurons(torch.nn.Module):
    def __init__(
            self,
            N,
            num_ensemble=2,
            latent_dim=2,
            seed=304857,
            noise=False,
            tuning_width=2.0,
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


def test_simulation():
    num_ensemble = 2
    num_neuron = 2
    model = StochasticNeurons(num_neuron, num_ensemble=num_ensemble)

    print('clean')
    plt.figure(figsize=(15, 3))
    for i in range(num_ensemble*2):
      plt.subplot(1, num_ensemble*2, i+1)
      inputs = torch.zeros((100, num_ensemble*2))
      inputs[:, i] = torch.linspace(0, 2*np.pi, 100)
      responses = model(inputs)
      plt.plot(responses.detach().numpy().T)
      plt.legend(np.arange(num_neuron * num_ensemble))
    plt.show()


    print('noisy')
    model.noise = True
    plt.figure(figsize=(15, 3))
    for i in range(num_ensemble*2):
      plt.subplot(1, num_ensemble*2, i+1)
      inputs = torch.zeros((100, num_ensemble*2))
      inputs[:, i] = torch.linspace(0, 2*np.pi, 100)
      responses = model(inputs)
      plt.plot(responses.detach().numpy().T)
      plt.legend(np.arange(num_neuron * num_ensemble))
    plt.show()

    # Persistence
    num_neuron = 50
    D = 500
    model = StochasticNeurons(num_neuron, num_ensemble=num_ensemble, noise=True)
    responses = model(torch.rand(D, num_ensemble * 2) * 2 * np.pi)

    # all
    t0 = time.time()
    VR = VietorisRipsPersistence(
        homology_dimensions=[0, 1, 2],
    )
    diagrams0 = VR.fit_transform([responses.detach().numpy().T])
    print(diagrams0.shape, time.time() - t0)
    fig0 = plot_diagram(diagrams0[0])
    fig0.show()

    # per ensemble
    for i in range(num_ensemble):
        t0 = time.time()
        VR = VietorisRipsPersistence(
            homology_dimensions=[0, 1, 2],
        )
        diagrams0 = VR.fit_transform(
            [responses[i * num_neuron:(i + 1) * num_neuron].detach().numpy().T])
        print(i, diagrams0.shape, time.time() - t0)
        fig0 = plot_diagram(diagrams0[0])
        fig0.show()


def test_training(num_ensemble=3, num_neuron=50, latent_dim=2, z_smoothness=3,
                  num_sample=100000, num_test=10000, feature_type='bump'):
    model = StochasticNeurons(
        num_neuron, num_ensemble=num_ensemble, noise=True, latent_dim=latent_dim).to(device)
    ensembler = LatentVariableModel(
        num_neuron * num_ensemble,
        num_hidden=256,
        num_ensemble=num_ensemble,
        latent_dim=latent_dim,
        seed=234587,
        tuning_width=2.0,
        nonlinearity='exp',
        kernel_size=9,
        feature_type=feature_type,
    ).to(device)
    print('model', ensembler)
    print('number of trainable parameters in model:', (count_parameters(
        ensembler)))

    if z_smoothness > 0:  # gp latents
        z_train = torch_circular_gp(num_sample, latent_dim * num_ensemble, z_smoothness)
        z_test = torch_circular_gp(num_test, latent_dim * num_ensemble, z_smoothness)
    else:  # iid latents
        z_train = torch.rand(num_sample, latent_dim * num_ensemble) * 2 * np.pi
        z_test = torch.rand(num_test, latent_dim * num_ensemble).to(device) * 2 * np.pi

    z_train = z_train.to(device)
    z_test = z_test.to(device)
    data_train = model(z_train).detach()
    model.noise = False
    data_test = model(z_test).detach()
    model.noise = True

    trainer = Trainer(
        model=ensembler,
        data_train=data_train.cpu().numpy(),
        data_test=data_test.cpu().numpy(),
        mode='full',
        z_train=None,
        z_test=None,
        num_steps=20000,
        batch_size=1024,
        seed=923683,
        learning_rate=3e-3
    )
    trainer.train()
    analysis(ensembler, model, trainer, z_test)


def analysis(ensembler, model, trainer, z_test):
    num_ensemble = ensembler.num_ensemble
    latent_dim = ensembler.latent_dim
    y_, z_, mu, logvar = ensembler(trainer.data_test)
    z_ = z_.view(z_test.shape)
    logvar = logvar.view(z_test.shape)

    # plot ensemble_weights
    ensemble_weights = torch.nn.functional.softmax(
        ensembler.ensemble_weights, dim=1).detach().cpu().numpy()
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
        plt.plot(trainer.data_test[i * 25, :200].detach().cpu().numpy())
        plt.plot(y_[i * 25, :200].detach().cpu().numpy())
        plt.legend(['true', 'predicted'])
        plt.title('Responses')
    plt.tight_layout()
    plt.show()

    # RF comparisons
    plt.figure(figsize=(18, 6))
    true_rfs = model.receptive_fields.detach().cpu().numpy()
    learned_rfs = ensembler.receptive_fields.detach().cpu().numpy().reshape(
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
