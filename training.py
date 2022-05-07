import torch
import time
import os
import numpy as np
from scipy.stats import pearsonr, spearmanr
from model import *
from hyperspherical_vae import log_likelihood


def check_grad(model, log_file):
    for name, param in model.named_parameters():
        if param.grad is not None:
            if torch.any(torch.isnan(param.grad)):
                print('NaN in gradient of %s, skipping step' % name, file=log_file)
                return False
    return True


class Trainer:
    def __init__(
            self,
            model,
            data_train,
            data_test,
            neurons_train_ind,  # boolean indices of training neurons
            mode='full',
            z_train=None,
            z_test=None,
            num_steps=50000,
            num_log_step=1000,
            batch_size=128,
            seed=23412521,
            learning_rate=3e-3,
            num_worse=5,  # if loss doesn't improve X times, stop.
            weight_kl=1e-2,
            weight_time=0,
            weight_entropy=0,
            log_dir='model_ckpt',
            log_training=False,
    ):
        if log_training:
            os.makedirs(log_dir, exist_ok=True)
            self.log_file = open(os.path.join(log_dir, "train_log.txt"), 'a', 1)
        else:
            self.log_file = None
        device = "cuda" if torch.cuda.is_available() else "cpu"
        torch.manual_seed(seed)
        self.model = model
        self.data_train = torch.Tensor(data_train).to(device)
        self.data_test = torch.Tensor(data_test).to(device)
        self.mode = mode
        if mode != 'full':
            self.z_train = torch.Tensor(z_train).to(device)
            self.z_test = torch.Tensor(z_test).to(device)
        else:
            self.z_test = None
        self.num_neuron_prediction = data_train.shape[0]  # all neurons
        self.neurons_train_ind = neurons_train_ind
        self.neurons_test_ind = np.logical_not(neurons_train_ind)
        self.num_neuron_inference = np.sum(neurons_train_ind)
        self.batch_size = batch_size
        self.seed = seed
        self.num_steps = num_steps
        self.num_log_step = num_log_step
        self.num_worse = num_worse
        self.weight_kl = weight_kl
        self.weight_time = weight_time
        self.weight_entropy = weight_entropy
        os.makedirs(log_dir, exist_ok=True)
        self.save_path = os.path.join(log_dir, 'model.pth')
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    def train(self):
        t0 = time.time()
        worse = 0  # counter for early stopping
        running_loss = 0.0
        loss_track = []
        for i in range(self.num_steps + 1):
            self.optimizer.zero_grad()
            np.random.seed(self.seed + i)
            ind = np.random.randint(self.data_train.shape[1] - self.batch_size)
            y_train = self.data_train[self.neurons_train_ind][:, ind:ind + self.batch_size]
            y_test = self.data_train[self.neurons_test_ind][:, ind:ind + self.batch_size]
            if self.mode != 'full':
                z = self.z_train[ind:ind + self.batch_size]
            else:
                z = None

            if self.model.latent_style == 'hack':
                y_train_, y_test_, z_, mu, logvar = self.model(y_train, z=z)
                if self.model.normalize_encodings:  # if normalized, take out of KL.
                    kld_loss = compute_kld_to_normal(torch.zeros_like(logvar), logvar)
                else:
                    kld_loss = compute_kld_to_normal(mu, logvar)
                poisson_loss = (
                    compute_poisson_loss(y_train, y_train_) +
                    compute_poisson_loss(y_test, y_test_)
                ) / 2
            elif self.model.latent_style == 'hyper':
                y_train_, y_test_, z_, mu, logvar, q_z, p_z = self.model(y_train, z=z)
                kld_loss = torch.distributions.kl.kl_divergence(
                    q_z, p_z).mean()  # mean over time and latents
                poisson_loss = (
                    torch.nn.PoissonNLLLoss(log_input=False, reduction='none')(
                        y_train_, y_train).mean() +  # mean over neurons and time
                    torch.nn.PoissonNLLLoss(log_input=False, reduction='none')(
                        y_test_, y_test).mean()  # mean over neurons and time
                ) / 2

            slowness_loss = compute_slowness_loss(mu)
            ensemble_weights = torch.nn.functional.softmax(
                self.model.ensemble_weights_train, dim=1)
            entropy = - torch.mean(
                ensemble_weights * torch.log(ensemble_weights + 1e-6))
            if self.mode == 'encoder':
                loss = torch.sum((angle2vector(z) - angle2vector(z_)) ** 2)
            else:
                loss = (
                    poisson_loss +
                    self.weight_kl * kld_loss +
                    self.weight_time * slowness_loss +
                    self.weight_entropy * entropy
                )
            loss.backward()
            if check_grad(self.model, self.log_file):
                self.optimizer.step()
            running_loss += loss.item()

            if i > 0 and not (i % self.num_log_step):
                self.model.eval()
                y_train = self.data_test[self.neurons_train_ind]
                y_test = self.data_test[self.neurons_test_ind]

                if self.model.latent_style == 'hack':
                    y_train_, y_test_, z_, mu, logvar = self.model(y_train, z=self.z_test)
                    if self.model.normalize_encodings:  # if normalized, take out of KL.
                        kld_loss = compute_kld_to_normal(torch.zeros_like(logvar), logvar)
                    else:
                        kld_loss = compute_kld_to_normal(mu, logvar)
                    poisson_loss_train = compute_poisson_loss(y_train, y_train_)
                    poisson_loss_test = compute_poisson_loss(y_test, y_test_)
                elif self.model.latent_style == 'hyper':
                    y_train_, y_test_, z_, mu, logvar, q_z, p_z = self.model(y_train, z=self.z_test)
                    kld_loss = torch.distributions.kl.kl_divergence(q_z, p_z).mean()
                    poisson_loss_train = torch.nn.PoissonNLLLoss(log_input=False, reduction='none')(
                        y_train_, y_train).mean()  # mean over neurons and time
                    poisson_loss_test = torch.nn.PoissonNLLLoss(log_input=False, reduction='none')(
                           y_test_, y_test).mean()  # mean over neurons and time

                slowness_loss = compute_slowness_loss(mu)
                if self.mode != 'full':
                    encoder_loss = torch.sum(
                        (angle2vector(z) - angle2vector(z_)) ** 2)
                    print('encoder_loss=', encoder_loss, file=self.log_file)
                corrs = []
                for j in range(y_test.shape[0]):
                    corrs.append(pearsonr(
                        y_test[j].detach().cpu().numpy(), y_test_[j].detach().cpu().numpy())[0])
                print('run=%s, running_loss=%.4e, negLLH_train=%.4e, negLLH_test=%.4e, KL=%.8e, '
                      'Slowness_loss=%.4e, corr=%.6f, H=%.4e, time=%.2f' % (
                    i, running_loss, poisson_loss_train.item(), poisson_loss_test.item(),
                    kld_loss.item(), slowness_loss.item(), np.nanmean(corrs), entropy.item(),
                    time.time() - t0
                ), file=self.log_file)

                # early stopping
                loss_track.append(running_loss)
                if loss_track[-1] > np.min(loss_track):
                    worse += 1
                    if worse > self.num_worse:
                        print('Early stopping at iteration', i, file=self.log_file)
                        break
                else:
                    # if improved, reset counter and save model
                    worse = 0  # reset counter
                    torch.save(self.model.state_dict(), self.save_path)
                running_loss = 0.0
                self.model.train()

        # after training, load best and set to eval mode.
        self.model.load_state_dict(torch.load(self.save_path))
        self.model.eval()

        # final evaluation
        ensemble_weights = torch.nn.functional.softmax(
            self.model.ensemble_weights_train, dim=1)
        entropy = - torch.mean(
            ensemble_weights * torch.log(ensemble_weights + 1e-6))
        y_train = self.data_test[self.neurons_train_ind]
        y_test = self.data_test[self.neurons_test_ind]
        if self.model.latent_style == 'hack':
            y_train_, y_test_, z_, mu, logvar = self.model(y_train, z=self.z_test)
            if self.model.normalize_encodings:  # if normalized, take out of KL.
                kld_loss = compute_kld_to_normal(torch.zeros_like(logvar), logvar)
            else:
                kld_loss = compute_kld_to_normal(mu, logvar)
            llh_train = compute_poisson_loss(y_train, y_train_)
            llh_test = compute_poisson_loss(y_test, y_test_)
        elif self.model.latent_style == 'hyper':
            y_train_, y_test_, z_, mu, logvar, q_z, p_z = self.model(y_train, z=self.z_test)
            z_ = angle2vector(z_)
            kld_loss = torch.distributions.kl.kl_divergence(q_z, p_z).mean()
            #llh_train = log_likelihood(y_train, y_train_, z_, q_z, p_z)
            #llh_test = log_likelihood(y_test, y_test_, z_, q_z, p_z)
            llh_train = compute_poisson_loss(y_train, y_train_)
            llh_test = compute_poisson_loss(y_test, y_test_)
        slowness_loss = compute_slowness_loss(mu)
        corrs = []
        for j in range(y_test.shape[0]):
            corrs.append(pearsonr(
                y_test[j].detach().cpu().numpy(), y_test_[j].detach().cpu().numpy())[0])

        return (i, running_loss, llh_train.item(), llh_test.item(), ensemble_weights,
                kld_loss.item(), slowness_loss.item(), np.nanmean(corrs), entropy.item())