import torch
import time
import os
import numpy as np
from scipy.stats import pearsonr, spearmanr
from NeuralLVM.code.utils import compute_slowness_loss
from NeuralLVM.code.utils import angle2vector
from NeuralLVM.code.utils import check_grad



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
            num_steps=5000,
            num_log_step=50,
            batch_size=16,
            batch_length=128,
            seed=23412521,
            learning_rate=3e-3,
            num_worse=50,  # if loss doesn't improve X times, stop.
            weight_kl=1e-6,
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
        self.batch_length = batch_length
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
            batch_indices = np.random.choice(self.data_train.shape[1] - self.batch_length, self.batch_size)
            y_train, y_test = [], []
            for ind in batch_indices:
                y_train.append(self.data_train[self.neurons_train_ind][:, ind:ind + self.batch_length])
                y_test.append(self.data_train[self.neurons_test_ind][:, ind:ind + self.batch_length])
            y_train = torch.stack(y_train, 0)
            y_test = torch.stack(y_test, 0)
            if self.mode != 'full':
                z = self.z_train[ind:ind + self.batch_length]
            else:
                z = None

            output = self.model(y_train, z=z)

            kld_loss, slowness_loss, encoder_loss = torch.zeros(1), torch.zeros(1), torch.zeros(1)
            for j, m in enumerate(self.model.latent_manifolds):
                # sum over time and neurons, mean over batch (same below for Poisson LLH)
                kld_loss = kld_loss + torch.distributions.kl.kl_divergence(
                    output['q_z'][j], output['p_z'][j]).sum((1, 2)).mean()
                # but see (https://github.com/nicola-decao/s-vae-pytorch/blob/master/examples/mnist.py#L144)
                slowness_loss = slowness_loss + compute_slowness_loss(output['mean'][j])
                if z is not None:
                    encoder_loss = encoder_loss + torch.sum((angle2vector(z) - angle2vector(output['z'][i])) ** 2)

            poisson_loss = (
                torch.nn.PoissonNLLLoss(log_input=False, reduction='none')(
                    output['responses_train'], y_train).sum((1, 2)).mean() +
                torch.nn.PoissonNLLLoss(log_input=False, reduction='none')(
                    output['responses_test'], y_test).sum((1, 2)).mean()
            ) / 2

            ensemble_weights = torch.nn.functional.softmax(self.model.decoder.ensemble_weights_train, dim=1)
            entropy = - torch.mean(ensemble_weights * torch.log(ensemble_weights + 1e-6))

            if self.mode == 'encoder':
                loss = encoder_loss
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
            if not np.isnan(loss.item()) and not np.isinf(loss.item()):
                running_loss += loss.item()

            if i > 0 and not (i % self.num_log_step):
                self.model.eval()
                y_train = self.data_test[self.neurons_train_ind][None]
                y_test = self.data_test[self.neurons_test_ind][None]
                if self.mode != 'full':
                    z = self.z_test[ind:ind + self.batch_length]
                else:
                    z = None

                output = self.model(y_train, z=z)

                kld_loss, slowness_loss, encoder_loss = torch.zeros(1), torch.zeros(1), torch.zeros(1)
                for j, m in enumerate(self.model.latent_manifolds):
                    # sum over time and neurons, mean over batch (same below for Poisson LLH)
                    kld_loss = kld_loss + torch.distributions.kl.kl_divergence(
                        output['q_z'][j], output['p_z'][j]).sum((1, 2)).mean()
                    # but see (https://github.com/nicola-decao/s-vae-pytorch/blob/master/examples/mnist.py#L144)
                    slowness_loss = slowness_loss + compute_slowness_loss(output['mean'][j])
                    if z is not None:
                        encoder_loss = encoder_loss + torch.sum((angle2vector(z) - angle2vector(output['z'][i])) ** 2)

                poisson_loss_train = torch.nn.PoissonNLLLoss(log_input=False, reduction='none')(
                                           output['responses_train'], y_train).sum((1, 2)).mean()
                poisson_loss_test = torch.nn.PoissonNLLLoss(log_input=False, reduction='none')(
                                           output['responses_test'], y_test).sum((1, 2)).mean()

                ensemble_weights = torch.nn.functional.softmax(self.model.decoder.ensemble_weights_train, dim=1)
                entropy = - torch.mean(ensemble_weights * torch.log(ensemble_weights + 1e-6))

                corrs = []
                y = y_test.detach().cpu().numpy()[0]
                y_ = output['responses_test'].detach().cpu().numpy()[0]
                for j in range(y.shape[0]):
                    corrs.append(pearsonr(y[j], y_[j])[0])

                print('run=%s, running_loss=%.4e, negLLH_train=%.4e, negLLH_test=%.4e, KL=%.4e, '
                      'Slowness_loss=%.4e, encoder_loss=%.4e, corr=%.6f, H=%.4e, time=%.2f' % (
                      i, running_loss, poisson_loss_train.item(), poisson_loss_test.item(),
                      kld_loss.item(), slowness_loss.item(), encoder_loss.item(), np.nanmean(corrs),
                      entropy.item(), time.time() - t0), file=self.log_file)

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

        # Final eval
        y_train = self.data_test[self.neurons_train_ind]
        y_test = self.data_test[self.neurons_test_ind]

        output = self.model(y_train, z=z)

        kld_loss, slowness_loss, encoder_loss = 0.0, 0.0, 0.0
        for j, m in enumerate(self.model.latent_manifolds):
            # sum over time and neurons, mean over batch (same below for Poisson LLH)
            kld_loss = kld_loss + torch.distributions.kl.kl_divergence(
                output['q_z'][j], output['p_z'][j]).sum((1, 2)).mean()
            # but see (https://github.com/nicola-decao/s-vae-pytorch/blob/master/examples/mnist.py#L144)
            slowness_loss = slowness_loss + compute_slowness_loss(output['mean'][j])
            encoder_loss = encoder_loss + torch.sum((angle2vector(z) - angle2vector(output['z'][i])) ** 2)

        poisson_loss_train = torch.nn.PoissonNLLLoss(log_input=False, reduction='none')(
            output['responses_train'], y_train).sum((1, 2)).mean()
        poisson_loss_test = torch.nn.PoissonNLLLoss(log_input=False, reduction='none')(
            output['responses_test'], y_test).sum((1, 2)).mean()

        ensemble_weights = torch.nn.functional.softmax(self.model.decoder.ensemble_weights_train, dim=1)
        entropy = - torch.mean(ensemble_weights * torch.log(ensemble_weights + 1e-6))

        corrs = []
        y = y_test.detach().cpu().numpy()
        y_ = output['responses_test'].detach().cpu().numpy()
        for j in range(y_test.shape[0]):
            corrs.append(pearsonr(y[j], y_[j])[0])

        print('\nFinal Performance:\n',
            'run=%s, running_loss=%.4e, negLLH_train=%.4e, negLLH_test=%.4e, KL=%.4e, '
            'Slowness_loss=%.4e, encoder_loss=%.4e, corr=%.6f, H=%.4e, time=%.2f' % (
            i, running_loss, poisson_loss_train.item(), poisson_loss_test.item(),
            kld_loss.item(), slowness_loss.item(), encoder_loss.item(), np.nanmean(corrs),
            entropy.item(), time.time() - t0), file=self.log_file
        )
        output['run'] = i
        output['running_loss'] = running_loss.item()
        output['negLLH_train'] = poisson_loss_train.item()
        output['negLLH_test'] = poisson_loss_test.item()
        output['KL'] = kld_loss.item()
        output['Slowness_loss'] = slowness_loss.item()
        output['encoder_loss'] = encoder_loss.item()
        output['corrs'] = corrs
        output['H'] = entropy.item()
        output['time'] = time.time() - t0

        return output