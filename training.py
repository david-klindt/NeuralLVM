import torch
import time, os
from scipy.stats import pearsonr, spearmanr
from model import *


class Trainer:
    def __init__(
            self,
            model,
            data_train,
            data_test,
            neurons_train_ind,  # logical indices of training neurons
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
    ):
        device = "cuda" if torch.cuda.is_available() else "cpu"
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
            y = self.data_train[:, ind:ind + self.batch_size]
            if self.mode is not 'full':
                z = self.z_train[ind:ind + self.batch_size]
            else:
                z = None
            y_, z_, mu, logvar = self.model(y[self.neurons_train_ind], z=z)
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
                self.model.eval()
                y_, z_, mu, logvar = self.model(
                    self.data_test[self.neurons_train_ind], z=self.z_test)
                slowness_loss = compute_slowness_loss(mu)
                if self.model.normalize_encodings:
                    mu = torch.zeros_like(logvar)
                kld_loss = compute_kld_to_normal(mu, logvar)
                poisson_loss = compute_poisson_loss(
                    self.data_test[self.neurons_test_ind],
                    y_[self.neurons_test_ind]
                )
                if self.mode is not 'full':
                    encoder_loss = torch.sum(
                        (angle2vector(z) - angle2vector(z_)) ** 2)
                    print('encoder_loss=', encoder_loss)
                corrs = []
                for j in np.where(self.neurons_test_ind)[0]:
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
                    # if improved, reset counter and save model
                    worse = 0  # reset counter
                    torch.save(self.model.state_dict(), self.save_path)
                running_loss = 0.0
                self.model.train()

        # after training, load best and set to eval mode.
        self.model.load_state_dict(torch.load(self.save_path))
        self.model.eval()