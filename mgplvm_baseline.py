import functools
from absl import app
from absl import flags
import numpy as np
import mgplvm as mgp
from data import get_data
import matplotlib.pyplot as plt
import torch
import pickle
import os
import logging

FLAGS = flags.FLAGS

flags.DEFINE_integer("n_z", 20, "number of inducing points")
flags.DEFINE_integer("random_seed", 100_000_000, "random seed")
flags.DEFINE_string("model_type", "cosyne",
                    "`orig` (Gaussian + uniform) or `cosyne` (Poisson + AR)")
flags.DEFINE_string("results_dir", "/scratches/cblgpu03/tck29/neurallvm",
                    "results directory")
flags.DEFINE_string("device", "cuda", "device: cuda or cpu")

d = 1  #model dimensionality
n_samples = 1  #number of trials
reps = 5
Ntrains = [20, 50, 100, 200]
Ttrains = [50, 100, 200, 500]


def add_suffix(rep, num_neuron_train, len_data_train, s):
    return f"{s}_{rep}_N{num_neuron_train}_T{len_data_train}"


def eval_lats(pred, target):
    errs = np.zeros((2, 1000))
    for i in range(2):
        for i_s, s in enumerate(np.linspace(0, 2 * np.pi, 1000)):
            newpred = 2 * (0.5 - i) * pred + s
            errs[i, i_s] = np.mean(np.arccos(np.cos(newpred - target)))
    return np.amin(errs)


lat_perfs = np.zeros((reps, len(Ntrains), len(Ttrains)))
pred_perfs = np.zeros((reps, len(Ntrains), len(Ttrains)))

num_neuron_test = 80
len_data_test = 500


def run(rep):
    index = rep
    for iN, num_neuron_train in enumerate(Ntrains):
        for iT, len_data_train in enumerate(Ttrains):
            suffix = functools.partial(add_suffix, rep, num_neuron_train,
                                       len_data_train)
            num_neuron_tot = num_neuron_train + num_neuron_test
            len_data_tot = len_data_train + len_data_test

            y_train, z_train, y_test, z_test, rf, neurons_train_ind = get_data(
                num_neuron_train, num_neuron_test, len_data_train,
                len_data_test, index, FLAGS.random_seed)

            logging.info(f"y_train: {y_train.shape}")  #Ntot x Ttrain
            logging.info(f"z_train: {z_train.shape}")  #d x Ttrain
            logging.info(f"y_test: {y_test.shape}")  #Ntot x Ttest
            logging.info(f"z_test:' {z_test.shape}")  #d x Ttest
            logging.info(f"N_train: {num_neuron_train}")

            ## plot data
            plt.figure()
            plt.imshow(y_train, cmap="Greys", aspect="auto")
            plt.xlabel("time")
            plt.ylabel("neuron")
            plt.xlim(0, len_data_train)
            plt.ylim(0, num_neuron_tot)
            plt.xticks([])
            plt.yticks([])
            plt.savefig(suffix(f"{FLAGS.results_dir}/raw_data") + ".png")

            #%% construct mgplvm model
            torch.cuda.empty_cache()

            if FLAGS.model_type == 'orig':  #Gaussian noise and uniform prior
                Y1 = np.sqrt(y_train)  #sqrt transform
                mu, sig = np.mean(Y1, axis=1,
                                  keepdims=True), np.std(Y1,
                                                         axis=1,
                                                         keepdims=True)
                sig = np.maximum(sig, 1e-3)
                Y1 = ((Y1 - mu) / sig)
                Y2 = ((np.sqrt(y_test) - mu) / sig)  #[neurons_train_ind, :]
            else:  #Poisson noise and AR prior
                Y1 = y_train
                Y2 = y_test  #[neurons_train_ind, :]

            #wrangle the data into the right form and put on gpu
            Ttrain, Ttest = np.arange(len_data_train), np.arange(
                len_data_train, len_data_train + len_data_test)
            Y = np.concatenate([Y1, Y2], axis=1)
            Y = Y[None, ...]
            Y1 = Y1[None, ...]
            z_tot = np.concatenate([z_train, z_test], axis=0)
            data = torch.tensor(Y).to(FLAGS.device)
            n_samples, n, m = Y.shape

            manif = mgp.manifolds.Torus(m, d)  # latent distribution manifold
            lat_dist = mgp.rdist.ReLie(
                manif,
                m,
                n_samples,
                Y=Y,
                initialization='random',
                sigma=np.pi / 2,
                diagonal=True)  # construct ReLie distribution
            kernel = mgp.kernels.QuadExp(
                n,
                manif.distance,
                Y=Y1,
                ell=np.ones(n) * 1,
            )  # Use an exponential quadratic (RBF) kernel

            if FLAGS.model_type == 'orig':  #Gaussian noise and uniform prior
                lik = mgp.likelihoods.Gaussian(n, Y=Y1,
                                               d=d)  # Gaussian likelihood
                lprior = mgp.lpriors.Uniform(
                    manif)  # Prior on the manifold distribution
            else:  #Poisson noise and AR prior
                lik = mgp.likelihoods.Poisson(n)  #Poisson likelihood
                lprior = mgp.lpriors.Brownian(manif,
                                              fixed_brownian_c=True,
                                              fixed_brownian_eta=False,
                                              brownian_eta=torch.ones(d) *
                                              np.pi**2)

            z = manif.inducing_points(n, FLAGS.n_z)  # build inducing points
            model = mgp.models.SvgpLvm(n,
                                       m,
                                       n_samples,
                                       z,
                                       kernel,
                                       lik,
                                       lat_dist,
                                       lprior,
                                       whiten=True)
            model = model.to(FLAGS.device)  #build full model

            def cb(mod, i, loss):
                """here we construct an (optional) function that helps us keep track of the training"""
                if i in [0, 50, 100, 150, 300, 500, 1000, 1500,
                         1999]:  #iterations to plot
                    X = mod.lat_dist.prms[0].detach().cpu().numpy()[0, ...]
                    X = X % (2 * np.pi)
                    plt.figure(figsize=(4, 4))
                    plt.xlim(0, 2 * np.pi)
                    plt.ylim(0, 2 * np.pi)
                    plt.xlabel("true latents")
                    plt.ylabel("model latents")
                    plt.scatter(z_tot[:, 0], X[:, 0], color='k')
                    plt.title('iter = ' + str(i))
                    plt.savefig(
                        suffix(f"{FLAGS.results_dir}/latents{i}") + ".png")

            train_ps = mgp.crossval.training_params(max_steps=2000,
                                                    n_mc=50,
                                                    lrate=5e-2,
                                                    burnin=200,
                                                    print_every=100,
                                                    callback=cb)

            #the way we do crossvalidation is by constructing a model with all the data.
            #we then mask the timepoints/neurons we don't want to train on
            #this information is stored in the training parameters
            train_ps = mgp.crossval.crossval.update_params(train_ps,
                                                           batch_pool=Ttrain,
                                                           prior_m=len(Ttrain))

            mod_train = mgp.crossval.train_model(model, data,
                                                 train_ps)  #train model

            # %% testing

            def mask_Ts(grad):
                ''' used to 'mask' some gradients for crossvalidation'''
                grad[:, Ttrain, ...] *= 0
                return grad

            for p in model.parameters():  #no gradients for most parameters
                p.requires_grad = False
            for p in model.lat_dist.parameters(
            ):  #only gradients for the latent distribution
                p.requires_grad = True

            #update what we're masking
            train_ps2 = mgp.crossval.crossval.update_params(
                train_ps,
                neuron_idxs=np.where(neurons_train_ind),
                mask_Ts=mask_Ts,
                prior_m=None,
                batch_pool=None,
                max_steps=1000)

            mod_train = mgp.crossval.train_model(model, data,
                                                 train_ps2)  #train

            # %% testing!

            latents = model.lat_dist.prms[0].detach()[:, Ttest,
                                                      ...]  #test latents
            query = latents.transpose(-1, -2)  #(ntrial, d, m)
            Ypred = model.svgp.sample(query, n_mc=500, noise=False)
            Ypred = Ypred.mean(0).cpu().numpy()[
                0, ~neurons_train_ind, :]  #(ntrial x N2 x T2)
            Ytarget = y_test[~neurons_train_ind, :]  #target

            from scipy.stats import pearsonr
            mean_corr = np.nanmean([
                pearsonr(Ypred[n, :], Ytarget[n, :])[0]
                for n in range(Ypred.shape[0])
            ])  #mean pearsonr across neurons
            logging.info(num_neuron_train, len_data_train)
            logging.info(f"result: {mean_corr}")

            pred = model.lat_dist.prms[0].detach().cpu().numpy()[0, ...]
            lat_err = eval_lats(pred[~Ttest, 0], z_tot[~Ttest, 0])
            print('result:', lat_err)

            lat_perfs[rep, iN, iT] = lat_err
            pred_perfs[rep, iN, iT] = mean_corr

            filename = suffix(
                f"{FLAGS.results_dir}/mgplvm_res_{FLAGS.model_type}")
            with open(f"{filename}_params.p", "wb") as f:
                pickle.dump([Ntrains, Ttrains, lat_perfs, pred_perfs], f)

            with open(f"{filename}.p", "wb") as f:
                pickle.dump([pred, Ypred, Y, z_tot, neurons_train_ind], f)


def main(_):
    os.makedirs(FLAGS.results_dir, exist_ok=True)
    for rep in range(reps):
        run(rep)


if __name__ == '__main__':
    app.run(main)
