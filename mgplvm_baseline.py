#%%

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
import time
from mgplvm_utils import add_suffix, eval_lats, plot_data, gen_cb
from scipy.stats import pearsonr

FLAGS = flags.FLAGS

flags.DEFINE_integer("n_z", 16, "number of inducing points")
flags.DEFINE_integer("random_seed", 42, "random seed")
flags.DEFINE_string("model_type", "cosyne",
                    "`orig` (Gaussian + uniform) or `cosyne` (Poisson + AR)")
flags.DEFINE_string("results_dir", "./mgplvm_results/",
                    "results directory")
flags.DEFINE_string("device", "cuda", "device: cuda or cpu")
flags.DEFINE_integer("max_iter", 2000, "training iterations")
flags.DEFINE_integer("n_mc", 20, "MC samples")

#%%

d = 1  #model dimensionality
n_samples = 1  #number of trials
reps = 20 #number of repetitions

Ntrains =  [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
Ttrains = [75, 100, 150, 200, 300, 400, 500, 750, 1000, 2500]
Ntrains, Ttrains = Ntrains + len(Ttrains) * [30], len(Ntrains) * [1000] + Ttrains
num_neuron_test = 30
len_data_test = 1000

#Ntrains, Ttrains = Ntrains[::2], Ttrains[::2]

lat_perfs = np.zeros((reps, len(Ntrains)))
pred_perfs = np.zeros((reps, len(Ntrains)))

#%%

def run(rep):
    index = rep
    for iparam in range(len(Ntrains)):
        tic = time.time()

        num_neuron_train, len_data_train = Ntrains[iparam], Ttrains[iparam]
        suffix = functools.partial(add_suffix, rep, num_neuron_train,
                                    len_data_train)
        num_neuron_tot = num_neuron_train + num_neuron_test
        len_data_tot = len_data_train + len_data_test

        np.random.seed(FLAGS.random_seed+index)
        torch.manual_seed(FLAGS.random_seed+index)
        y_train, z_train, y_test, z_test, rf, neurons_train_ind = get_data(
            num_neuron_train, num_neuron_test, len_data_train,
            len_data_test, index, FLAGS.random_seed)

        logging.info(f"y_train: {y_train.shape}")  #Ntot x Ttrain
        logging.info(f"z_train: {z_train.shape}")  #d x Ttrain
        logging.info(f"y_test: {y_test.shape}")  #Ntot x Ttest
        logging.info(f"z_test:' {z_test.shape}")  #d x Ttest
        logging.info(f"N_train: {num_neuron_train}")

        ## plot data
        fname = suffix(f"{FLAGS.results_dir}/raw_data") + ".png"
        plot_data(y_train, len_data_train, num_neuron_tot, fname)

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

        #print(Ttrain)
        #print(Ttest)
        print(Y.shape)

        manif = mgp.manifolds.Torus(m, d)  # latent distribution manifold

        if FLAGS.model_type == 'orig':  #Gaussian noise and uniform prior
            lik = mgp.likelihoods.Gaussian(n, Y=Y1,
                                            d=d, sigma = 0.5*torch.ones(n,))  # Gaussian likelihood
            lprior = mgp.lpriors.Uniform(
                manif)  # Prior on the manifold distribution
            sig_kernel = np.sqrt(1 - 0.5**2)
            ell, lat_sig = 1.0, np.pi/2
        else:  #Poisson noise and AR prior
            lik = mgp.likelihoods.Poisson(n)  #Poisson likelihood
            lprior = mgp.lpriors.Brownian(manif,
                                            fixed_brownian_c=True,
                                            fixed_brownian_eta=False,
                                            brownian_eta=torch.ones(d) *
                                            np.pi**2)
            sig_kernel = 0.05
            ell, lat_sig = 0.5, np.pi/4

        lat_dist = mgp.rdist.ReLie(
            manif,
            m,
            n_samples,
            Y=Y,
            initialization='random',
            sigma = lat_sig,
            diagonal=True)  # construct ReLie distribution

        kernel = mgp.kernels.QuadExp(
            n,
            manif.distance,
            Y=Y1,
            ell=np.ones(n) * ell,
            scale = sig_kernel * np.ones(n)
        )  # Use an exponential quadratic (RBF) kernel

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

        cb = gen_cb(z_tot, suffix(f"{FLAGS.results_dir}/latents") + '_' + FLAGS.model_type + ".png")
        train_ps = mgp.crossval.training_params(max_steps=int(round(FLAGS.max_iter)),
                                                n_mc=FLAGS.n_mc,
                                                lrate=5e-2,
                                                burnin=200,
                                                print_every=100,
                                                callback=cb)

        #the way we do crossvalidation is by constructing a model with all the data.
        #we then mask the timepoints/neurons we don't want to train on
        #this information is stored in the training parameters
        train_ps = mgp.crossval.crossval.update_params(train_ps,
                                                        batch_pool=Ttrain,
                                                        prior_m=len(Ttrain),
                                                        neuron_idxs=np.where(neurons_train_ind)[0])
        #print(train_ps)

        mod_train = mgp.crossval.train_model(model, data,
                                                train_ps)  #train model

        
        # %% now get tuning curves for the test neurons

        for p in model.lprior.parameters():
            p.requires_grad = False

        train_ps1 = mgp.crossval.crossval.update_params(train_ps,
                                                        max_steps=int(round(FLAGS.max_iter/2)),
                                                        neuron_idxs=np.where(~neurons_train_ind)[0],
                                                        mask_Ts = (lambda x: x*0)
                                                        )

        #print(train_ps1)
        mod_train = mgp.crossval.train_model(model, data,
                                                train_ps1)  #train model

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
            neuron_idxs=np.where(neurons_train_ind)[0],
            mask_Ts=mask_Ts,
            prior_m=None,
            batch_pool=None,
            max_steps=int(round(FLAGS.max_iter/2)))
        #print(train_ps2)
        #for p in model.lat_dist.parameters():
        #    print(p)

        mod_train = mgp.crossval.train_model(model, data,
                                                train_ps2)  #train

        # %% testing!

        latents = model.lat_dist.prms[0].detach()[:, ...]  #test latents
        query = latents.transpose(-1, -2)  #(ntrial, d, m)
        Ypred = model.svgp.sample(query, n_mc=500, noise=False)
        Ypred = Ypred.mean(0).cpu().numpy()[0, ...]  
        Ypred_sub = Ypred[~neurons_train_ind, :][:, Ttest]#(ntrial x N2 x T2)
        

        if FLAGS.model_type == 'orig':
            print('normalizing for eval')
            Ytarget = ((np.sqrt(y_test) - mu) / sig) #normalize
            Ytarget = Ytarget[~neurons_train_ind, :]
        else:
            Ytarget = y_test[~neurons_train_ind, :]  #target

        mean_corr = np.nanmean([
            pearsonr(Ypred_sub[n, :], Ytarget[n, :])[0]
            for n in range(Ypred_sub.shape[0])
        ])  #mean pearsonr across neurons
        print(num_neuron_train, len_data_train)
        #logging.info(f"result: {mean_corr}")
        print('result:', mean_corr)

        z_pred = model.lat_dist.prms[0].detach().cpu().numpy()[0, ...]
        lat_err = eval_lats(z_pred[Ttest, 0], z_tot[Ttest, 0])
        print('result:', lat_err)
        print('elapsed time:', time.time()-tic)

        lat_perfs[rep, iparam] = lat_err
        pred_perfs[rep, iparam] = mean_corr

        filename = suffix(
            f"{FLAGS.results_dir}/mgplvm_res_{FLAGS.model_type}")
        with open(f"{filename}_params.p", "wb") as f:
            pickle.dump([Ntrains, Ttrains, lat_perfs, pred_perfs], f)

        data = {'y_train': y_train, 'y_test': y_test, 'neuron_train_ind': neurons_train_ind,
                'z_train': z_train, 'z_test': z_test, 'y_pred': Ypred, 'z_pred': z_pred}
        with open(f"{filename}.p", "wb") as f:
            pickle.dump(data, f)
        with open(f"{filename}_pt.p", "wb") as f:
            pickle.dump(model, f)


def main(_):
    os.makedirs(FLAGS.results_dir, exist_ok=True)
    for rep in range(reps):
        run(rep)


if __name__ == '__main__':
    app.run(main)
