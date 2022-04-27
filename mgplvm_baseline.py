#%% load some libraries

import numpy as np
import mgplvm as mgp
from data import  get_data
import matplotlib.pyplot as plt
import torch
import pickle

#%% evaluate model

# set some parameters
global_seed = np.random.choice(np.arange(100000000))
d = 1 #model dimensionality
n_samples = 1 #number of trials
n_z = 20 #number of inducing points
type_ = 'cosyne' #orig (Gaussian+uniform) or cosyne (Poisson+AR)

def eval_lats(pred, target):
    errs = np.zeros((2, 1000))
    for i in range(2):
        for i_s, s in enumerate(np.linspace(0, 2*np.pi, 1000)):
            newpred = 2*(0.5-i) * pred + s
            errs[i, i_s] = np.mean(np.arccos(np.cos(newpred-target)))
    return np.amin(errs)

reps = 2
Ntrains = [20, 50, 100, 200]
Ttrains = [50, 100, 200, 500]

lat_perfs = np.zeros((reps, len(Ntrains), len(Ttrains)))
pred_perfs = np.zeros((reps, len(Ntrains), len(Ttrains)))

for rep in range(reps):
    index = rep
    for iN, num_neuron_train in enumerate(Ntrains):
        for iT, len_data_train in enumerate(Ttrains):

            #%% set a bunch of parameters and generate data

            num_neuron_test = 80
            len_data_test = 500
            num_neuron_tot = num_neuron_train + num_neuron_test
            len_data_tot = len_data_train + len_data_test

            device = torch.device("cuda")

            y_train, z_train, y_test, z_test, rf, neurons_train_ind = get_data(
                                                                                num_neuron_train,
                                                                                num_neuron_test,
                                                                                len_data_train,
                                                                                len_data_test, 
                                                                                index, 
                                                                                global_seed
                                                                            )

            print('y_train:', y_train.shape) #Ntot x Ttrain
            print('z_train:', z_train.shape) #d x Ttrain
            print('y_test:', y_test.shape) #Ntot x Ttest
            print('z_test:', z_test.shape) #d x Ttest
            print('N_train:', num_neuron_train)

            ## plot data
            plt.figure()
            plt.imshow(y_train, cmap="Greys", aspect="auto")
            plt.xlabel("time")
            plt.ylabel("neuron")
            plt.xlim(0, len_data_train)
            plt.ylim(0, num_neuron_tot)
            plt.xticks([])
            plt.yticks([])
            plt.show()


            #%% construct mgplvm model
            torch.cuda.empty_cache()

            if type_ == 'orig': #Gaussian noise and uniform prior
                Y1 = np.sqrt(y_train) #sqrt transform
                mu, sig = np.mean(Y1, axis = 1, keepdims = True), np.std(Y1, axis = 1, keepdims = True)
                sig = np.maximum(sig, 1e-3)
                Y1 = ( ( Y1 - mu ) / sig )
                Y2 = ( ( np.sqrt(y_test) - mu) / sig )#[neurons_train_ind, :]
            else: #Poisson noise and AR prior
                Y1 = y_train
                Y2 = y_test#[neurons_train_ind, :]

            #wrangle the data into the right form and put on gpu
            Ttrain, Ttest = np.arange(len_data_train), np.arange(len_data_train, len_data_train+len_data_test)
            Y = np.concatenate([Y1, Y2], axis = 1)
            Y = Y[None, ...]
            Y1 = Y1[None, ...]
            z_tot = np.concatenate([z_train, z_test], axis = 0)
            data = torch.tensor(Y).to(device)
            n_samples, n, m = Y.shape

            manif = mgp.manifolds.Torus(m, d)  # latent distribution manifold
            lat_dist = mgp.rdist.ReLie(manif, m, n_samples, Y = Y, initialization = 'random', sigma = np.pi/2, diagonal = True)  # construct ReLie distribution
            kernel = mgp.kernels.QuadExp(
                n, manif.distance, Y = Y1, ell = np.ones(n)*1,
            )  # Use an exponential quadratic (RBF) kernel

            if type_ == 'orig': #Gaussian noise and uniform prior
                lik = mgp.likelihoods.Gaussian(n, Y = Y1, d = d)  # Gaussian likelihood
                lprior = mgp.lpriors.Uniform(manif)  # Prior on the manifold distribution
            else: #Poisson noise and AR prior
                lik = mgp.likelihoods.Poisson(n) #Poisson likelihood
                lprior = mgp.lpriors.Brownian(manif, fixed_brownian_c = True, fixed_brownian_eta = False, brownian_eta = torch.ones(d)*np.pi**2)

            z = manif.inducing_points(n, n_z)  # build inducing points
            model = mgp.models.SvgpLvm(
                n, m, n_samples, z, kernel, lik, lat_dist, lprior, whiten=True
            ).to(device) #build full model


            # %% training

            def cb(mod, i, loss):
                """here we construct an (optional) function that helps us keep track of the training"""
                if i in [0, 50, 100, 150, 300, 500, 1000, 1500, 1999]: #iterations to plot
                    X = mod.lat_dist.prms[0].detach().cpu().numpy()[0, ...]
                    X = X % (2*np.pi)
                    plt.figure(figsize = (4,4))
                    plt.xlim(0, 2*np.pi); plt.ylim(0, 2*np.pi)
                    plt.xlabel("true latents"); plt.ylabel("model latents")
                    plt.scatter(z_tot[:, 0], X[:, 0], color = 'k')
                    plt.title('iter = '+str(i))
                    plt.show()

            train_ps = mgp.crossval.training_params(max_steps = 2000, n_mc = 50, lrate = 5e-2, burnin = 200, print_every = 100, callback = cb)

            #the way we do crossvalidation is by constructing a model with all the data.
            #we then mask the timepoints/neurons we don't want to train on
            #this information is stored in the training parameters
            train_ps = mgp.crossval.crossval.update_params(train_ps, batch_pool=Ttrain, prior_m=len(Ttrain))

            mod_train = mgp.crossval.train_model(model, data, train_ps) #train model

            # %% testing

            def mask_Ts(grad):
                ''' used to 'mask' some gradients for crossvalidation'''
                grad[:, Ttrain, ...] *= 0
                return grad

            for p in model.parameters():  #no gradients for most parameters
                p.requires_grad = False
            for p in model.lat_dist.parameters():  #only gradients for the latent distribution
                p.requires_grad = True

            #update what we're masking
            train_ps2 = mgp.crossval.crossval.update_params(train_ps, neuron_idxs=np.where(neurons_train_ind), mask_Ts=mask_Ts, prior_m=None, batch_pool = None, max_steps = 1000)

            mod_train = mgp.crossval.train_model(model, data, train_ps2) #train

            # %% testing!

            latents = model.lat_dist.prms[0].detach()[:, Ttest, ...] #test latents
            query = latents.transpose(-1, -2)  #(ntrial, d, m)
            Ypred = model.svgp.sample(query, n_mc=500, noise=False)
            Ypred = Ypred.mean(0).cpu().numpy()[0, ~neurons_train_ind, :]  #(ntrial x N2 x T2)
            Ytarget = y_test[~neurons_train_ind, :] #target

            from scipy.stats import pearsonr
            mean_corr = np.mean([pearsonr(Ypred[n, :], Ytarget[n, :])[0] for n in range(Ypred.shape[0])]) #mean pearsonr across neurons
            print('\n', num_neuron_train, len_data_train)
            print('result:', mean_corr)

            pred = model.lat_dist.prms[0].detach().cpu().numpy()[0, ...]
            lat_err = eval_lats(pred[~Ttest, 0], z_tot[~Ttest, 0])
            print('result:', lat_err)

            lat_perfs[rep, iN, iT] = lat_err
            pred_perfs[rep, iN, iT] = mean_corr

            pickle.dump([Ntrains, Ttrains, lat_perfs, pred_perfs], open('mgplvm_res_'+type_+'.p', 'wb'))

# %%
