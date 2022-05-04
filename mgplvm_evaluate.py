import numpy as np
import matplotlib.pyplot as plt
from mgplvm_utils import add_suffix, eval_lats
import functools
import pickle
from scipy.stats import pearsonr
import torch
plt.rcParams['font.size'] = 20
plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.spines.top'] = False

Ntrains =  [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
Ttrains = [75, 100, 150, 200, 300, 400, 500, 750, 1000, 2500]
Ntrains, Ttrains = Ntrains + len(Ttrains) * [30], len(Ntrains) * [1000] + Ttrains

data_type = 'synthetic'
#data_type = 'peyrache'

res = 2000
reps = 20
model_types = ['cosyne', 'orig']
results_dir = 'mgplvm_results'

if data_type == 'peyrache':
    Ntrains, Ttrains = [23], [19000]
    num_neuron_test, len_data_test = 3, 1000
    results_dir = 'peyrache_results'

lat_errs = np.zeros((2, reps, len(Ntrains)))
pred_corrs = np.zeros((2, reps, len(Ntrains)))
train_pred_losses = np.zeros((2, reps, len(Ntrains)))
marg_liks = np.zeros((2, reps, len(Ntrains)))
compute_marg_lik = True

def load_model(filename):
    # if torch.cuda.device_count() > 1.5:
    #     model = pickle.load(open(filename+"_pt.p", "rb")).to("cpu")
    #     torch.save(model.to('cpu'), filename+".pt") #save properly serialized object
    # else:
    #     model = torch.load(filename+"_pt.p")
    model = torch.load(filename+"_pt.p")
    return model

def compute_marg_lik(model, data, batch_idxs, neuron_idxs, n_mc = 500):
    '''compute log marginal likelihood for the training data'''

    all_LLs = torch.zeros(n_mc)
    torch.cuda.empty_cache()
    cuda_model = model.to("cuda")
    data = torch.Tensor(data).to("cuda")[..., batch_idxs] #only use training data

    for i in range(n_mc):
        svgp_elbo, kl = cuda_model.elbo(data, 1, kmax=5, m=len(batch_idxs), batch_idxs = batch_idxs)
        #print(svgp_elbo.shape, kl.shape) #n_mc x n_neuron, n_mc
        svgp_elbo = svgp_elbo[:, neuron_idxs]
        svgp_elbo = svgp_elbo.sum(-1)  #sum over neurons (n_mc)
        LLs = svgp_elbo - kl  # LL for each batch (n_mc)
        assert (LLs.shape == torch.Size([1]))
        all_LLs[i] = LLs[0].detach().cpu()

    LL = (torch.logsumexp(all_LLs, 0) - np.log(n_mc)) / (len(batch_idxs)*len(neuron_idxs))

    return LL.numpy()


for imod, model_type in enumerate(model_types):

    for rep in range(reps):
        for iparam in range(len(Ntrains)):

            num_neuron_train, len_data_train = Ntrains[iparam], Ttrains[iparam]
            suffix = functools.partial(add_suffix, rep, num_neuron_train, len_data_train)

            filename = suffix(f"./{results_dir}/mgplvm_res_{model_type}")

            data = pickle.load(open(f"{filename}.p", "rb"))
            keys = ['y_train', 'y_test', 'z_train', 'z_test', 'y_pred', 'z_pred', 'neuron_train_ind']
            y_train, y_test, z_train, z_test, y_pred, z_pred, neuron_train_ind = [data[k] for k in keys]

            y_test_test = y_test[~neuron_train_ind, :] #test neurons
            z_pred_test = z_pred[z_train.shape[0]:, :]

            y_pred_test = y_pred[~neuron_train_ind, :][:, y_train.shape[1]:] #test time points
            y_train_train = y_train[neuron_train_ind, :] #training time points and neurons
            if model_type == 'orig': #we sqrt transformed things
                mu = np.mean(np.sqrt(y_train), axis=1, keepdims=True)
                sig = np.std(np.sqrt(y_train), axis=1, keepdims=True)
                sig = np.maximum(sig, 1e-3) #avoid dividing by zero
                y_pred_test = (y_pred_test * sig[~neuron_train_ind, :] + mu[~neuron_train_ind, :])**2
                y_train_train = (np.sqrt(y_train_train) - mu[neuron_train_ind, :]) / sig[neuron_train_ind, :]

                Y1, Y2 = (np.sqrt(y_train) - mu) / sig, (np.sqrt(y_test) - mu) / sig
                Y = np.concatenate([Y1, Y2], axis = 1)[None, ...]
            else:
                Y = np.concatenate([y_train, y_test], axis=1)[None, ...]

            lat_err = eval_lats(z_pred_test[:, 0], z_test[:, 0], res = res)
            lat_errs[imod, rep, iparam] = lat_err

            corrs_byneuron = [pearsonr(y_pred_test[n, :], y_test_test[n, :])[0] for n in range(y_test_test.shape[0])]
            pred_corr = np.mean(corrs_byneuron)  #mean pearsonr across neurons
            pred_corrs[imod, rep, iparam] = pred_corr

            LL = None
            if compute_marg_lik:
                neuron_idxs = np.where(neuron_train_ind)[0] #training neurons
                batch_idxs = np.arange(y_train.shape[1]) #training time points
                model = load_model(filename) #load model
                LL = compute_marg_lik(model, Y, batch_idxs, neuron_idxs, n_mc = 100)
                marg_liks[imod, rep, iparam] = LL

            y_pred_train = y_pred[neuron_train_ind, :][:, :y_train.shape[1]] #training rates
            if model_type == 'cosyne':
                train_pred_loss = np.mean(y_pred_train - y_train_train * np.log(y_pred_train + 1e-9))
            else:
                if not compute_marg_lik: model = load_model(filename)
                sigmas = model.obs.likelihood.sigma.cpu().numpy()[neuron_train_ind, None]
                negliks = np.log(np.sqrt(2*np.pi)*sigmas) + 0.5 * ( (y_train_train - y_pred_train) / sigmas )**2
                train_pred_loss = np.mean(negliks)

            train_pred_losses[imod, rep, iparam] = train_pred_loss

            print(model_type, rep, num_neuron_train, len_data_train, 'lat_err:', lat_err, 'pred_corr:', pred_corr, 'pred_lik:', train_pred_loss, 'marg_lik:', LL)


nn = int(len(Ntrains)/2)
xs = [Ntrains[:nn], Ntrains[:nn], Ttrains[nn:], Ttrains[nn:]]
dats = [pred_corrs[..., :nn], lat_errs[..., :nn], pred_corrs[..., nn:], lat_errs[..., nn:]]
xlabs = ['# neurons', '# neurons', '# timepoints', '# timepoints']
ylabs = ['rate corr.', 'latent error', 'rate corr.', 'latent error']
ylims = [[-0.05, 1], [0, np.pi/2], [-0.05, 1], [0, np.pi/2]]
cols = ['k', 'b']

### store results of analysis
analysis_results = {'Ntrains': Ntrains, 'Ttrains': Ttrains, 'pred_corrs': pred_corrs, 
                    'model_types': model_types, 'lat_errs': lat_errs, 
                    'train_pred_losses': train_pred_losses, 'marg_liks': marg_liks}
pickle.dump(analysis_results, open('plots/'+results_dir+'.p', "wb"))

### plot some results 
fig = plt.figure(figsize = (7,5))
gs = fig.add_gridspec(2, 2, left=0, right=1, bottom=0, top=1., wspace = 0.3, hspace = 0.50)
col = 'k'
for i, dat in enumerate(dats):

    ax = fig.add_subplot(gs[i % 2, i // 2])

    for imod in range(2):
        m = np.mean(dat[imod, ...], axis = 0)
        s = np.std(dat[imod, ...], axis = 0) / np.sqrt(dat.shape[0])
        ax.errorbar(xs[i], m, yerr = s, color = col, ls = ['-', '--'][imod], capsize = 2)

    ax.set_xlabel(xlabs[i])
    ax.set_ylabel(ylabs[i])
    ax.set_ylim(ylims[i])

    if 'timepoints' in xlabs[i]:
        ax.set_xscale('log')

    if i == 0:
        ax.legend(['cosyne', 'neurips'], frameon = False)

plt.savefig('plots/'+results_dir+'.png', bbox_inches = 'tight')
plt.close()


print('prediction corrs:', np.mean(pred_corrs, axis = (1,2)))
print('latent errors:', np.mean(lat_errs, axis = (1,2)))
print('train predictive loss:', np.mean(train_pred_losses, axis = (1,2)))
print('train marginal LL:', np.mean(marg_liks, axis = (1,2)))

print([pearsonr(train_pred_losses[0, :, iparam], pred_corrs[0, :, iparam]) for iparam in range(len(Ntrains))])

print([pearsonr(train_pred_losses[1, :, iparam], pred_corrs[1, :, iparam]) for iparam in range(len(Ntrains))])


### compare train and test performance

for ilik, data in enumerate([train_pred_losses, marg_liks]):
    fig = plt.figure(figsize = (8,6))

    gs = fig.add_gridspec(2, 2, left=0, right=1, bottom=0, top=1., wspace = 0.5, hspace = 0.70)
    for i1 in range(2):
        for i2 in range(2):
            ax = fig.add_subplot(gs[i1, i2])

            if i2 == 0:
                ax.scatter(data[i1, :, -1], pred_corrs[i1, :, -1], color = "k")
                ax.set_ylabel("prediction accuracy")
            else:
                ax.scatter(data[i1, :, -1], lat_errs[i1, :, -1], color = "k")
                ax.set_ylabel("latent error")

            if ilik == 0:
                ax.set_xlabel("predictive loss (train)")
            else:
                ax.set_xlabel("marginal likelihood (train)")

    if ilik == 0:
        plt.savefig('plots/'+results_dir+'_LL_cor.png', bbox_inches = 'tight')
    else:
        plt.savefig('plots/'+results_dir+'_marg_LL_cor.png', bbox_inches = 'tight')
    plt.close()