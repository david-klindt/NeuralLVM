import numpy as np
import matplotlib.pyplot as plt
from mgplvm_utils import add_suffix, eval_lats
import functools
import pickle
from scipy.stats import pearsonr
plt.rcParams['font.size'] = 20
plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.spines.top'] = False

Ntrains =  [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
Ttrains = [75, 100, 150, 200, 300, 400, 500, 750, 1000, 2500]
Ntrains, Ttrains = Ntrains + len(Ttrains) * [30], len(Ntrains) * [1000] + Ttrains

results_dir = 'mgplvm_results'

res = 2000
reps = 4
model_types = ['cosyne', 'orig']

lat_errs = np.zeros((2, reps, len(Ntrains)))
pred_corrs = np.zeros((2, reps, len(Ntrains)))

for imod, model_type in enumerate(model_types):

    for rep in range(reps):
        for iparam in range(len(Ntrains)):

            num_neuron_train, len_data_train = Ntrains[iparam], Ttrains[iparam]
            suffix = functools.partial(add_suffix, rep, num_neuron_train, len_data_train)

            filename = suffix(f"./{results_dir}/mgplvm_res_{model_type}")

            data = pickle.load(open(f"{filename}.p", "rb"))
            keys = ['y_train', 'y_test', 'z_train', 'z_test', 'y_pred', 'z_pred', 'neuron_train_ind']
            y_train, y_test, z_train, z_test, y_pred, z_pred, neuron_train_ind = [data[k] for k in keys]
            
            y_test = y_test[~neuron_train_ind, :] #test neurons
            z_pred_test = z_pred[z_train.shape[0]:, :]

            y_pred_test = y_pred[~neuron_train_ind, :][:, y_train.shape[1]:] #test time points
            if model_type == 'orig': #we sqrt transformed things
                mu = np.mean(np.sqrt(y_train), axis=1, keepdims=True)[~neuron_train_ind, :]
                sig = np.std(np.sqrt(y_train), axis=1, keepdims=True)[~neuron_train_ind, :]
                sig = np.maximum(sig, 1e-3) #avoid dividing by zero
                y_pred_test = (y_pred_test * sig + mu)**2

            lat_err = eval_lats(z_pred_test[:, 0], z_test[:, 0], res = res)
            lat_errs[imod, rep, iparam] = lat_err

            corrs_byneuron = [pearsonr(y_pred_test[n, :], y_test[n, :])[0]for n in range(y_test.shape[0])]
            pred_corr = np.mean(corrs_byneuron)  #mean pearsonr across neurons
            pred_corrs[imod, rep, iparam] = pred_corr

            print(model_type, rep, num_neuron_train, len_data_train, lat_err, pred_corr)


nn = int(len(Ntrains)/2)
xs = [Ntrains[:nn], Ntrains[:nn], Ttrains[nn:], Ttrains[nn:]]
dats = [pred_corrs[..., :nn], lat_errs[..., :nn], pred_corrs[..., nn:], lat_errs[..., nn:]]
xlabs = ['# neurons', '# neurons', '# timepoints', '# timepoints']
ylabs = ['rate corr.', 'latent error', 'rate corr.', 'latent error']
ylims = [[-0.05, 1], [0, np.pi/2], [-0.05, 1], [0, np.pi/2]]
cols = ['k', 'b']

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

    if i == 0:
        ax.legend(['cosyne', 'neurips'], frameon = False)

plt.savefig('plots/'+results_dir+'.png', bbox_inches = 'tight')
plt.close()


print(np.mean(pred_corrs, axis = (1,2)))
print(np.mean(lat_errs, axis = (1,2)))