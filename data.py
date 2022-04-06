import numpy as np
from scipy.spatial import distance


class data_generation:
    def __init__(
            self,
            num_neuron=50,
            len_data=1000,
            num_ensemble=1,
            dim=1,
            periodicity=True,
            kernel_type='exp',
            kernel_sdev=5.0,
            kernel_scale=50.0,
            bump_placement='random',
            peak_firing=0.5,
            back_firing=0.005,
            tuning_width=1.2,
            snr_scale=1.0,
            poisson_noise=False,
            seed=1337
    ):
        self.len_data = len_data
        self.kernel_type = kernel_type
        self.kernel_sdev = kernel_sdev
        self.kernel_scale = kernel_scale
        self.dim = dim
        self.periodicity = periodicity
        
        self.num_neuron = num_neuron
        self.num_ensemble = num_ensemble
        self.bump_placement = bump_placement 
        self.peak_firing = peak_firing
        self.back_firing = back_firing
        self.tuning_width = tuning_width
        self.snr_scale = snr_scale
        self.poisson_noise = poisson_noise
        
        self.seed = seed
        
        self.ensemble_weights = np.zeros((self.num_neuron * self.num_ensemble, self.num_ensemble))
        for i in range(self.num_ensemble):
            self.ensemble_weights[i * self.num_neuron:(i + 1) * self.num_neuron, i] = 1
        
    def generate_z(self):
        np.random.seed(self.seed)

        if self.kernel_type == 'exp':
            dist = distance.cdist(np.linspace(1, self.len_data,self.len_data).reshape((self.len_data,1)),
                                  np.linspace(1, self.len_data,self.len_data).reshape((self.len_data,1)), 
                                  'euclidean')
            K_t = self.kernel_sdev * np.exp(-dist / self.kernel_scale)
            
        elif self.kernel_type != 'exp':
            # might want different kernels
            print("Not fixed yet")
        z = []
        for i in range(self.dim * self.num_ensemble):
            z.append(np.random.multivariate_normal(np.zeros(self.len_data), K_t))
        z = np.asarray(z).T
        if self.periodicity:
            z = z % (2 * np.pi)
            
        return z
       
    def generate_receptive_fields(self, z):
        np.random.seed(self.seed)
        
        if self.periodicity:
            if self.bump_placement == 'random':
                rf_location = 2 * np.pi * np.random.rand(self.num_neuron * self.num_ensemble, self.dim)
            elif self.bump_placement == 'uniform':
                rf_location = np.tile(np.array([[(i + 0.5) / self.num_neuron * (2 * np.pi)
                                                 for i in range(self.num_neuron)] 
                                                for j in range(self.dim)]).T, (self.num_ensemble, 1))
        else:
            min_z = np.min(z, axis=0)
            max_z = np.max(z, axis=0)
            
            if self.bump_placement == 'random':
                rf_location = np.random.rand(self.num_neuron * self.num_ensemble, self.dim)
                rf_location = min_z + rf_location * (max_z - min_z)
            elif self.bump_placement == 'uniform':
                rf_location = np.tile(np.array([min_z + [(i + 0.5) / self.num_neuron * (max_z - min_z) 
                                                         for i in range(self.num_neuron)] 
                                                for j in range(self.dim)]).T, (self.num_ensemble, 1))
        
        return rf_location
    
    def generate_spikes(self, z, rf_location):
        np.random.seed(self.seed)
        
        selector = np.stack([np.eye(self.dim) for i in range(self.num_ensemble)], 0)
        selector = self.ensemble_weights[..., None, None] * selector[None]
        selector = np.concatenate(np.split(selector, self.num_ensemble, axis=1), axis=3).reshape(
            self.num_neuron * self.num_ensemble, self.dim, self.num_ensemble * self.dim)
        selected = np.matmul(selector, z.T)
        
        dist = (rf_location[..., None] - selected)
        if self.periodicity:
            dist = np.abs(dist)
            dist[dist > np.pi] = 2 * np.pi - dist[dist > np.pi]
        dist = dist**2
        dist = np.sum(dist, axis=1)
            
        response = np.log(self.back_firing) + (np.log(self.peak_firing / self.back_firing)) * np.exp(-dist / (2 * self.tuning_width))
        response = np.exp(response) * self.snr_scale
        if self.poisson_noise:
            response = np.random.poisson(response)
        response = response / self.snr_scale
        
        return response

def get_data(
    num_neuron_train,
    num_neuron_test,
    len_data_train,
    len_data_test, 
    index, 
    global_seed
):
    num_neuron = num_neuron_train + num_neuron_test
    data = data_generation(
        len_data=len_data_train + len_data_test,
        dim=1, 
        num_neuron=num_neuron,
        poisson_noise=True,
        bump_placement='random', 
        seed=global_seed + index
    )
    
    print("Generating latents\n")
    z = data.generate_z()
    z_train = z[:len_data_train, :]
    z_test = z[len_data_train:, :]
    
    print("Generating receptive fields\n")
    rf = data.generate_receptive_fields(z)
    
    print("Generating spikes")
    y_train = data.generate_spikes(z_train, rf)
    data.poisson_noise = False
    y_test = data.generate_spikes(z_test, rf)

    # select training and test neurons
    np.random.seed(global_seed + index)
    neurons_train_ind = np.zeros(num_neuron, dtype=bool)
    ind = np.random.choice(num_neuron, num_neuron_train, replace=False)
    neurons_train_ind[ind] = True
    
    return y_train, z_train, y_test, z_test, rf, neurons_train_ind
