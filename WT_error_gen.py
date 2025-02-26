import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
from copy import deepcopy

def WT_sce_gen(WT_no, samples_no, seed = 0):
    ### step 1, find WT standard curve, i.e., the prototype curve
    WT_forecast = np.load(os.path.join(os.getcwd(), 'data', 'GEFC2012_wind_forecast.npy'))
    WT_forecast = WT_forecast.reshape([1, -1, 1])
    WT_forecast = np.tile(WT_forecast, (1, 1, WT_no))

    ## get the WT error
    WT_error_samples = np.load(os.path.join(os.getcwd(), 'data', 'GEFC2012_temporal_correlated_error_samples.npy'))
    WT_error_samples = WT_error_samples[:samples_no * WT_no].reshape(samples_no, WT_no, -1)
    # permute the last two dimensions
    WT_error_samples = np.transpose(WT_error_samples, (0, 2, 1))

    WT_samples_full = WT_forecast + WT_error_samples
    ### wind scenarios must be non-negative
    WT_samples_full[WT_samples_full < 0] = 0
    # re_calculate the error
    WT_error_samples = WT_samples_full - WT_forecast

    # normalize to make the maximum generation to be 1
    norm_factor = np.max(WT_samples_full)
    WT_samples_full = WT_samples_full / norm_factor
    WT_error_samples = WT_error_samples / norm_factor
    WT_forecast = WT_forecast / norm_factor

    # check if WT_samples == WT_forecast + WT_error
    if not np.allclose(WT_samples_full, WT_forecast + WT_error_samples):
        raise ValueError('the WT samples are not equal to the sum of forecast and error')

    if np.min(WT_samples_full) < 0:
        raise ValueError('the RES samples have negative generation!!!')

    return WT_forecast[0], WT_error_samples, WT_samples_full

if __name__ == '__main__':
    WT_no = 10
    samples_no = 2000
    WT_forecast, WT_error_samples, WT_samples_full = WT_sce_gen(WT_no, samples_no)
    # plot the histogram of the error and the WT generation scenarios
    fig, axs = plt.subplots(2, 1, figsize=(6, 6))
    axs[0].hist(WT_error_samples.flatten(), bins=150, density=True, alpha=0.75, label='error samples')
    # plot the Gaussian fit
    from scipy.stats import norm
    mu, std = np.mean(WT_error_samples.flatten()), np.std(WT_error_samples.flatten())
    x = np.linspace(np.min(WT_error_samples), np.max(WT_error_samples), 100)
    p = norm.pdf(x, mu, std)
    axs[0].plot(x, p, 'r', linewidth=1, label='Gaussian fit')

    axs[0].set_title('WT error samples histgram')
    axs[0].set_xlabel('normalised error')
    axs[0].set_ylabel('frequency')
    axs[0].legend()

    axs[1].plot(WT_samples_full[..., 0].T)
    axs[1].set_title('WT generation samples')
    axs[1].set_xlabel('hour')
    axs[1].set_ylabel('normalised generation')
    axs[1].set_xlim([0, WT_forecast.shape[0] - 1])
    plt.tight_layout()
    plt.show()