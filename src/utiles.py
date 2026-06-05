import os
import numpy as np
from src.get_spikes import (fget_spk_python ,retrieve_log, tori)
from scipy.ndimage import gaussian_filter1d


def is_it_complex(directory, plot=False):

    ''' defines if the neuron in the folder us complex or simple'''

    results={}
    for folder in os.listdir(directory):
        folder_path = os.path.join(directory, folder)

        
        
        tune_logs = [
            f for f in os.listdir(folder_path)
            if f.endswith("tune.log")
        ]

        first_file = sorted(tune_logs)[0]

        
        
        flog = retrieve_log(
        path=directory/folder,
        filename=first_file,
        channels=None
    )

        info = tori(flog, cluster_index=0, plot=plot)

        relmod = info["relmod"]

        if relmod > 1.0:
            cell_type = "simple / linear-like"
        else:
            cell_type = "complex / nonlinear-like"

        results[folder] = {
            "file": first_file,
            "relmod": relmod,
            "cell_type": cell_type,
            
        }

    return results



        


def compute_sta(spike_files, stim, n_lags, frame_rate):
    T, N        = stim.shape
    lag_offsets = np.arange(1, n_lags + 1)       # [1, 2, ..., n_lags]; row 0 = most recent
    sta_accum   = np.zeros((n_lags, N))
    n_total     = 0

    for path in spike_files:
        _, spk_sec, _ = fget_spk_python(path) 
        frames = np.floor(spk_sec * frame_rate).astype(int)
        valid  = (frames >= n_lags) & (frames < T)
        frames = frames[valid]

        windows    = frames[:, np.newaxis] - lag_offsets[np.newaxis, :]   # (n_spikes, n_lags)
        sta_accum += stim[windows].sum(axis=0)                            # (n_lags, N)
        n_total   += len(frames)
        

   
    return sta_accum / n_total, n_total


def compute_r_estimate(stim, kernel):
    T, N = stim.shape
    n_lags, N_kernel = kernel.shape

    assert N == N_kernel

    r_est = np.full(T, np.nan)

    for t in range(n_lags, T):
        total = 0.0

        for lag in range(n_lags):
            stimulus_frame = stim[t - lag - 1]
            kernel_frame = kernel[lag]

            total += np.sum(stimulus_frame * kernel_frame)

        r_est[t] = total

    return r_est





def gaussian_rate_convolution(spike_times, t_start=None, t_end=None, dt=0.001, sigma=0.2):
    """
    Gaussian-smoothed firing rate using binning + convolution.

    spike_times : spike times in seconds
    dt          : bin width in seconds
    sigma       : Gaussian std in seconds

    returns:
        t        : time axis
        rate     : firing rate in spikes/s
        counts   : binned spike counts
    """
    spike_times = np.asarray(spike_times)

    if t_start is None:
        t_start = spike_times[0]
    if t_end is None:
        t_end = spike_times[-1]

    bins = np.arange(t_start, t_end + dt, dt)
    counts, edges = np.histogram(spike_times, bins=bins)

    # Convert sigma from seconds to bins
    sigma_bins = sigma / dt

    # Smooth spike counts
    smoothed_counts = gaussian_filter1d(
        counts.astype(float),
        sigma=sigma_bins,
        mode="constant"
    )

    # Convert counts/bin to spikes/second
    rate = smoothed_counts / dt

    # Bin centers
    t = edges[:-1] + dt / 2

    return t, rate, counts