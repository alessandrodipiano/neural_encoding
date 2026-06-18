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


def compute_r_estimate(stim, kernel, r0=0.0):
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

        r_est[t] = r0 + total

    return r_est







def bin_spikes_to_frames(spike_times_sec, T, frame_rate):
     
    edges = np.arange(T + 1) / frame_rate
    counts, _ = np.histogram(spike_times_sec, bins=edges)
    return counts


def smooth_frame_counts(counts, frame_rate, sigma_sec=0.05):
    sigma_frames = sigma_sec * frame_rate
    return gaussian_filter1d(
        counts.astype(float),
        sigma=sigma_frames,
        mode="constant"
    ) * frame_rate



