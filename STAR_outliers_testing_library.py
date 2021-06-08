import numpy as np
import os
import pdb
from copy import deepcopy as COPY
from matplotlib import pyplot as plt
from scipy.stats import norm
from scipy.stats import pearsonr
from scipy.stats import gaussian_kde as smooth
from scipy.stats import expon
from scipy.stats import norm

def bin_data(x, n_bins):

    """
    Purpose
    -------
    to convert numeric data into evenly spaced bins and bin counts
    
    Parameters
    ----------
    x: any numpy array with numeric data  
    n_bins: number of bins

    Returns
    -------
    real_domain: numpy array of bin midpoints
    real_range: numpy array of bin counts

    """

    delta = 0.000001*np.mean(np.abs(x))
    bounds = np.linspace(np.min(x), np.max(x) + delta, n_bins).reshape(-1, 1)
    lbs, ubs = bounds[:-1], bounds[1:]
    bins = np.logical_and(x >= lbs, x < ubs)
    real_domain = (ubs.reshape(-1) + lbs.reshape(-1))/2
    real_range = np.sum(bins, axis = 1)
    return(real_domain, real_range)

def detect_monotonic_data(x, n_bins):

    """
    Purpose
    -------
    to quantify the degree of monotonicity in data
    
    Parameters
    ----------
    x: any numpy array with numeric data  
    n_bins: number of bins

    Returns
    -------
    monotonicity: a heuristic measure of a dataset's monotonicity
                  Perfectly monotonic data will have a score of 1,
                  while a score of 0.5 implies no monotonic trend. 

    """

    real_domain, real_range = bin_data(x, n_bins)
    real_range = real_range/np.sum(real_range)
    indices = np.arange(len(real_range))
    index_pairs = np.random.choice(indices, size = (100000, 2), p = real_range)
    sorted_index_pairs = np.sort(index_pairs, axis = 1)
    range_pairs = real_range[sorted_index_pairs]
    directions = (range_pairs[:, 0] - range_pairs[:, 1]) < 0
    monotonicity = np.max([np.mean(directions), 1 - np.mean(directions)])
    return(monotonicity)   

def detect_exponential_data(x_unadjusted, n_bins):

    """
    Purpose
    -------
    to determine if data is similar to an exponential distribution
    
    Parameters
    ----------
    x_unadjusted: numeric input numpy array
    n_bins: number of bins

    Returns
    -------
    exp_status: if true, indicates that data either fits an exponential
                function or some function steeper than an exponential 

    """

    x = x_unadjusted - np.min(x_unadjusted)
    real_domain, real_range = bin_data(x, n_bins)
    nonzeros = real_range > 0
    B1x = (real_domain[nonzeros] - np.mean(real_domain[nonzeros]))
    B1y = (np.log(real_range[nonzeros]) - np.mean(np.log(real_range[nonzeros])))
    B1 = np.sum(B1x*B1y)/np.sum(B1x*B1x)
    B0 = np.mean(np.log(real_range[nonzeros])) - np.mean(real_domain[nonzeros])
    est_log_range = B0 + B1*real_domain[nonzeros]
    real_log_range = np.log(real_range[nonzeros])

    exp_status1 =  pearsonr(est_log_range, real_log_range)[0] > 0.95 
    monotonicity = detect_monotonic_data(x, np.max([10, int(len(x)/150)]))
    exp_status2 = monotonicity > 0.9
    exp_status = exp_status1 or exp_status2

    return(exp_status)

