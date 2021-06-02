import numpy as np
import os
import pdb
from copy import deepcopy as COPY
from statsmodels.stats.stattools import medcouple 
from matplotlib import pyplot as plt
from scipy.stats import norm
from scipy.stats import pearsonr
from scipy.stats import gaussian_kde as smooth
from scipy.stats import expon
from scipy.stats import norm
from scipy.stats.mstats import gmean

#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
# SECTION 1: code that removes some spikes and severe outliers from data
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------

def remove_spikes(x, x_spiked, name, prefix, count, spikes, decreases):
    x_unique, x_counts = np.unique(x, return_counts = True)
    if np.max(x_counts)/np.sum(x_counts) < 0.5 or count == 3:
        return(x, spikes, decreases)
    decreases.append(1 - np.max(x_counts)/np.sum(x_counts))
    new_spike = x_unique[np.argmax(x_counts)]
    spikes.append(new_spike)
    x = x[x != new_spike]
    count += 1
    return(remove_spikes(x, x_spiked, name, prefix, 
                         count, spikes, decreases))

def remove_severe_outliers(x, name):
    p_low, p_high = np.percentile(x, [2.5, 97.5])
    range0 = p_high - p_low
    lb = p_low - 5*range0
    ub = p_high + 5*range0
    severe_outliers = x[np.logical_or(x < lb, x > ub)]
    message = "The following severe outliers were removed for feature "
    message += name + ": " + str(np.unique(severe_outliers))
    if len(severe_outliers > 0):
        print(message)
    return(x[np.isin(x, severe_outliers) == False], severe_outliers)

def clean_data(x_spiked, name, prefix, count, 
               spikes, decreases, despike = True):
    x = COPY(x_spiked)[np.isnan(x_spiked)==False]
    if despike:
        x, spikes, decreases = remove_spikes(x, x_spiked, name, 
                                             prefix, 0, [], [])
        if len(spikes) > 3:
            return(backup_test(x_spiked, name, prefix, gmean(decreases)))
    x, severe_outliers = remove_severe_outliers(x, name) 
    return(x, severe_outliers)

#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
# SECTION 2: code that computes quantiles from the continuous
#            approximation of a problematically discrete distribution
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------

def get_fitted_quantiles(percentiles, fitted_cdf, range0, pstart, pend):
    Q_vec = np.zeros(5)
    if range0[np.where(fitted_cdf >= 0.25)[0][0]] <= pstart: 
        Q_vec[0] = range0[np.where(fitted_cdf >= percentiles[0]/100)[0][0]]
    else:
        Q_vec[0] = pstart
    Q_vec[1] = range0[np.where(fitted_cdf >= 0.25)[0][0]]  
    Q_vec[2] = range0[np.where(fitted_cdf >= 0.5)[0][0]] 
    Q_vec[3] = range0[np.where(fitted_cdf >= 0.75)[0][0]]  
    if range0[np.where(fitted_cdf >= 0.75)[0][0]] >= pend: 
        Q_vec[4] = range0[np.where(fitted_cdf >= percentiles[4]/100)[0][0]]
    else:
        Q_vec[4] = pend   
    return(Q_vec)

def approximate_quantiles(x, percentiles, bw_coef = 0.3):

    """
    Purpose
    -------
    to smoothly approximate discrete distributions for
    the purpose of computing quantiles when necessary.
    
    Parameters
    ----------
    x: numeric input numpy array
    percentiles: list of percentiles at which quantiles are computed
                 from the x distribution's smooth approximation
    bw_coef: smoothing parameter value that usually works well

    Returns
    -------
    Q_vec: list of quantiles that were computed from the 
           x distribution's smooth approximation. Extreme
           quantiles are taken from x when possible. 

    """

    pstart = np.percentile(x, percentiles[0])
    p1, p20, p35, p65, p80, p99 = np.percentile(x, [0.5, 20, 35, 65, 80, 99.5])
    spacer = (p99 - p1)/2.75
    pend = np.percentile(x, percentiles[-1])
    range1 = np.linspace(p1 - spacer, p35, 200)
    range2 = np.linspace(p35, p65, 200)
    range3 = np.linspace(p65, p99 + spacer, 200)
    range0 = np.concatenate([range1[:-1], range2[:-1], range3])
    x0 = np.where(range0 >= p20)[0][0]
    x1 = np.where(range0 <= p80)[0][-1]
    x_bounded = x[np.logical_and(x >= p1, x <= p99)]
    smooth_x = smooth(x_bounded, bw_method = bw_coef)(range0)
    mid_x = (smooth_x[:-1] + smooth_x[1:])/2

    # If there are too few unique values to uniquely define p20 from nearby 
    # percentiles, then this method works if the distribution isn't too wierd.
    integrand1 = (mid_x*(range0[1:] - range0[:-1]))
    cdf1 = np.cumsum(integrand1)
    lb_is_good = np.min(cdf1) < 0.05 and np.min(cdf1) > -0.05
    ub_is_good = np.max(cdf1) < 1.05 and np.max(cdf1) > 0.95
    if lb_is_good and ub_is_good:
        Q_vec = get_fitted_quantiles(percentiles, cdf1, range0, pstart, pend)
        return(Q_vec)

    # if there are many unique values but the distribution is wierd
    # then this method works by focusing only on the middle
    integrand2 = (mid_x*(range0[1:] - range0[:-1]))[x0:(x1 + 1)]
    cdf2 = np.cumsum(integrand2) + 0.2
    lb_is_good = np.min(cdf2) < 0.25 and np.min(cdf2) > 0.15
    ub_is_good = np.max(cdf2) < 0.85 and np.max(cdf2) > 0.75
    if lb_is_good and ub_is_good:
        Q_vec = get_fitted_quantiles(percentiles, cdf2, range0, pstart, pend)
        return(Q_vec)

    # Rarely, a smaller smoothing parameter helps. 
    # TODO: implement backup test if this fails
    if bw_coef == 0.3:
        approximate_quantiles(x, percentiles, bw_coef = 0.03)
    else:
        pdb.set_trace()

