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
    if despike:
        return(x, severe_outliers, spikes)
    return(x, severe_outliers)

#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
# SECTION 2: code that corrects for real-world continuity violations
#            in computing W and the relevant parameter estimates
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------

def adjust_median_values(x, Q_vec):

    dists_sub50, dists_sup50 = x - Q_vec[2],  x - Q_vec[2]
    dists_sub50[dists_sub50 >= 0] = -np.inf
    dists_sup50[dists_sup50 <= 0] = np.inf
    x_sub50 = x[np.argmax(dists_sub50)]
    x_sup50 = x[np.argmin(dists_sup50)]
    n_sub50 = np.sum(x == x_sub50)
    n_sup50 = np.sum(x == x_sup50)
    p_sup50 = n_sup50/(n_sup50 + n_sub50)

    x2 = COPY(x)
    n_50 =  np.sum(x == Q_vec[2])
    choices = [x_sub50, x_sup50]
    p_vec = [1 - p_sup50, p_sup50]
    x2[x == Q_vec[2]] = np.random.choice(choices, n_50, True, p_vec)
    return(x2)

def remove_worst_continuity_violations(W):
    W_unique = np.unique(W)
    dists = W_unique[1:] - W_unique[:-1]
    if dists[0] > dists[1]:
        W = W[W != W_unique[0]]
        W_ignored = W_unique[0:1]
    elif dists[1] > dists[2] and dists[1] > dists[0]:
        W = W[W != W_unique[0]]
        W = W[W != W_unique[1]]
        W_ignored = W_unique[0:2]
    else:
        W_ignored = np.array([])
    return(W, W_ignored)

#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
# SECTION 3: code that computes quantiles from the continuous
#            approximation of a problematically discrete distribution
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
 
def get_fitted_quantiles(percentiles, fitted_cdf, range0, 
                         qstart, qend, good_cdf_tails):
    Q_vec = np.zeros(5)
    if good_cdf_tails == True: 
        Q_vec[0] = range0[np.where(fitted_cdf >= percentiles[0]/100)[0][0]]
    else:
        Q_vec[0] = qstart
    Q_vec[1] = range0[np.where(fitted_cdf >= 0.25)[0][0]]  
    Q_vec[2] = range0[np.where(fitted_cdf >= 0.5)[0][0]] 
    Q_vec[3] = range0[np.where(fitted_cdf >= 0.75)[0][0]]  
    if good_cdf_tails == True: 
        Q_vec[4] = range0[np.where(fitted_cdf >= percentiles[4]/100)[0][0]]
    else:
        Q_vec[4] = qend   
    return(Q_vec)

def approximate_quantiles(x, percentiles):

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
    q5, q95 = np.percentile(x, [5, 95])
    x_main = x[np.logical_and(x <= q95, x >= q5)]
    bw_coef = 0.075
    if len(np.unique(x_main)) < 1000:
        bw_coef = 0.3
    if len(np.unique(x_main)) < 100:
        bw_coef = 0.5
    if len(np.unique(x_main)) < 30:
        bw_coef = 0.7

    qstart = np.percentile(x, percentiles[0])
    q1, q20, q35, q65, q80, q99 = np.percentile(x, [0.5, 20, 35, 65, 80, 99.5])
    spacer = (q99 - q1)/2.75
    qend = np.percentile(x, percentiles[-1])
    range1 = np.linspace(q1 - spacer, q35, 200)
    range2 = np.linspace(q35, q65, 200)
    range3 = np.linspace(q65, q99 + spacer, 200)
    range0 = np.concatenate([range1[:-1], range2[:-1], range3])
    x0 = np.where(range0 >= q20)[0][0]
    x1 = np.where(range0 <= q80)[0][-1]
    x_bounded = x[np.logical_and(x >= q1, x <= q99)]
    smooth_x = smooth(x_bounded, bw_method = bw_coef)(range0)
    mid_x = (smooth_x[:-1] + smooth_x[1:])/2

    integrand1 = (mid_x*(range0[1:] - range0[:-1]))
    cdf1 = np.cumsum(integrand1)
    lb_is_good = np.min(cdf1) < 0.05 and np.min(cdf1) > -0.05
    ub_is_good = np.max(cdf1) < 1.05 and np.max(cdf1) > 0.95
    if lb_is_good and ub_is_good:
        Q_vec = get_fitted_quantiles(percentiles, cdf1, range0, 
                                     qstart, qend, True)
        return(Q_vec)

    # This is a reasonable backup method if the cdf fits the tails poorly
    integrand2 = (mid_x*(range0[1:] - range0[:-1]))[x0:(x1 + 1)]
    cdf2 = np.cumsum(integrand2) + 0.2
    lb_is_good = np.min(cdf2) < 0.25 and np.min(cdf2) > 0.15
    ub_is_good = np.max(cdf2) < 0.85 and np.max(cdf2) > 0.75
    if lb_is_good and ub_is_good:
        Q_vec = get_fitted_quantiles(percentiles, cdf2, range0, 
                                     qstart, qend, False)
        return(Q_vec)

    else:
        pdb.set_trace()

