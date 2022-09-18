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
# SECTION 2: Removes spikes from the data.
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

#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
# SECTION 2: code that corrects for various continuity violations.
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
 
def get_fitted_quantiles(percentiles, fitted_cdf, range0, 
                         qstart, qend, good_cdf_tails):
    Q_vec = np.zeros(5)
    if good_cdf_tails == True:
        try:
            Q_vec[0] = range0[np.where(fitted_cdf >= percentiles[0]/100)[0][0]]
        except:
            return(Q_vec)
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
    q1, q5, q35, q65, q95, q99 = np.percentile(x, [0.5, 5, 35, 65, 95, 99.5])
    qstart = np.percentile(x, percentiles[0])
    spacer = (q99 - q1)/2.75
    qend = np.percentile(x, percentiles[-1])
    range1 = np.linspace(q1 - spacer, q35, 200)
    range2 = np.linspace(q35, q65, 200)
    range3 = np.linspace(q65, q99 + spacer, 200)
    range0 = np.concatenate([range1[:-1], range2[:-1], range3])
    x_bounded = x[np.logical_and(x >= q1, x <= q99)]
    smooth_x = smooth(x_bounded, bw_method = 'silverman')(range0)
    mid_x = (smooth_x[:-1] + smooth_x[1:])/2
    integrand1 = (mid_x*(range0[1:] - range0[:-1]))
    cdf1 = np.cumsum(integrand1)
    good_bounds = np.abs(np.min(cdf1) - 0) + np.abs(np.max(cdf1) - 1) < 0.1
    Q_vec = get_fitted_quantiles(percentiles, cdf1, range0, qstart, qend, True)
    if good_bounds and len(np.unique(Q_vec)) == 5:
        return(Q_vec)

    for i in [0.01, 0.05, 0.1, 0.5, 1, 5, 10]:
        smooth_x = smooth(x_bounded, bw_method = i)(range0)
        mid_x = (smooth_x[:-1] + smooth_x[1:])/2
        integrand1 = (mid_x*(range0[1:] - range0[:-1]))
        cdf1 = np.cumsum(integrand1)
        good_bounds = np.abs(np.min(cdf1) - 0) + np.abs(np.max(cdf1) - 1) < 0.1
        Q_vec = get_fitted_quantiles(percentiles, cdf1, range0, qstart, qend, True)
        if good_bounds and len(np.unique(Q_vec)) == 5:
            return(Q_vec)

    mu = np.mean(x)
    sig = np.std(x)
    return(norm(mu, sig).ppf(np.array(percentiles)/100))
        

