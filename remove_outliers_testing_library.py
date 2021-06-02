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
# SECTION 1: code to determine if an exponential tail fit is a good main test
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------

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

#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
# SECTION 2: code to determine if the backup geometric test is appropriate
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------

def get_gmean_decrease(x):
    total_decrease = 1
    decreases = []
    x_new = COPY(x)
    while total_decrease > 0.05:
        x_unique, x_counts = np.unique(x_new, return_counts = True)
        decrease = 1 - np.max(x_counts)/np.sum(x_counts)
        decreases.append(decrease)
        total_decrease = total_decrease*decrease
        x_new = x_new[x_new != x_unique[np.argmax(x_counts)]]
    return(gmean(decreases))

def get_geometric_info(x):
    x_unique, x_counts = np.unique(x, return_counts = True)
    sorted_indices = np.flip(np.argsort(x_counts))
    cdf = np.cumsum(x_counts[sorted_indices])/np.sum(x_counts)
    p1 = np.where(cdf >= 0.025)[0][0]
    p99 = np.where(cdf <= 0.975)[0][-1]
    num_main_values = len(cdf[p1:p99])
    gmean_decrease = get_gmean_decrease(x)
    return(num_main_values, gmean_decrease)

#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
# SECTION 3: code for the backup test if the main test fits the data poorly
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------

def adjusted_IQR(x, x_spiked, name, severe_outliers):
    message = "The adjusted IQR test is being used for feature "
    message += name + " because the main test fit the data poorly."
    print(message)
    sampled_indices = np.random.choice(np.arange(len(x)), (1000, 4000))
    MC = np.mean(medcouple(x[sampled_indices]))
    # integrate N(0,1) from -inf to 2.7822 gets 99.73
    C = (2.7822 - 0.675)/1.35
    Q13 = np.nanpercentile(x, [25, 75])
    IQR = Q13[1] - Q13[0]
    ub = Q13[1] + C*IQR*np.exp(3.87*MC)
    lb = Q13[0] - C*IQR*np.exp(-3.79*MC)
    outliers = x[np.logical_or(x < lb, x > ub)]
    all_outliers = np.union1d(severe_outliers, outliers)
    x_spiked[np.isin(x_spiked, all_outliers)] = np.nan
    return(x_spiked, outliers)

def geometric_test(x, x_spiked, name, mean_decrease, severe_outliers):
    message = "The geometric test is being used for feature "
    message += name + " because the main test fit the data poorly."
    print(message)
    x_unique, x_counts = np.unique(x, return_counts = True)
    sorted_indices = np.flip(np.argsort(x_counts))
    cdf = 1 - np.cumprod([mean_decrease]*len(x_unique))
    cutoff_index = np.where(cdf <= 0.9973)[0][-1] + 2
    outliers = x_unique[sorted_indices[cutoff_index:]]
    all_outliers = np.union1d(severe_outliers, outliers)
    x_spiked[np.isin(x_spiked, all_outliers)] = np.nan
    return(x_spiked, outliers)

def backup_test(x, x_spiked, name, prefix,
                severe_outliers, mean_decrease = None):

    if mean_decrease is None:
        x_spiked, outliers = adjusted_IQR(x, x_spiked, name, severe_outliers)
        test = " (adjusted IQR)"
    else:
        x_spiked, outliers = geometric_test(x, x_spiked, name,
                                            mean_decrease, severe_outliers)
        test = " (geometric test)"

    label2 = "outliers (n = " + str(len(outliers)) + ")"
    plt.hist(x[np.isin(x, outliers) == False], bins = 100, label = "inliers")
    plt.hist(x[np.isin(x, outliers)], bins = 100, label = label2)
    plt.xlabel('feature_value')
    plt.ylabel('count')
    p1, p99 = np.percentile(x, 1), np.percentile(x, 99)
    delta = (p99 - p1)/2
    plt.xlim([p1 - delta, p99 + delta])
    plt.title("field " + name + " outlier cutoffs" + test)
    plt.legend()
    if not os.path.exists(prefix + "_outlier_plots_untransformed"):
        os.mkdir(prefix + "_outlier_plots_untransformed")
    plt.savefig(prefix + "_outlier_plots_untransformed/" + name + ".png")
    plt.clf()
    return(x_spiked)



