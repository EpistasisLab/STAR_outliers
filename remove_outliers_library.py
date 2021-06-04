import numpy as np
import os
import pdb
import platform
from copy import deepcopy as COPY
from statsmodels.stats.stattools import medcouple 
from matplotlib import pyplot as plt
from scipy.stats import norm
from scipy.stats import pearsonr
from scipy.stats import gaussian_kde as smooth
from scipy.stats import expon
from scipy.stats import norm
from scipy.stats.mstats import gmean

from remove_outliers_plotting_library import plot_data
from remove_outliers_plotting_library import plot_test
from remove_outliers_testing_library import detect_exponential_data
from remove_outliers_testing_library import backup_test
from remove_outliers_testing_library import get_geometric_info
from remove_outliers_polishing_library import approximate_quantiles
from remove_outliers_polishing_library import clean_data
from remove_outliers_polishing_library import remove_worst_continuity_violations
from remove_outliers_polishing_library import adjust_median_values

# requires "module load R/3.5.3"
# If that is insufficient, then use install.packages("OpVaR") in an R environment before running this
# install.packages("OpVaR")
from rpy2.robjects.packages import importr
system = platform.uname()[0]
if system == "Linux":
    try:
        OpVaR_path = '/home/runner/work/_temp/Library'
        TGH = importr('OpVaR', lib_loc = OpVaR_path)
    except:
        exit()
        
if system == "Windows":
    try:
        OpVaR_path = 'D:/a/_temp/Library'
        TGH = importr('OpVaR', lib_loc = OpVaR_path)
    except:
        OpVaR_path = 'R-3.5.2/library'
        TGH = importr('OpVaR', lib_loc = OpVaR_path)

# source title: Outlier identification for skewed and/or 
#               heavy-tailed unimodal multivariate distributions

def estimate_tukey_params(W, bound):

    """
    Purpose
    -------
    To estimate tukey parameters for the distribution of deviation scores
    
    
    Parameters
    ----------
    W: the distribution of deviation scores
    bound: the highest percentile on which to fit the tukey parameters
           also, 100 - bound is the corresponding lowest percentile. 

    Returns
    -------
    A: tukey location parameter
    B: tukey scale parameter
    g: tukey skew parameter
    h: tukey tail heaviness parameter
    W_ignored: values of W that are too close to the median.
               very few values are too close in truly continuous data
               most data is not truly continuous and has an overabundance. 
               Note that the tukey only needs to fit the right side of 
               the W distribution, and ignored values are far to the left. 

    """

    W, W_ignored = remove_worst_continuity_violations(W)

    Q_vec = np.percentile(W, [10, 25, 50, 75, 90])
    if len(np.unique(Q_vec)) != 5:
        Q_vec = approximate_quantiles(W, [10, 25, 50, 75, 90])

    A = Q_vec[2]
    
    IQR = Q_vec[3] - Q_vec[1]
    SK = (Q_vec[4] + Q_vec[0] - 2*Q_vec[2])/(Q_vec[4] - Q_vec[0])
    T = (Q_vec[4] - Q_vec[0])/(Q_vec[3] - Q_vec[1])
    phi = 0.6817766 + 0.0534282*SK + 0.1794771*T - 0.0059595*(T**2)
    B = (0.7413*IQR)/phi

    Q_vec2 = np.percentile(W, [100 - bound, 25, 50, 75, bound])
    if len(np.unique(Q_vec2)) != 5:
        Q_vec2 = approximate_quantiles(W, [100 - bound, 25, 50, 75, bound])

    zv = norm.ppf(bound/100, 0, 1)
    UHS = Q_vec2[4] - Q_vec2[2]
    LHS = Q_vec2[2] - Q_vec2[0]
    g = (1/zv)*np.log(UHS/LHS)

    y = (W - A)/B        
    Q_vec3 = np.percentile(y, [100 - bound, 25, 50, 75, bound])
    if len(np.unique(Q_vec3)) != 5:
        Q_vec3 = approximate_quantiles(y, [100 - bound, 25, 50, 75, bound])

    Q_ratio = (Q_vec3[4]*Q_vec3[0])/(Q_vec3[4] + Q_vec3[0])
    h = (2/(zv**2))*np.log(-g*Q_ratio)
    if np.isnan(h):
        h = 0
    else:
        h = np.max([h, 0])    
    return((A, B, g, h, W_ignored))

def compute_w(x):

    """
    Purpose
    -------
    To convert x values into a statistic that quantifies
    deviation from the mean relative to skew and tail heaviness

    
    Parameters
    ----------
    x: numeric input numpy array and the original
       dataset that needs outliers to be removed

    Returns
    -------
    W: a numeric numpy array and the distribution 
       of deviation scores for the x values

    """

    Q_vec = np.percentile(x, [10, 25, 50, 75, 90])
    if len(np.unique(Q_vec)) != 5:
        Q_vec = approximate_quantiles(x, [10, 25, 50, 75, 90])
    x2 = adjust_median_values(x, Q_vec)
    
    c = 0.7413
    ASO_high = (x2 - Q_vec[2])/(2*c*(Q_vec[3] - Q_vec[2]))
    ASO_low = (Q_vec[2] - x2)/(2*c*(Q_vec[2] - Q_vec[1]))

    ASO = np.zeros(len(x2))
    ASO[x2 >= Q_vec[2]] = ASO_high[x2 >= Q_vec[2]]
    ASO[x2 < Q_vec[2]] = ASO_low[x2 < Q_vec[2]]
    
    W = norm.ppf((ASO + 1E-10)/(np.min(ASO) + np.max(ASO) + 2E-10))
    return(W)

def run_backup_test(x, x_spiked, name, prefix):
    content = clean_data(x_spiked, name, prefix, 0, [], [], False)
    x_geo, severe_outliers = content[0], content[1]
    num_main_values, gmean_decrease = get_geometric_info(x_geo)
    if num_main_values > 10:
        return(backup_test(x, x_spiked, name, prefix,
                           severe_outliers))
    else:
        return(backup_test(x_geo, x_spiked, name, prefix,
                           severe_outliers, gmean_decrease))

def attempt_exponential_fit(x, x_spiked, name, alt_tests, 
                            prefix, bw_coef, spike_vals,
                            exp_status, severe_outliers):

    p50 = np.percentile(x, 50)
    left_to_right = p50 - np.min(x) < np.max(x) - p50
    if not left_to_right:
        x = np.max(x) - x
    alpha_body = 0.05
    alpha_tail = (.9973 - (1 - alpha_body))/alpha_body
    x_tail = x[x >= np.percentile(x, 100*(1 - alpha_body))]
    loc = np.min(x_tail)
    scale = np.mean(x_tail) - loc
    tail_unique, tail_counts = np.unique(x_tail, return_counts = True)
    if np.max(tail_counts)/np.sum(tail_counts) < 0.05:
        scale = (np.percentile(x_tail, 50) - loc)/np.log(2)
    range0 = expon.ppf(np.linspace(0, 0.9973, len(x_tail)), loc, scale)
    curve_dist = x[x >= np.percentile(x, 100*(1 - 2*alpha_body))]
    curve_range = expon.ppf(np.linspace(0, 0.9973, len(curve_dist)), loc, scale)
    curve = [curve_dist, curve_range]
    fitted_curve = expon.pdf(range0, loc, scale)
    cutoff = expon.ppf(alpha_tail, loc, scale)
    x_outliers = x[x > cutoff]
    if not left_to_right:
        x_outliers = np.max(x) - x_outliers
    r_sq = plot_test(x_tail, fitted_curve, range0, exp_status, bw_coef, 
                     prefix, cutoff, x_outliers, name, curve)
    if r_sq < 0.6 and alt_tests == True:
        return(run_backup_test(x, x_spiked, name, prefix))
    plot_data(x, cutoff, x_outliers, spike_vals, name, prefix)
    all_outliers = np.union1d(severe_outliers, x_outliers)
    x_spiked[np.isin(x_spiked, all_outliers)] = np.nan
    return(x_spiked)

def attempt_tukey_fit(x, x_spiked, name, alt_tests, 
                      prefix, bound, bw_coef, spike_vals,
                      exp_status, severe_outliers):

    W = compute_w(x)
    A, B, g, h, W_ignored = estimate_tukey_params(W, bound)
    fitted_TGH = TGH.rgh(len(x), float(A), float(B), float(g), float(h))
    delta = (np.percentile(W, 99) - np.percentile(W, 1))/2
    xlims = [np.percentile(W, 1) - delta, np.percentile(W, 99) + delta]
    range0 = np.linspace(xlims[0] - delta, xlims[1] + delta, 
                        np.max([100, int(len(W)/300)]))
    smooth_TGH = smooth(fitted_TGH, bw_method =  bw_coef)(range0)
    cutoff = np.percentile(fitted_TGH, 99.73)
    x_outliers = np.unique(x[W > cutoff])
    r_sq = plot_test(W, smooth_TGH, range0, exp_status, bw_coef, prefix, 
                     cutoff, x_outliers, name, ignored_values = W_ignored)
    if r_sq < 0.8 and alt_tests == True:
        return(run_backup_test(x, x_spiked, name, prefix))
    plot_data(x, cutoff, x_outliers, spike_vals, name, prefix)
    all_outliers = np.union1d(severe_outliers, x_outliers)
    x_spiked[np.isin(x_spiked, all_outliers)] = np.nan
    return(x_spiked)

def compute_outliers(x_spiked, name, prefix, bound, alt_tests):

    x_spiked = x_spiked.astype(float)
    x, severe_outliers, spike_vals = clean_data(x_spiked, name,
                                                prefix, 0, [], [])
    x_unique, x_counts = np.unique(x, return_counts = True)
    bw_coef = 0.3
   
    exp_status = detect_exponential_data(x, np.max([10, int(len(x)/300)]))

    #TODO: something for this
    if exp_status == True: 
        return(attempt_exponential_fit(x, x_spiked, name, alt_tests,
                                       prefix, bw_coef, spike_vals,
                                       exp_status, severe_outliers))
    else:
        return(attempt_tukey_fit(x, x_spiked, name, alt_tests,
                                 prefix, bound, bw_coef, spike_vals,
                                 exp_status, severe_outliers))



    

   
