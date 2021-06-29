import numpy as np
import pandas as pd
import os
import pdb
from copy import deepcopy as COPY
from matplotlib import pyplot as plt
from scipy.stats import norm
from scipy.stats import pearsonr
from scipy.stats import gaussian_kde as smooth
from scipy.stats import expon
from scipy.stats import norm
from scipy.stats import yeojohnson
from tqdm import tqdm

from STAR_outliers_plotting_library import plot_data
from STAR_outliers_plotting_library import plot_test
from STAR_outliers_testing_library import detect_exponential_data
from STAR_outliers_polishing_library import approximate_quantiles
from STAR_outliers_polishing_library import clean_data
from STAR_outliers_polishing_library import remove_worst_continuity_violations
from STAR_outliers_polishing_library import adjust_median_values

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

    # pdb.set_trace()
    
    W, W_ignored = remove_worst_continuity_violations(W)
    if len(np.unique(W)) < 30:
        bound = 90

    Q_vec = np.percentile(W, [10, 25, 50, 75, 90])
    W_main = W[np.logical_and(W <= Q_vec[4], W >= Q_vec[0])]
    if len(np.unique(Q_vec)) != 5 or len(np.unique(W_main)) < 30:
        Q_vec = approximate_quantiles(W, [10, 25, 50, 75, 90])

    A = Q_vec[2]
    
    IQR = Q_vec[3] - Q_vec[1]
    SK = (Q_vec[4] + Q_vec[0] - 2*Q_vec[2])/(Q_vec[4] - Q_vec[0])
    T = (Q_vec[4] - Q_vec[0])/(Q_vec[3] - Q_vec[1])
    phi = 0.6817766 + 0.0534282*SK + 0.1794771*T - 0.0059595*(T**2)
    B = (0.7413*IQR)/phi

    Q_vec2 = np.percentile(W, [100 - bound, 25, 50, 75, bound])
    if len(np.unique(Q_vec2)) != 5 or len(np.unique(W_main)) < 30:
        Q_vec2 = approximate_quantiles(W, [100 - bound, 25, 50, 75, bound])

    zv = norm.ppf(bound/100, 0, 1)
    UHS = Q_vec2[4] - Q_vec2[2]
    LHS = Q_vec2[2] - Q_vec2[0]
    g = (1/zv)*np.log(UHS/LHS)

    y = (W - A)/B        
    Q_vec3 = np.percentile(y, [100 - bound, 25, 50, 75, bound])
    if len(np.unique(Q_vec3)) != 5 or len(np.unique(W_main)) < 30:
        Q_vec3 = approximate_quantiles(y, [100 - bound, 25, 50, 75, bound])

    Q_ratio = (Q_vec3[4]*Q_vec3[0])/(Q_vec3[4] + Q_vec3[0])
    h = (2/(zv**2))*np.log(-g*Q_ratio)
    if np.isnan(h):
        h = 0   
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
    x_main = x[np.logical_and(x <= Q_vec[4], x >= Q_vec[0])]
    if len(np.unique(Q_vec)) != 5 or len(np.unique(x_main)) < 30:
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

def attempt_exponential_fit(x, x_spiked, name, pcutoff,
                            prefix, bw_coef, spike_vals,
                            exp_status, severe_outliers):

    p50 = np.percentile(x, 50)
    left_to_right = p50 - np.min(x) < np.max(x) - p50
    if not left_to_right:
        x = np.max(x) - x
    alpha_body = np.min([0.05, np.max([0.005, 500/len(x)])])
    tail_cutoff = (pcutoff - (1 - alpha_body))/alpha_body
    x_tail = x[x >= np.percentile(x, 100*(1 - alpha_body))]
    if len(np.unique(x_tail < 3)):
        alpha_body = 0.05
        tail_cutoff = (pcutoff - (1 - alpha_body))/alpha_body
        x_tail = x[x >= np.percentile(x, 100*(1 - alpha_body))]
    loc = np.min(x_tail)
    scale = np.mean(x_tail) - loc
    tail_unique, tail_counts = np.unique(x_tail, return_counts = True)
    if np.max(tail_counts)/np.sum(tail_counts) < 0.05:
        scale = (np.percentile(x_tail, 50) - loc)/np.log(2)
    range0 = expon.ppf(np.linspace(0, pcutoff, len(x_tail)), loc, scale)
    curve_dist = x[x >= np.percentile(x, 100*(1 - 2*alpha_body))]
    curve_range = expon.ppf(np.linspace(0, pcutoff, len(curve_dist)), loc, scale)
    curve = [curve_dist, curve_range]
    fitted_curve = expon.pdf(range0, loc, scale)
    cutoff = expon.ppf(tail_cutoff, loc, scale)
    x_outliers = x[x > cutoff]
    all_outliers = np.concatenate([severe_outliers, x_outliers])
    r_sq = plot_test(x_tail, fitted_curve, range0, exp_status, bw_coef, 
                     prefix, cutoff, all_outliers, name, curve)
    if not left_to_right:
        x_outliers = np.max(x) - x_outliers
        all_outliers = np.concatenate([severe_outliers, x_outliers])
        x = np.max(x) - x
    plot_data(x, all_outliers, spike_vals, name, prefix)
    x_spiked[np.isin(x_spiked, all_outliers)] = np.nan
    return(x_spiked, r_sq)

def attempt_tukey_fit(x, x_spiked, name, pcutoff, prefix, 
                      bound, bw_coef, spike_vals,
                      exp_status, severe_outliers):

    if bw_coef == 0.075:
        W = compute_w(yeojohnson((x - np.mean(x))/np.std(x))[0])
    else:
        W = compute_w(x)
    A, B, g, h, W_ignored = estimate_tukey_params(W, bound)
    z = np.random.normal(0, 1, 200000)
    fitted_TGH  = A + B*(1/(g + 1E-10))*(np.exp((g + 1E-10)*z)-1)*np.exp(h*(z**2)/2)
    delta = (np.percentile(W, 99) - np.percentile(W, 1))/2
    xlims = [np.percentile(W, 1) - delta, np.percentile(W, 99) + delta]
    range0 = np.linspace(xlims[0] - delta, xlims[1] + delta, 
                        np.max([250, int(len(W)/300)]))
    smooth_TGH = smooth(fitted_TGH, bw_method =  bw_coef)(range0)
    cutoff = np.percentile(fitted_TGH, pcutoff*100)
    x_outliers = x[W > cutoff]
    all_outliers = np.concatenate([severe_outliers, x_outliers])
    r_sq = plot_test(W, smooth_TGH, range0, exp_status, bw_coef, prefix, 
                     cutoff, all_outliers, name, ignored_values = W_ignored)
    
    plot_data(x, all_outliers, spike_vals, name, prefix)
    x_spiked[np.isin(x_spiked, all_outliers)] = np.nan
    return(x_spiked, r_sq)

def get_constrained_min(x_spiked, disallowed_vals):
    x = COPY(x_spiked)
    x_min = np.nanmin(x)
    if x_min in disallowed_vals:
        x = x[x != x_min]
        return(get_constrained_min(x, disallowed_vals))
    else:
        return(x_min)

def get_constrained_max(x_spiked, disallowed_vals):
    x = COPY(x_spiked)
    x_max = np.nanmax(x)
    if x_max in disallowed_vals:
        x = x[x != x_max]
        return(get_constrained_max(x, disallowed_vals))
    else:
        return(x_max)

def compute_outliers(x_spiked, name, prefix, bound, pcutoff):

    x_spiked = x_spiked.astype(float)
    x_spiked_old = COPY(x_spiked)
    x, severe_outliers, spike_vals = clean_data(x_spiked, name,
                                                prefix, 0, [], [])

    x_unique, x_counts = np.unique(x, return_counts = True)
    q5, q95 = np.percentile(x, [5, 95])
    x_main = x[np.logical_and(x <= q95, x >= q5)]
    bw_coef = 0.075
    if len(np.unique(x_main)) < 1000:
        bw_coef = 0.3
    if len(np.unique(x_main)) < 100:
        bw_coef = 0.5
    if len(np.unique(x_main)) < 30:
        bw_coef = 0.7
    
    exp_status = detect_exponential_data(x, np.max([10, int(len(x)/300)]))

    outlier_info = [name]
    old_count = np.sum(np.isnan(x_spiked)==False)

    if exp_status == True:
        x_spiked_new, r_sq = attempt_exponential_fit(x, x_spiked, name, pcutoff,
                                                     prefix, bw_coef, spike_vals,
                                                     exp_status, severe_outliers)
        outlier_info.append(np.sum(np.isnan(x_spiked_new)==False)/old_count)
        outlier_info.append(np.nanmin(x_spiked_old))
        outlier_info.append(get_constrained_min(x_spiked_new, spike_vals))
        outlier_info.append(np.nanpercentile(x, 50))
        outlier_info.append(get_constrained_max(x_spiked_new, spike_vals))
        outlier_info.append(np.nanmax(x_spiked_old))
        return(x_spiked_new, r_sq, severe_outliers, outlier_info)
    else:
        x_spiked_new, r_sq = attempt_tukey_fit(x, x_spiked, name, pcutoff, prefix,
                                               bound, bw_coef, spike_vals,
                                               exp_status, severe_outliers)
        outlier_info.append(np.sum(np.isnan(x_spiked_new)==False)/old_count)
        outlier_info.append(np.nanmin(x_spiked_old))
        outlier_info.append(get_constrained_min(x_spiked, spike_vals))
        outlier_info.append(np.nanpercentile(x, 50))
        outlier_info.append(get_constrained_max(x_spiked, spike_vals))
        outlier_info.append(np.nanmax(x_spiked_old))
        return(x_spiked_new, r_sq, severe_outliers, outlier_info)


def remove_all_outliers(input_file_name, index_name, bound, pcutoff):
    fields = pd.read_csv(input_file_name, delimiter = "\t", header = 0)
    field_names = fields.columns
    if not index_name is None:
        field_names = field_names[field_names != index_name]
        index_col = fields[index_name]
        fields = fields[field_names]
    field_cols = [fields.loc[:, name].to_numpy() for name in field_names]

    if bound is None:
        bound = np.min([pcutoff*100, 99])

    r_sq_vals = []
    names = []
    fields_with_poor_fits = []
    poor_r_sq_values = []
    severe_outlier_sets = []
    cleaned_field_cols = []
    outlier_info_sets = []
    for i in tqdm(range(len(field_names))):
        field = field_cols[i]
        unique_vals = np.unique(field)
        if len(unique_vals[np.isnan(unique_vals) == False]) >= 10:
            name = field_names[i]
            names.append(name)
            prefix = input_file_name.split(".")[0]
            output = compute_outliers(field, name, prefix, bound, pcutoff)
            cleaned_field_cols.append(output[0])
            r_sq_vals.append(output[1])
            severe_outlier_sets.append(output[2])
            outlier_info_sets.append(output[3])
            if output[1] < 0.8:
                fields_with_poor_fits.append(name)
                poor_r_sq_values.append(output[1])
        else:
            cleaned_field_cols.append(field)

    cleaned_data = pd.DataFrame(np.transpose(cleaned_field_cols))
    cleaned_data.columns = field_names
    if not index_name is None:
        cleaned_data[index_name] = index_col
    return(cleaned_data,
           r_sq_vals, names,
           fields_with_poor_fits,
           poor_r_sq_values,
           severe_outlier_sets,
           cleaned_field_cols,
           outlier_info_sets)
   
