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

from STAR_outliers_testing_library import bin_data

def smooth_tail_subsection(curve, range0, bw_coef, cutoff, main_dist):
    smoothed_curve = smooth(curve[0], bw_method = bw_coef)(curve[1])
    if cutoff > np.percentile(main_dist, 50):
        smoothed_curve = smoothed_curve[(len(smoothed_curve) - len(range0)):]
        subrange = curve[1][(len(curve[1]) - len(range0)):]
    else:
        smoothed_curve = smoothed_curve[:len(range0)]
        subrange = curve[1][:len(range0)]
    deltas = subrange[1:] - subrange[:-1]
    smoothed_curve = smoothed_curve/np.sum(smoothed_curve[:-1]*deltas)
    return(smoothed_curve)

def plot_test(test_dist, fitted_curve, range0, exp_status, bw_coef, prefix,
              cutoff, outliers, name, curve = None, ignored_values = None):

    if not ignored_values is None:
        main_dist = test_dist[np.isin(test_dist, ignored_values) == False]
    else:
        main_dist = test_dist
     
    if curve != None:
        smoothed_curve = smooth_tail_subsection(curve, range0,
                                                np.max([1.5*bw_coef, 0.3]),
                                                cutoff, main_dist)
    else:
        smoothed_curve = smooth(main_dist, bw_method =  bw_coef)(range0)

    r_sq = pearsonr(smoothed_curve, fitted_curve)[0]**2
    test_name = (["tukey", "exp_tail_fit"])[np.array([exp_status]).astype(int)[0]]
    title = "field " + name + " vs fitted " + test_name 
    title += "\n(R^2 = " + str(r_sq)[0:6] + ")"

    message =  "empirical smoothed density"
    if not ignored_values is None:
        if len(ignored_values) > 0:
            message += "\n values ignored: " + str(ignored_values)
    plt.plot(range0, smoothed_curve, "g-", label = message)
    plt.plot(range0, fitted_curve, "r-", label = "fitted " + test_name + "density")

    num_outliers = len(outliers)
    outlier_label = "outlier threshold: " + str(cutoff) 
    outlier_label += " (" + str(num_outliers) + " outliers detected)"
    nbins = np.max([int(len(np.unique(main_dist))/300), 100])
    vals, counts = bin_data(main_dist, nbins)
    deltas = np.min(vals[1:] - vals[:-1])
    halfmax = 0.5*(np.max(counts)/np.sum(counts))/np.min(deltas)
    plt.plot(2*[cutoff], halfmax*np.arange(2), "k-", label = outlier_label)
    plt.hist(main_dist, bins = nbins, density = True, label = "real histogram")
    plt.xlabel('test statistic')
    plt.ylabel('density')
    p1, p99 = np.percentile(main_dist, 1), np.percentile(main_dist, 99)
    delta = (p99 - p1)/4
    plt.xlim([p1 - delta, cutoff + delta])
    plt.title(title)
    plt.legend()
    if not os.path.exists(prefix + "_outlier_plots"):
        os.mkdir(prefix + "_outlier_plots")
    plt.savefig(prefix + "_outlier_plots/" + name + ".png")
    plt.clf()
    return(r_sq)

def plot_highliers(plot_object, y_vec, data_dist, outliers, p50):
    if len(outliers[outliers > p50]) > 0:
        min_highlier = np.min(outliers[outliers > p50])
        high_cutoff = np.max(data_dist[data_dist < min_highlier]) 
        num_highliers = np.sum(outliers > p50)
        label = "upper bound: " +  str(num_highliers) 
        label += " values > " + str(high_cutoff)[0:6]
        plt.plot(np.repeat(high_cutoff, 2), y_vec, "k-", label = label)
    else:
        high_cutoff = np.max(data_dist)
        label = "no high outliers (max value: " + str(high_cutoff) + ")"
        plt.plot([high_cutoff], [0], "ko", label = label)
    return(high_cutoff)
        
def plot_lowliers(plot_object, y_vec, data_dist, outliers, p50):
    label = ""
    if len(outliers[outliers < p50]) > 0:
        if len(outliers[outliers > p50]) > 0:
            label += "\n"
        max_lowlier = np.max(outliers[outliers < p50])
        low_cutoff = np.min(data_dist[data_dist > max_lowlier])
        num_lowliers = np.sum(outliers < p50)
        label += "lower bound: " + str(num_lowliers) 
        label += " values < " + str(low_cutoff)[0:6]
        plt.plot(np.repeat(low_cutoff, 2),  y_vec, "k-", label = label)
    else:
        low_cutoff = np.min(data_dist)
        label += "no low outliers (min value: " + str(low_cutoff) + ")"
        plt.plot([low_cutoff], [0], "ko", label = label)
    return(low_cutoff)

def plot_data(data_dist, outliers, spike_vals, name, prefix):
    
    num_outliers = len(outliers)
    label0 = "feature distribution"
    nbins = np.max([int(len(np.unique(data_dist))/300), 100])
    vals, counts = bin_data(data_dist, nbins)
    deltas = np.min(vals[1:] - vals[:-1])
    halfmax = 0.5*(np.max(counts)/np.sum(counts))/np.min(deltas)
    if len(spike_vals) > 0:
        label0 += "\n removed vals: " + str(spike_vals)
    if len(outliers) == 0:
        label0 += " (no outliers)"
    
    p50 = np.percentile(data_dist, 50)
    high_cutoff = plot_highliers(plt, halfmax*np.arange(2), data_dist, outliers, p50)
    low_cutoff = plot_lowliers(plt, halfmax*np.arange(2), data_dist, outliers, p50)
    p1, p99 = np.percentile(data_dist, 1), np.percentile(data_dist, 99)
    delta = (np.min([p99, high_cutoff]) - np.max([p1, low_cutoff]))/4
    x_lims = [low_cutoff - delta, high_cutoff + delta]
    plot_condition = np.logical_and(data_dist >= x_lims[0],
                                    data_dist <= x_lims[1])
    plt.hist(data_dist[plot_condition], bins = nbins, density = True, label = label0)
    
    plt.xlabel('feature_value')
    plt.ylabel('density')
    plt.xlim(x_lims)
    plt.title("field " + name + " outlier cutoffs")
    plt.legend()
    if not os.path.exists(prefix + "_outlier_plots_untransformed"):
        os.mkdir(prefix + "_outlier_plots_untransformed")
    plt.savefig(prefix + "_outlier_plots_untransformed/" + name + ".png")
    plt.clf()
