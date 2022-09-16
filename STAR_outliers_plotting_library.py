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

def get_len_4_float(value):
    str_value = str(value)
    if len(str(value)) > 4:
        offset = 0
        if "-" in str(value)[0:4]:
            offset += 1
            if "." in str(value)[0:5]:
                offset += 1
        elif "." in str(value)[0:4]:
            offset += 1
        str_value = str(value)[0:(4 + offset)]
    return(str_value)

def compute_overlap(W, fitted_TGH, cutoff, n_bins):

    W_main = W[W <= cutoff]
    Wx, Wy = bin_data(W_main, n_bins)
    dx = Wx[1] - Wx[0]
    lbs, ubs = (Wx - 0.5*dx).reshape(-1, 1), (Wx + 0.5*dx).reshape(-1, 1)
    TGH_bins = np.logical_and(fitted_TGH >= lbs, fitted_TGH < ubs)
    TGHy = np.sum(TGH_bins, axis = 1)
    Wy_norm = Wy/len(W_main)
    TGHy_norm = TGHy/len(fitted_TGH)
    Wy_overrun = np.sum(np.max([Wy_norm - TGHy_norm, np.zeros(len(Wy))], axis = 0))
    return(1 - np.sum(Wy_overrun))

def plot_highliers(plot_object, y_vec, data_dist, outliers, p50):
    if len(outliers[outliers > p50]) > 0:
        min_highlier = np.min(outliers[outliers > p50])
        high_cutoff = np.max(data_dist[data_dist < min_highlier]) 
        num_highliers = np.sum(outliers > p50)
        label = "upper bound: " +  str(num_highliers)
        str_high_cutoff = get_len_4_float(high_cutoff)
        label += " values > " + str_high_cutoff
        plt.plot(np.repeat(high_cutoff, 2), y_vec, "k-", label = label)
    else:
        high_cutoff = np.max(data_dist)
        str_high_cutoff = get_len_4_float(high_cutoff)
        label = "no high outliers (max value: " + str_high_cutoff + ")"
        plt.plot([high_cutoff], [0], "ko", label = label)
    return(high_cutoff)
        
def plot_lowliers(plot_object, y_vec, data_dist, outliers, p50):
    label = ""
    if len(outliers[outliers < p50]) > 0:
        max_lowlier = np.max(outliers[outliers < p50])
        low_cutoff = np.min(data_dist[data_dist > max_lowlier])
        num_lowliers = np.sum(outliers < p50)
        label += "lower bound: " + str(num_lowliers)
        str_low_cutoff = get_len_4_float(low_cutoff)
        label += " values < " + str_low_cutoff
        plt.plot(np.repeat(low_cutoff, 2),  y_vec, "k-", label = label)
    else:
        low_cutoff = np.min(data_dist)
        str_low_cutoff = get_len_4_float(low_cutoff)
        label += "no low outliers (min value: " + str_low_cutoff + ")"
        plt.plot([low_cutoff], [0], "ko", label = label)
    return(low_cutoff)

def plot_x(x, x_outliers, spike_vals, name, prefix, n_bins):

    label0 = "feature distribution"
    vals, counts = bin_data(x, n_bins)
    deltas = np.min(vals[1:] - vals[:-1])
    halfmax = 0.5*(np.max(counts)/np.sum(counts))/np.min(deltas)
    if len(spike_vals) > 0:
        ignored_val_stubs = [get_len_4_float(val) for val in spike_vals]
        label0 += " (omitted spikes: " + str(ignored_val_stubs) + ")"
    if len(x_outliers) == 0:
        label0 += " (no outliers)"
    
    p50 = np.percentile(x, 50)
    high_cutoff = plot_highliers(plt, halfmax*np.arange(2), x, x_outliers, p50)
    low_cutoff = plot_lowliers(plt, halfmax*np.arange(2), x, x_outliers, p50)
    p1, p99 = np.percentile(x, 1), np.percentile(x, 99)
    delta = (np.min([p99, high_cutoff]) - np.max([p1, low_cutoff]))/4
    x_lims = [low_cutoff - delta, high_cutoff + delta]
    plot_condition = np.logical_and(x >= x_lims[0],
                                    x <= x_lims[1])
    plt.hist(x[plot_condition], bins = n_bins, density = True, label = label0)
    
    plt.xlabel('feature value')
    plt.ylabel('density')
    plt.xlim(x_lims)
    plt.title("field " + name + " outlier cutoffs")
    plt.legend(loc = "upper left")
    if not os.path.exists(prefix + "_outlier_plots_untransformed"):
        os.mkdir(prefix + "_outlier_plots_untransformed")
    plt.savefig(prefix + "_outlier_plots_untransformed/" + name + ".png")
    plt.clf()

def plot_W(W, fitted_TGH, name, prefix, cutoff, n_bins):

    fit = compute_overlap(W, fitted_TGH, cutoff, n_bins)
    label1 = "fitted distribution (area overlap = " + get_len_4_float(fit) + ")"
    label2 = "actual distribution"
    plt.hist(fitted_TGH, bins = n_bins, density = True, fc=(0, 0, 1, 0.5), label = label1)
    plt.hist(W, bins = n_bins, density = True, fc=(1, 0, 0, 0.5), label = label2)
    outlier_label = "outlier threshold: " + get_len_4_float(cutoff) + " ("
    outlier_label += get_len_4_float(100*np.sum(W > cutoff)/len(W))
    outlier_label += "% exceding outlier cutoff)"
    plt.plot([cutoff, cutoff], [0, 0.25], label = outlier_label)
    plt.legend()
    if not os.path.exists(prefix + "_outlier_plots"):
        os.mkdir(prefix + "_outlier_plots")
    plt.savefig(prefix + "_outlier_plots/" + name + ".png")
    plt.clf()

    return(fit)
