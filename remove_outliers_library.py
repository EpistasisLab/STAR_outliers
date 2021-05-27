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

# requires "module load R/3.5.3"
# If that is insufficient, then use install.packages("OpVaR") in an R environment before running this
# install.packages("OpVaR")
from rpy2.robjects.packages import importr
TGH = importr('OpVaR')

def bin_data(x, n_bins):
    delta = 0.000001*np.mean(np.abs(x))
    bounds = np.linspace(np.min(x), np.max(x) + delta, n_bins).reshape(-1, 1)
    lbs, ubs = bounds[:-1], bounds[1:]
    bins = np.logical_and(x >= lbs, x < ubs)
    real_domain = (ubs.reshape(-1) + lbs.reshape(-1))/2
    real_range = np.sum(bins, axis = 1)
    return(real_domain, real_range)

def gen_exp(params, x):
    A, B = params
    y_est = A*np.exp(B*x)
    return(y_est)

def detect_monotonic_data(x, n_bins):
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

    return(exp_status1 or exp_status2)

def approximate_quantiles(x, quantiles, bw_coef = 0.3):

    pstart = np.percentile(x, quantiles[0])
    p1, p20, p35, p65, p80, p99 = np.percentile(x, [0.5, 20, 35, 65, 80, 99.5])
    spacer = (p99 - p1)/2.75
    pend = np.percentile(x, quantiles[-1])
    range1 = np.linspace(p1 - spacer, p35, 200)
    range2 = np.linspace(p35, p65, 200)
    range3 = np.linspace(p65, p99 + spacer, 200)
    range = np.concatenate([range1[:-1], range2[:-1], range3])
    x0 = np.where(range >= p20)[0][0]
    x1 = np.where(range <= p80)[0][-1]

    x_bounded = x[np.logical_and(x >= p1, x <= p99)]
    smooth_x = smooth(x_bounded, bw_method = bw_coef)(range)
    mid_x = (smooth_x[:-1] + smooth_x[1:])/2

    # if there are too few unique values to uniquely define p20 from nearby quantiles
    # then this method works if the distribution isn't too wierd
    integrand1 = (mid_x*(range[1:] - range[:-1]))
    cdf1 = np.cumsum(integrand1)
    lb_is_good = np.min(cdf1) < 0.05 and np.min(cdf1) > -0.05
    ub_is_good = np.max(cdf1) < 1.05 and np.max(cdf1) > 0.95
    good_bounds = lb_is_good and ub_is_good

    if good_bounds == True:
        Q_vec = np.zeros(5)
        if range[np.where(cdf1 >= 0.25)[0][0]] <= pstart: 
            Q_vec[0] = range[np.where(cdf1 >= quantiles[0]/100)[0][0]]
        else:
            Q_vec[0] = pstart
        Q_vec[1] = range[np.where(cdf1 >= 0.25)[0][0]]  
        Q_vec[2] = range[np.where(cdf1 >= 0.5)[0][0]] 
        Q_vec[3] = range[np.where(cdf1 >= 0.75)[0][0]]  
        if range[np.where(cdf1 >= 0.75)[0][0]] >= pend: 
            Q_vec[4] = range[np.where(cdf1 >= quantiles[4]/100)[0][0]]
        else:
            Q_vec[4] = pend   
        return(Q_vec)

    # if there are many unique values but the distribution is wierd
    # then this method works by focusing only on the middle
    integrand2 = (mid_x*(range[1:] - range[:-1]))[x0:(x1 + 1)]
    cdf2 = np.cumsum(integrand2) + 0.2
    lb_is_good = np.min(cdf2) < 0.25 and np.min(cdf2) > 0.15
    ub_is_good = np.max(cdf2) < 0.85 and np.max(cdf2) > 0.75
    good_bounds = lb_is_good and ub_is_good

    if good_bounds == True:
        Q_vec = np.zeros(5)
        if range[np.where(cdf2 >= 0.25)[0][0]] <= pstart: 
            Q_vec[0] = range[np.where(cdf2 >= quantiles[0]/100)[0][0]]
        else:
            Q_vec[0] = pstart
        Q_vec[1] = range[x0:(x1 + 1)][np.where(cdf2 >= 0.25)[0][0]]  
        Q_vec[2] = range[x0:(x1 + 1)][np.where(cdf2 >= 0.5)[0][0]] 
        Q_vec[3] = range[x0:(x1 + 1)][np.where(cdf2 >= 0.75)[0][0]]   
        Q_vec[4] = np.max([pend, range[x0:(x1 + 1)][np.where(cdf2 >= 0.75)[0][0]]])
        if range[np.where(cdf2 >= 0.75)[0][0]] >= pend: 
            Q_vec[4] = range[np.where(cdf2 >= quantiles[4]/100)[0][0]]
        else:
            Q_vec[4] = pend         
        return(Q_vec)

    if bw_coef == 0.3:
        approximate_quantiles(x, quantiles, bw_coef = 0.03)
    else:
        pdb.set_trace()
 
def estimate_tukey_params(W, name, bound):

    body = np.logical_and(W <= np.percentile(W, 99), W >= np.percentile(W, 1))
    W_main = W[body]
    W_unique = np.unique(W_main)
    if len(W_unique) < 4:
        print("\noutliers cannot be computed for " + name + " because too much probability mass exists in 3 or fewer unique values.")
        print("\nyour dataset may contain a probability mass spike that should be removed.\n\n")
        return(0, 0, 0, 0, 0)
    dists = W_unique[1:] - W_unique[:-1]
    try:
        if dists[0] > dists[1]:
            W = W_main[W_main != W_unique[0]]
            W_ignored = W_unique[0:1]
        elif dists[1] > dists[2] and dists[1] > dists[0]:
            W = W_main[W_main != W_unique[0]]
            W = W[W != W_unique[1]]
            W_ignored = W_unique[0:2]
        else:
            W_ignored = np.array([])
    except:
        pdb.set_trace()


    Q_vec = np.percentile(W, [10, 25, 50, 75, 90])
    if Q_vec[0] == Q_vec[4]:
        print("\nonly one unique value in " + name + " occupies all percentiles 10 through 90")
        print("\nyour dataset may contain a probability mass spike that should be removed.\n\n")
        return(0, 0, 0, 0, 0)
    if len(np.unique(Q_vec)) != 5:
        Q_vec = approximate_quantiles(W, [10, 25, 50, 75, 90])
    if not np.all(Q_vec[1:] > Q_vec[:4]):
        pdb.set_trace()

    A = Q_vec[2]

    IQR = Q_vec[3] - Q_vec[1]
    SK = (Q_vec[4] + Q_vec[0] - 2*Q_vec[2])/(Q_vec[4] - Q_vec[0])
    T = (Q_vec[4] - Q_vec[0])/(Q_vec[3] - Q_vec[1])
    phi = 0.6817766 + 0.0534282*SK + 0.1794771*T - 0.0059595*(T**2)
    B = (0.7413*IQR)/phi

    Q_vec2 = np.percentile(W, [100 - bound, 25, 50, 75, bound])
    if len(np.unique(Q_vec2)) != 5:
        Q_vec2 = approximate_quantiles(W, [100 - bound, 25, 50, 75, bound])
    if not np.all(Q_vec2[1:] > Q_vec2[:4]):
        pdb.set_trace()

    zv = norm.ppf(bound/100, 0, 1)
    UHS = Q_vec2[4] - Q_vec2[2]
    LHS = Q_vec2[2] - Q_vec2[0]
    g = (1/zv)*np.log(UHS/LHS)

    y = (W - A)/B
    if np.any(y == np.inf):
        pdb.set_trace()
        
    Q_vec3 = np.percentile(y, [100 - bound, 25, 50, 75, bound])
    if len(np.unique(Q_vec3)) != 5:
        Q_vec3 = approximate_quantiles(y, [100 - bound, 25, 50, 75, bound])
    if not np.all(Q_vec3[1:] > Q_vec3[:4]):
        pdb.set_trace()
    Q_ratio = (Q_vec3[4]*Q_vec3[0])/(Q_vec3[4] + Q_vec3[0])
    h = (2/(zv**2))*np.log(-g*Q_ratio)
    if np.isnan(h):
        h = 0
    else:
        h = np.max([h, 0])    
    return((A, B, g, h, W_ignored))

def compute_w(x):

    # values that equal the median get a w of -inf
    # this messes up the fit with a spike at a high negative location.
    # hence, each median value is randomly converted to
    # either the next smallest or next largest existing value

    Q_vec = np.percentile(x, [10, 25, 50, 75, 90])
    if len(np.unique(Q_vec)) != 5:
        Q_vec = approximate_quantiles(x, [10, 25, 50, 75, 90])
    if not np.all(Q_vec[1:] > Q_vec[:4]):
        pdb.set_trace()
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

    c = 0.7413
    ASO_high = (x2 - Q_vec[2])/(2*c*(Q_vec[3] - Q_vec[2]))
    ASO_low = (Q_vec[2] - x2)/(2*c*(Q_vec[2] - Q_vec[1]))
    
    ASO = np.zeros(len(x2))
    ASO[x2 >= Q_vec[2]] = ASO_high[x2 >= Q_vec[2]]
    ASO[x2 < Q_vec[2]] = ASO_low[x2 < Q_vec[2]]
    
    ASO = (ASO + 1E-10)/(np.min(ASO) + np.max(ASO) + 2E-10)
    return(norm.ppf(ASO))
    
def plot_test(test_dist, fitted_curve, range, exp_status, bw_coef, prefix,
              cutoff, outliers, name, curve = None, ignored_values = None):

    if not ignored_values is None:
        main_dist = test_dist[np.isin(test_dist, ignored_values) == False]
    else:
        main_dist = test_dist
     
    if curve != None:
        smoothed_curve = smooth(curve[0], bw_method =  bw_coef)(curve[1])
        if cutoff > np.percentile(main_dist, 50):
            smoothed_curve = smoothed_curve[(len(smoothed_curve) - len(range)):]
            subrange = curve[1][(len(curve[1]) - len(range)):]
            deltas = subrange[1:] - subrange[:-1]
            smoothed_curve = smoothed_curve/np.sum(smoothed_curve[:-1]*deltas)
        else:
            smoothed_curve = smoothed_curve[:len(range)]
            subrange = curve[1][:len(range)]
            deltas = subrange[1:] - subrange[:-1]
            smoothed_curve = smoothed_curve/np.sum(smoothed_curve[1:]*deltas)

    else:
        smoothed_curve = smooth(main_dist, bw_method =  bw_coef)(range)
    r_sq = pearsonr(smoothed_curve, fitted_curve)[0]**2
    test_name = (["tukey", "exp_tail_fit"])[np.array([exp_status]).astype(int)[0]]
    title = "field " + name + " vs fitted " + test_name 
    title += "\n(R^2 = " + str(r_sq)[0:6] + ")"

    message =  "empirical smoothed density"
    if not ignored_values is None:
        if len(ignored_values) > 0:
            message += "\n values ignored: " + str(ignored_values)
    plt.plot(range, smoothed_curve, "g-", label = message)
    plt.plot(range, fitted_curve, "r-", label = "fitted " + test_name + "density")

    num_outliers = len(outliers)
    outlier_label = "outlier threshold: " + str(cutoff) 
    outlier_label += " (" + str(num_outliers) + " outliers detected)"
    nbins = np.max([int(len(main_dist)/300), 100])
    vals, counts = bin_data(main_dist, nbins)
    deltas = np.min(vals[1:] - vals[:-1])
    halfmax = 0.5*(np.max(counts)/np.sum(counts))/np.min(deltas)
    plt.plot(2*[cutoff], halfmax*np.arange(2), "k-", label = outlier_label)
    plt.hist(main_dist, bins = nbins, density = True, label = "real histogram")
    plt.xlabel('test statistic')
    plt.ylabel('density')
    p1, p99 = np.percentile(main_dist, 1), np.percentile(main_dist, 99)
    delta = (p99 - p1)/2
    plt.xlim([p1 - delta, p99 + delta])
    plt.title(title)
    plt.legend()
    if not os.path.exists(prefix + "_outlier_plots"):
        os.mkdir(prefix + "_outlier_plots")
    plt.savefig(prefix + "_outlier_plots/" + name + ".png")
    plt.clf()
    return(r_sq)

def plot_data(data_dist, cutoff, outliers, spike_vals, name, prefix):
    num_outliers = len(outliers)
    label0 = "feature distribution"
    nbins = np.max([int(len(data_dist)/300), 100])
    vals, counts = bin_data(data_dist, nbins)
    deltas = np.min(vals[1:] - vals[:-1])
    halfmax = 0.5*(np.max(counts)/np.sum(counts))/np.min(deltas)
    if len(spike_vals) > 0:
        label0 += "\n removed vals: " + str(vals_to_isolate)
    if len(outliers) == 0:
        label0 += " (no outliers)"
    plt.hist(data_dist, bins = nbins, density = True, label = label0)
    p50 = np.percentile(data_dist, 50)
    if len(outliers[outliers > p50]) > 0:
        min_highlier = np.min(outliers[outliers > p50])
        high_cutoff = np.max(data_dist[data_dist < min_highlier]) 
        num_highliers = np.sum(outliers > p50)
        label1 = "upper bound: " +  str(num_highliers) + " values > " + str(high_cutoff)[0:6]
        plt.plot(np.repeat(high_cutoff, 2), halfmax*np.arange(2), "k-", label = label1)
    if len(outliers[outliers < p50]) > 0:
        if len(outliers[outliers > p50]) > 0:
            label2 = "\n"
        else:
            label2 = ""
        max_lowlier = np.max(outliers[outliers < p50])
        low_cutoff = np.min(data_dist[data_dist > max_lowlier])
        num_lowliers = np.sum(outliers < p50)
        label2 += "lower bound: " + str(num_lowliers) + " values < " + str(low_cutoff)[0:6]
        plt.plot(np.repeat(low_cutoff, 2),  halfmax*np.arange(2), "k-", label = label2)
    plt.xlabel('feature_value')
    plt.ylabel('density')
    p1, p99 = np.percentile(data_dist, 1), np.percentile(data_dist, 99)
    delta = (p99 - p1)/2
    plt.xlim([p1 - delta, p99 + delta])
    plt.title("field " + name + " outlier cutoffs")
    plt.legend()
    if not os.path.exists(prefix + "_outlier_plots_untransformed"):
        os.mkdir(prefix + "_outlier_plots_untransformed")
    plt.savefig(prefix + "_outlier_plots_untransformed/" + name + ".png")
    plt.clf()

def adjusted_IQR(x, x_spiked, name):
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
    x_spiked[np.isin(x_spiked, outliers)] = np.nan
    return(x_spiked, outliers)

def geometric_test(x, x_spiked, name, mean_decrease):
    message = "The geometric test is being used for feature "
    message += name + " because the main test fit the data poorly."
    print(message)
    x_unique, x_counts = np.unique(x, return_counts = True)
    sorted_indices = np.flip(np.argsort(x_counts))
    cdf = 1 - np.cumprod([mean_decrease]*len(x_unique))
    cutoff_index = np.where(cdf <= 0.9973)[0][-1] + 2
    outliers = x_unique[sorted_indices[cutoff_index:]]
    x_spiked[np.isin(x_spiked, outliers)] = np.nan
    return(x_spiked, outliers)

def backup_test(x_spiked, name, prefix, mean_decrease = None):
    x = COPY(x_spiked)[np.isnan(x_spiked)==False]
    x = remove_severe_outliers(x, name)
    if mean_decrease is None:
        x_spiked, outliers = adjusted_IQR(x, x_spiked, name)
        test = " (adjusted IQR)"
    else:
        x_spiked, outliers = geometric_test(x, x_spiked, name, mean_decrease)
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
    range = p_high - p_low
    lb = p_low - 5*range
    ub = p_high + 5*range
    severe_outliers = x[np.logical_or(x < lb, x > ub)]
    message = "The following severe outliers were removed for feature "
    message += name + ": " + str(np.unique(severe_outliers))
    if len(severe_outliers > 0):
        print(message)
    return(x[np.isin(x, severe_outliers) == False])

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
        
def compute_outliers(x_spiked, name, prefix, bound):

    x_spiked = x_spiked.astype(float)
    x = COPY(x_spiked)[np.isnan(x_spiked)==False]
    x, spikes, decreases = remove_spikes(x, x_spiked, name, prefix, 0, [], [])
    if len(spikes) > 3:
        return(backup_test(x_spiked, name, prefix, gmean(decreases)))

    x = remove_severe_outliers(x, name) 
    x_unique, x_counts = np.unique(x, return_counts = True)

    bw_coef = 0.3    
    exp_status = detect_exponential_data(x, np.max([10, int(len(x)/300)]))

    #TODO: something for this
    if exp_status == True:
        print(name)    
        spike_vals = np.array([])
        alpha_body = 0.05
        alpha_tail = (.9973 - (1 - alpha_body))/alpha_body
        x_tail = x[x >= np.percentile(x, 100*(1 - alpha_body))]
        loc = np.min(x_tail)
        scale = (np.percentile(x_tail - loc, 50)/np.log(2))
        if scale == 0:
            print("outliers have not been removed from feature " + name + ".")
            print("the minimum value contains more than 50% of the probability mass.")
            print("you need to specify a list of spike values for each input feature.")
            return([])
        range = expon.ppf(np.linspace(0, 0.9973, len(x_tail)), loc, scale)
        curve_dist = x[x >= np.percentile(x, 100*(1 - 2*alpha_body))]
        curve_range = expon.ppf(np.linspace(0, 0.9973, len(curve_dist)), loc, scale)
        curve = [curve_dist, curve_range]
        fitted_curve = expon.pdf(range, loc, scale)
        cutoff = expon.ppf(alpha_tail, loc, scale)
        if cutoff > np.percentile(x, 50):
            outliers = x[x > cutoff]
        else:
            outliers = x[x < cutoff]
        r_sq = plot_test(x_tail, fitted_curve, range, exp_status, bw_coef, 
                         prefix, cutoff, outliers, name, curve)
        if r_sq < 0.6:
            num_main_values, gmean_decrease = get_geometric_info(x)
            if num_main_values > 10: 
                return(backup_test(x_spiked, name, prefix))
            else:
                return(backup_test(x_spiked, name, prefix, gmean_decrease))
        plot_data(x, cutoff, outliers, spike_vals, name, prefix)

    else:
        spike_vals = np.array([])
        W = compute_w(x)
        A, B, g, h, W_ignored = estimate_tukey_params(W, name, bound)
        if A == B == g == h == W_ignored == 0:
            return([])
        fitted_TGH = TGH.rgh(len(x), float(A), float(B), float(g), float(h))
        delta = (np.percentile(W, 99) - np.percentile(W, 1))/2
        xlims = [np.percentile(W, 1) - delta, np.percentile(W, 99) + delta]
        range = np.linspace(xlims[0] - delta, xlims[1] + delta, 
                            np.max([100, int(len(W)/300)]))
        smooth_TGH = smooth(fitted_TGH, bw_method =  bw_coef)(range)
        cutoff = np.percentile(fitted_TGH, 99.73)
        x_outliers = np.unique(x[W > cutoff])
        r_sq = plot_test(W, smooth_TGH, range, exp_status, bw_coef, prefix, 
                         cutoff, x_outliers, name, ignored_values = W_ignored)
        if r_sq < 0.8:
            num_main_values, gmean_decrease = get_geometric_info(x)
            if num_main_values > 10:
                return(backup_test(x_spiked, name, prefix))
            else:
                return(backup_test(x_spiked, name, prefix, gmean_decrease))
        plot_data(x, cutoff, x_outliers, spike_vals, name, prefix)
        x_spiked[np.isin(x_spiked, x_outliers)] = np.nan
    return(x_spiked)



    

   