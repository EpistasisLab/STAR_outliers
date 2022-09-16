import numpy as np
import pandas as pd
import os
import pdb
from copy import deepcopy as COPY
from matplotlib import pyplot as plt
from scipy.optimize import minimize
from scipy.stats import skew
from scipy.stats import norm
from scipy.stats import pearsonr
from scipy.stats import gaussian_kde as smooth
from scipy.stats import expon
from scipy.stats import norm
from scipy.stats import yeojohnson
from scipy.signal import savgol_filter 
from tqdm import tqdm

from STAR_outliers_plotting_library import plot_x
from STAR_outliers_plotting_library import plot_W
from STAR_outliers_polishing_library import approximate_quantiles
from STAR_outliers_polishing_library import remove_spikes
from STAR_outliers_polishing_library import adjust_median_values
from STAR_outliers_testing_library import test_monotonicity
from STAR_outliers_testing_library import test_multimodality
from STAR_outliers_testing_library import get_smooth_peak
from STAR_outliers_testing_library import bin_data

# source title: Outlier identification for skewed and/or 
#               heavy-tailed unimodal multivariate distributions

def estimate_tukey_params(W, A2 = None, no_penalty = False, q_bounds = [1, 99],
                          num_percentiles = 100, custom_weights = None, A_penalty_coef = 0.01):

    """
    Purpose
    -------
    To estimate tukey parameters for the distribution of deviation scores
    
    
    Parameters
    ----------
    W: the distribution of deviation scores
    A2: Value of A tukey parameter for the other
        tukey if EM fits a tukey 2-mixture. 

    Returns
    -------
    A: tukey location parameter
    B: tukey scale parameter
    g: tukey skew parameter
    h: tukey tail heaviness parameter
    """

    Q_vec = np.percentile(W, [10, 25, 50, 75, 90])
    if len(np.unique(Q_vec)) != 5:
        Q_vec = approximate_quantiles(W, [10, 25, 50, 75, 90])

    # rough A estimation
    A = Q_vec[2]

    # rough B estimation
    IQR = Q_vec[3] - Q_vec[1]
    QR2 = Q_vec[4] - Q_vec[0]
    if IQR == 0 or QR2 == 0:
        if len(W[W <  Q_vec[0]]) != 0 and len(W[W > Q_vec[4]]) != 0:
            IQR = np.min(W[W > Q_vec[3]]) - np.max(W[W <  Q_vec[1]])
            QR2 = np.min(W[W > Q_vec[4]]) - np.max(W[W <  Q_vec[0]])
        else:
            IQR = 1.35*np.std(W)
            QR2 = 2.56*np.std(W)
    SK = (Q_vec[4] + Q_vec[0] - 2*Q_vec[2])/QR2
    T = QR2/IQR
    phi = 0.6817766 + 0.0534282*SK + 0.1794771*T - 0.0059595*(T**2)
    B = (0.7413*IQR)/phi

    # rough g estimation
    zv = norm.ppf(0.9, 0, 1)
    UHS = Q_vec[4] - Q_vec[2]
    LHS = Q_vec[2] - Q_vec[0]
    if UHS == 0 or LHS == 0:
        if len(W[W <  Q_vec[0]]) != 0 and len(W[W > Q_vec[4]]) != 0:
            UHS = np.min(W[W >= Q_vec[4]]) - np.max(W[W <  Q_vec[2]])
            LHS = np.min(W[W > Q_vec[2]]) - np.max(W[W <=  Q_vec[0]])
        else:
            UHS = np.max(W) - np.mean(W)
            LHS = np.mean(W) - np.min(W)
            zv = norm.ppf(1 - ((0.1)**np.log10(len(W))), 0, 1)
    g = (1/zv)*np.log(UHS/LHS)

    # rough h estimation
    y = (W - A)/B        
    Q_vec2 = np.percentile(y, [10, 25, 50, 75, 90])
    if len(np.unique(Q_vec2)) != 5:
        Q_vec2 = approximate_quantiles(y, [10, 25, 50, 75, 90])
        
    Q_ratio = (Q_vec2[4]*Q_vec2[0])/(Q_vec2[4] + Q_vec2[0])
    if -g*Q_ratio <= 0:
        h = 0
    else:
        h = (2/(zv**2))*np.log(-g*Q_ratio)
    if np.isnan(h) or np.abs(h) == np.inf:
        h = 0

    q_start, q_end = q_bounds
    qi = np.linspace(q_start, q_end, num_percentiles)
    pi_real = np.percentile(W, qi)
    zi = norm(0,1).ppf(qi/100)
    g += 1E-6
    h += 1E-6
    with np.errstate(all = 'ignore'):
        old_err = (pi_real - (A + (B/g)*(np.exp(g*zi) - 1)*np.exp(h*(zi**2)/2)))**2
    def tukey_loss(theta, A_alt, no_penalty):
        A, B, g, h = theta
        with np.errstate(all = 'ignore'):
            pi_est = A + (B/g)*(np.exp(g*zi) - 1)*np.exp(h*(zi**2)/2)
        weights = qi/100
        if custom_weights != None:
            weights = custom_weights
        penalty = 0.05*((g/10)**2 + h**2)
        if A_alt != None:
            penalty += A_penalty_coef*(1/(A - A_alt + 1E-6))**2
        if no_penalty:
            penalty = 0
        with np.errstate(all = 'ignore'):
            output = np.mean(weights*((pi_est - pi_real)**2)) + penalty
        return(output)
    theta0 = [A, B, g, h]
    theta_est_data = minimize(tukey_loss, theta0, (A2, no_penalty))
    A, B, g, h = theta_est_data.x
    g += 1E-6
    with np.errstate(all = 'ignore'):
        new_err = (pi_real - (A + (B/g)*(np.exp(g*zi) - 1)*np.exp(h*(zi**2)/2)))**2
    return(A, B, g, h, theta_est_data.success)

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

class Tukey:
    
    def __init__(self, A, B, g, h, pi, N, cutoff):

        self.z = np.sort(np.random.normal(0, 1, int(pi*N)))
        self.T = (A + B*(1/(g))*(np.exp((g)*self.z)-1)*np.exp(h*(self.z**2)/2))
        self.lb = np.min(self.T)
        self.ub = np.max(self.T)
        self.cutoff = cutoff

        left_tail = self.T[self.T <= np.percentile(self.T, cutoff)]
        self.left_tail_end = np.max(left_tail)
        left_tail = self.left_tail_end - left_tail
        right_tail = self.T[self.T >= np.percentile(self.T, 100 - cutoff)]
        self.right_tail_end = np.min(right_tail)
        
        self.loc_left = np.min(left_tail)
        self.scale_left = (np.median(left_tail) - self.loc_left)/np.log(2)
        self.loc_right = np.min(right_tail)
        self.scale_right = (np.median(right_tail) - self.loc_right)/np.log(2)

    def left_tail(self, x):
        new_x = self.left_tail_end - x
        return(0.01*self.cutoff*expon.pdf(new_x, self.loc_left, self.scale_left))

    def right_tail(self, x):
        return(0.01*self.cutoff*expon.pdf(x, self.loc_right, self.scale_right))

    def pdf(self, t, stable):

        pt = np.zeros(len(t))
        pt0 = np.zeros(len(t))
        is_main = np.logical_and(t >= self.lb, t <= self.ub)
        lesser, greater = t < self.left_tail_end, t > self.right_tail_end

        if stable == True:        
            domain, pT = bin_data(self.T, 200)
            pT = pT/np.sum(pT*(domain[1] - domain[0]))
            pT = savgol_filter(pT, 51, 6)
            pT[pT < 0] = 0
            pt_ind = np.searchsorted(domain, t[is_main])
            pt[is_main] = (pT[pt_ind])

        else:
            spread = self.ub - self.lb
            domain0 = np.linspace(self.lb - 0.1*spread, self.ub + 0.1*spread, 200)
            pT0 = smooth(self.T, bw_method =  0.2)(domain0)
            pt_ind0 = np.searchsorted(domain0, t[is_main])
            pt[is_main] = (pT0[pt_ind0] + pT0[pt_ind0 - 1])/2

        '''
        pdb.set_trace()
        left_domain = np.linspace(self.left_tail_end, self.lb, 50)
        left_range = self.left_tail(left_domain)
        right_domain = np.linspace(self.right_tail_end, self.ub, 50)
        right_range = self.right_tail(right_domain)
        plt.plot(left_domain, left_range)
        plt.plot(right_domain, right_range)
        plt.hist(self.T, bins = 400, density = True, fc=(0, 0, 1, 0.5))
        plt.plot(domain0, pT0, '-')
        plt.plot(domain, pT, '-')
        plt.show()
        '''
        
        pt[lesser] = self.left_tail(t[lesser])
        pt[greater] = self.right_tail(t[greater])
        return(pt)
    
def estimate_tukey_mixture(W, stable = True, no_penalty = False, q_bounds = [1, 99],
                           max_length = 20000, num_percentiles = 100,
                           custom_weights = None, unit_testing = False):

    parametersets, LLs = [], []
    for A10, A20 in [[5, 95], [33, 66]]:
        for pi10, pi20 in [[0.34, 0.66], [0.66, 0.34]]:
            for A_penalty_coef in [0, 0.05]:

                if unit_testing == True and A_penalty_coef == 0.05:
                    parametersets.append("a single tukey fits well")
                    stop = True
                    LLs.append(-np.inf)
                    continue
                
                # parameter initialization
                A1, A2 = np.percentile(W, A10), np.percentile(W, A20)
                B1, B2 = np.std(W)/2, np.std(W)/2
                g1, g2, h1, h2 = 1E-6, 1E-6, 1E-6, 1E-6
                pi1, pi2, N, cutoff = 0.5, 0.5, np.min([len(W), max_length]), 0.5

                # E step part 1
                T1 = Tukey(A1, B1, g1, h1, pi1, N, cutoff)
                T2 = Tukey(A2, B2, g2, h2, pi2, N, cutoff)

                # likelihood initialization
                L1, L2 = pi1*T1.pdf(W, stable) , pi2*T2.pdf(W, stable)
                LL_seq = [np.sum(np.log(L1 + L2))]
                LL_argmax = 0
                param_seq = [[(A1, B1, g1, h1, pi1), (A2, B2, g2, h2, pi2)]]
                counter = 0
                stop = False
                j = 0
                while stop == False and counter < 1000:

                    j += 1
                    # E step part2
                    L = np.array([T1.pdf(W, stable) +1E-16, T2.pdf(W, stable) + 1E-16])
                    pi_vec = np.array([pi1, pi2]).reshape(-1,1)
                    P = [L[z]*pi_vec[z]/np.sum(L*pi_vec, axis = 0) for z in range(2)]
                    CDF = np.cumsum(P, axis = 0)
                    rand = np.random.rand(len(P[0])).reshape(-1,1)
                    Z = np.argmax(rand <= CDF.T, axis = 1)

                    # M step
                    if len(np.unique(np.round(W[Z == 0], 6))) < 10 or len(np.unique(np.round(W[Z == 1], 6))) < 10:
                        parametersets.append("a single tukey fits well")
                        stop = True
                        LLs.append(-np.inf)
                        continue

                    A1, B1, g1, h1, status1 = estimate_tukey_params(W[Z == 0], A2, no_penalty, q_bounds,
                                                                    num_percentiles, custom_weights, A_penalty_coef)
                    A2, B2, g2, h2, status2 = estimate_tukey_params(W[Z == 1], A1, no_penalty, q_bounds,
                                                                    num_percentiles, custom_weights, A_penalty_coef)
                    if (status1 == False or status2 == False) and stable == True:
                        parametersets.append("numerical_instability_occurred")
                        stop = True
                        LLs.append(-np.inf)
                        continue
            
                    pi1, pi2 = np.sum(Z == 0)/len(Z), np.sum(Z == 1)/len(Z)
                    if np.min([pi1, pi2]) < np.max([0.01, 200/len(W)]):
                        parametersets.append("a single tukey fits well")
                        stop = True
                        LLs.append(-np.inf)
                        continue
                    
                    param_seq.append([(A1, B1, g1, h1, pi1), (A2, B2, g2, h2, pi2)])

                    # E step part1
                    T1 = Tukey(A1, B1, g1, h1, pi1, N, cutoff)
                    T2 = Tukey(A2, B2, g2, h2, pi2, N, cutoff)

                    # likelihood update
                    L1, L2 = pi1*T1.pdf(W, stable) , pi2*T2.pdf(W, stable)
                    if np.all(L1 + L2 > 0):
                        LL_seq.append(np.sum(np.log(L1 + L2)))
                    else:
                        LL_seq.append(-np.inf)
        
                    if LL_seq[-1] > LL_seq[LL_argmax]:
                        counter = 0
                        LL_argmax = len(LL_seq) - 1
                    else:
                        counter += 1

                    if unit_testing == True:
                        max_counter = 50
                    else:
                        max_counter = 20
                    if counter == max_counter:
                        parametersets.append(param_seq[LL_argmax])
                        LLs.append(LL_seq[LL_argmax])
                        stop = True

    alt_outcomes = ["use single tukey", "numerical_instability_occurred"]
    if np.all([i in alt_outcomes for i in parametersets]):
        return(estimate_tukey_mixture(W, False, no_penalty,
                                      q_bounds, max_length, num_percentiles,
                                      custom_weights, unit_testing))
    if parametersets == ["a single tukey fits well"]*8:
        return("use single tukey")
    else:
        return(parametersets[np.argmax(LLs)])

def get_body(x):

    p1, p2, p3, p4 = np.percentile(x, [2.5, 5, 95, 97.5])
    
    if skew(x) > 0:
        x_body = x[x < p4]
        if skew(x_body) > 0:
            x_body = x_body[x_body < p3]
        else:
            x_body = x_body[x_body > p1]
        return(x_body)

    else:
        x_body = x[x > p1]
        if skew(x_body) <= 0:
            x_body = x_body[x_body > p2]
        else:
            x_body = x_body[x_body < p4]
        return(x_body)

def get_outlier_fit(x, x_spiked, name, pcutoff, spike_vals, prefix):
    x_body = get_body(x)
    n_bins = np.max([int(len(x_body)/200), 50])
    out = test_monotonicity(x_body, x, 100*pcutoff, n_bins, not_sensitive = True)
    is_monotonic, mirrored_data, status = out
    num_unique = len(np.unique(x_body))
    is_low_count = num_unique < 60
    if is_monotonic or is_low_count:
        is_multimodal = False
    else:
        is_multimodal = test_multimodality(x_body, n_bins)
    dist_type = [is_monotonic, is_multimodal, is_low_count]
    if is_multimodal or is_low_count:
        x_outliers, area_overlap = fit_tukey(x, mirrored_data, None, n_bins, dist_type,
                                             name, pcutoff, spike_vals, prefix)
    elif is_monotonic:
        x_outliers, area_overlap = fit_tukey(x, mirrored_data, status, n_bins, dist_type,
                                             name, pcutoff, spike_vals, prefix)
    else:
        peak = get_smooth_peak(x_body)
        left_half, right_half = x[x <= peak], x[x >= peak]
        left_mirror = 2*np.max(left_half) - left_half
        right_mirror = 2*np.min(right_half) - right_half
        x_left = np.concatenate([left_half, left_mirror])
        x_right = np.concatenate([right_half, right_mirror])
        all_x_outliers, all_area_overlap = [], []
        for x_side, side in [(x_left, "_left"), (x_right, "_right")]:
            x_outliers, area_overlap = fit_tukey(x_side, mirrored_data, side, n_bins,
                                                 dist_type, (name + side), pcutoff,
                                                 spike_vals, prefix, yes_plot_x = False)
            all_x_outliers.append(x_outliers)
            all_area_overlap.append(area_overlap)

        x_outliers = np.union1d(*all_x_outliers)
        plot_x(x, x_outliers, spike_vals, name, prefix, n_bins)
        p_vec = np.array([len(left_half)/len(x), len(right_half)/len(x)])
        area_overlap = np.sum(p_vec*np.array(all_area_overlap))
    x_spiked[np.isin(x_spiked, x_outliers)] = np.nan
    return(x_spiked, area_overlap)

def fit_TGH(A, B, g, h, N):
    z = np.random.normal(0, 1, N)
    main = (np.exp((g + 1E-10)*z)-1)*np.exp(h*(z**2)/2)
    coef = B*(1/(g + 1E-10))
    fitted_TGH  = A + coef*main
    return(fitted_TGH)

def fit_tukey(x, mirrored_data, side, n_bins, dist_type,
              name, pcutoff, spike_vals, prefix, yes_plot_x = True):

    is_monotonic, is_multimodal, is_low_count = dist_type
    is_none = not(is_monotonic or is_multimodal or is_low_count)
    if is_low_count:
        W = compute_w(x)
        A, B, g, h, void = estimate_tukey_params(W)
        fitted_TGH = fit_TGH(A, B, g, h, 100000)
    elif is_monotonic:
        W = compute_w(mirrored_data)
        if test_multimodality(get_body(W), n_bins):
            output = estimate_tukey_mixture(W)
            if output == "use single tukey":
                A, B, g, h, void = estimate_tukey_params(W)
                fitted_TGH = fit_TGH(A, B, g, h, 100000)
            else:
                T1, T2 = output
                TGH1 = fit_TGH(T1[0], T1[1], T1[2], T1[3], int(T1[4]*100000))
                TGH2 = fit_TGH(T2[0], T2[1], T2[2], T2[3], int(T2[4]*100000))
                fitted_TGH = np.concatenate([TGH1, TGH2])
        else:
            W = compute_w(mirrored_data)
            A, B, g, h, void = estimate_tukey_params(W)
            fitted_TGH = fit_TGH(A, B, g, h, 100000)
    elif is_multimodal:
        W = compute_w(x)
        output = estimate_tukey_mixture(W)
        if output == "use single tukey":
            A, B, g, h, void = estimate_tukey_params(W)
            fitted_TGH = fit_TGH(A, B, g, h, 100000)
        else:
            T1, T2 = output
            TGH1 = fit_TGH(T1[0], T1[1], T1[2], T1[3], int(T1[4]*100000))
            TGH2 = fit_TGH(T2[0], T2[1], T2[2], T2[3], int(T2[4]*100000))
            fitted_TGH = np.concatenate([TGH1, TGH2])
    else:
        W = compute_w(x)
        if test_multimodality(get_body(W), n_bins) and len(np.unique(W)) > 30:
            output = estimate_tukey_mixture(W)
            if output == "use single tukey":
                A, B, g, h, void = estimate_tukey_params(W)
                fitted_TGH = fit_TGH(A, B, g, h, 100000)
            else:
                T1, T2 = output
                TGH1 = fit_TGH(T1[0], T1[1], T1[2], T1[3], int(T1[4]*100000))
                TGH2 = fit_TGH(T2[0], T2[1], T2[2], T2[3], int(T2[4]*100000))
                fitted_TGH = np.concatenate([TGH1, TGH2])
        else:
            A, B, g, h, void = estimate_tukey_params(W)
            fitted_TGH = fit_TGH(A, B, g, h, 100000)

    cutoff = np.percentile(fitted_TGH, pcutoff*100)
    if is_monotonic:
        x_outliers = mirrored_data[W > cutoff]
    else:
        x_outliers = x[W > cutoff]
    if (is_monotonic or is_none) and (side == '_left' or side == "increasing"):
        x_outliers = x_outliers[x_outliers < np.median(x)]
        x = x[x <= np.median(x)]
    elif (is_monotonic or is_none) and (side == '_right' or side == "decreasing"):
        x_outliers = x_outliers[x_outliers > np.median(x)]
        x = x[x >= np.median(x)]
    area_overlap = plot_W(W, fitted_TGH, name, prefix, cutoff, n_bins)
    if yes_plot_x: plot_x(x, x_outliers, spike_vals, name, prefix, n_bins)
    return(x_outliers, area_overlap)

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

def compute_outliers(x_spiked, name, prefix, pcutoff):

    x_spiked = x_spiked.astype(float)
    x_spiked_old = COPY(x_spiked)
    x = COPY(x_spiked)[np.isnan(x_spiked)==False]
    x, spike_vals, decreases = remove_spikes(x, x_spiked, name, prefix, 0, [], [])

    outlier_info = [name]
    old_count = np.sum(np.isnan(x_spiked)==False)
    x_spiked_new, area_overlap = get_outlier_fit(x, x_spiked, name, pcutoff, spike_vals, prefix)
    outlier_info.append(np.sum(np.isnan(x_spiked_new)==False)/old_count)
    outlier_info.append(np.nanmin(x_spiked_old))
    outlier_info.append(get_constrained_min(x_spiked, spike_vals))
    outlier_info.append(np.nanpercentile(x, 50))
    outlier_info.append(get_constrained_max(x_spiked, spike_vals))
    outlier_info.append(np.nanmax(x_spiked_old))
    return(x_spiked_new, area_overlap, outlier_info)

def remove_all_outliers(input_file_name, index_name, pcutoff):
    
    fields = pd.read_csv(input_file_name, delimiter = "\t", header = 0)
    field_names = fields.columns
    if not index_name is None:
        field_names = field_names[field_names != index_name]
        index_col = fields[index_name]
        fields = fields[field_names]
    field_cols = [fields.loc[:, name].to_numpy() for name in field_names]

    names = []
    cleaned_field_cols = []
    area_overlap_vals = []
    outlier_info_sets = []
    for i in tqdm(range(len(field_names))):
        field = field_cols[i]
        unique_vals = np.unique(field)
        if len(unique_vals[np.isnan(unique_vals) == False]) >= 10:
            name = field_names[i]
            names.append(name)
            prefix = input_file_name.split(".")[0]
            output = compute_outliers(field, name, prefix, pcutoff)
            
            cleaned_field_cols.append(output[0])
            area_overlap_vals.append(output[1])
            outlier_info_sets.append(output[2])
        else:
            cleaned_field_cols.append(field)

    cleaned_data = pd.DataFrame(np.transpose(cleaned_field_cols))
    cleaned_data.columns = field_names
    if not index_name is None:
        cleaned_data[index_name] = index_col
    return(cleaned_data,
           area_overlap_vals, names,
           cleaned_field_cols,
           outlier_info_sets)
   
