import numpy as np
import pandas as pd
import pdb
from copy import deepcopy as COPY
from matplotlib import pyplot as plt
from tqdm import tqdm
from scipy.stats import spearmanr
from scipy.stats import ttest_1samp as ttest

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
    bounds = np.linspace(np.min(x), np.max(x) + delta, n_bins + 1).reshape(-1, 1)
    lbs, ubs = bounds[:-1], bounds[1:]
    bins = np.logical_and(x >= lbs, x < ubs)
    real_domain = (ubs.reshape(-1) + lbs.reshape(-1))/2
    real_range = np.sum(bins, axis = 1)

    dx = real_domain[1] - real_domain[0]
    domain2, range2 = np.zeros(n_bins + 4), np.zeros(n_bins + 4)
    range2[2:-2] = real_range
    domain2[2:-2] = real_domain
    domain2[0] = domain2[2] - 2*dx
    domain2[1] = domain2[2] - dx
    domain2[-2] = domain2[-3] + dx
    domain2[-1] = domain2[-3] + 2*dx
    return(domain2, range2)

# pool adjacent violators algorithm
def pava(bins, bin_counts, old_counts, total_counts, coef):
    cdf_range = coef*(np.cumsum(bin_counts) + old_counts)/total_counts
    x1, x2 = bins[0], bins[1]
    y1, y2 = cdf_range[0], cdf_range[1]
    b1 = (y2 - y1)/(x2 - x1)
    violators = []
    keepers = [0, 1]
    for i in range(2, len(cdf_range)):
        x3, y3 = bins[i], cdf_range[i]
        b2 = (y3 - y2)/(x3 - x2)
        if b2 >= b1:
            keepers.append(i)
            x1, x2 = COPY(x2), COPY(x3)
            y1, y2 = COPY(y2), COPY(y3)
            b1 = COPY(b2)
        elif len(keepers) == 2:
            violators.append(keepers.pop())
            keepers.append(i)
            x1, x2 = bins[keepers[-2]], bins[keepers[-1]]
            y1, y2 = cdf_range[keepers[-2]], cdf_range[keepers[-1]]
            b1 = (y2 - y1)/(x2 - x1)
        else:
            violators.append(keepers.pop())
            violations_hidden = True 
            while violations_hidden:
                x1, x2 = bins[keepers[-2]], bins[keepers[-1]]
                y1, y2 = cdf_range[keepers[-2]], cdf_range[keepers[-1]]
                b1 = (y2 - y1)/(x2 - x1)
                b2 = (y3 - y2)/(x3 - x2)
                if b2 >= b1:
                    violations_hidden = False
                    keepers.append(i)
                    x1, x2 = COPY(x2), COPY(x3)
                    y1, y2 = COPY(y2), COPY(y3)
                    b1 = COPY(b2)
                else:
                    violators.append(keepers.pop())
                if keepers == [0]:
                    violations_hidden = False
                    keepers.append(i)
                    x1, x2 = bins[keepers[-2]], bins[keepers[-1]]
                    y1, y2 = cdf_range[keepers[-2]], cdf_range[keepers[-1]]
                    b1 = (y2 - y1)/(x2 - x1)
                
    keepers, violators = np.array(keepers), np.array(violators)
    gcm_bins, gcm_cdf_range = bins[keepers], cdf_range[keepers] 
    slopes = (gcm_cdf_range[1:] - gcm_cdf_range[:-1])/(gcm_bins[1:] - gcm_bins[:-1])
    slopes = np.concatenate([slopes[i]*np.ones(keepers[i + 1] - keepers[i]) for i in range(len(keepers) - 1)])
    gcm_range_full = np.cumsum(slopes*(bins[1:] - bins[:-1])) + cdf_range[0]
    gcm_range_full = np.concatenate([[cdf_range[0]], gcm_range_full])
    #check1 = np.round((gcm_cdf_range*1E6).astype(int)/1E6, 4) == np.round((gcm_range_full[keepers]*1E6).astype(int)/1E6, 4)
    #check2 = np.round((cdf_range*1E6).astype(int)/1E6, 4) >= np.round((gcm_range_full*1E6).astype(int)/1E6, 4)
    check1 = np.abs(gcm_cdf_range - gcm_range_full[keepers]) < 0.0001
    check2 = cdf_range - gcm_range_full > -0.0001
    
    if not (np.all(check1) and np.all(check2)):
        pdb.set_trace()

    if coef < 0:
        gcm_range_full *= -1
        cdf_range *= -1

    return(gcm_range_full, cdf_range)
    
def compute_d(data, nbins, binned = False):
    if binned:
        bins, bin_counts = data
    else:
        bins, bin_counts = bin_data(data, nbins)
    N = np.sum(bin_counts)
    cdf_range = np.cumsum(bin_counts)/N
    Xl, Xu, D, d = 0, len(bin_counts) - 1, 0, 1
    Xl0, Xu0, D0 = 0, len(bin_counts) - 1, 0
    while d > D:
        Xl, Xu, D = Xl0 + Xl, Xu0 + Xl, D0
        bins1 = bins[Xl:(Xu + 1)]
        bin_counts1 = bin_counts[Xl:(Xu + 1)]
        cdf_range1 = cdf_range[Xl:(Xu + 1)]
        old_counts = np.sum(bin_counts[:Xl])
        # TODO_confirm cdf output
        gcm_range, cdf_range1 = pava(bins1, bin_counts1, old_counts, N, 1)
        lcm_range, cdf_range2 = pava(bins1, bin_counts1, old_counts, N, -1)
        gi = np.where(np.round(cdf_range1, 6) == np.round(gcm_range, 6))[0]
        li = np.where(np.round(cdf_range1, 6) == np.round(lcm_range, 6))[0]
        bins_gi, bins_li = bins1[gi], bins1[li]
        d_gi = np.abs(gcm_range[gi] - lcm_range[gi])
        d_gi_argmax = np.argmax(d_gi)
        d_gi_max = d_gi[d_gi_argmax]
        d_li = np.abs(gcm_range[li] - lcm_range[li])
        d_li_argmax = np.argmax(d_li)
        d_li_max = d_li[d_li_argmax]
        if d_gi_max > d_li_max:
            Xl0 = gi[d_gi_argmax]
            Xu0_val = np.round(np.min(bins_li[bins_li >= bins1[Xl0]]), 6)
            Xu0 = np.where(np.round(bins1, 6) == Xu0_val)[0][0]
            d = d_gi_max
        else:
            Xu0 = li[d_li_argmax]
            Xl0_val = np.round(np.max(bins_gi[bins_gi <= bins1[Xu0]]), 6)
            Xl0 = np.where(np.round(bins1, 6) == Xl0_val)[0][0]
            d = d_li_max
        if d > D:
            max1 = D
            max2 = np.max(np.abs((gcm_range - cdf_range1)[:Xl0 + 1]))
            max3 = np.max(np.abs((lcm_range - cdf_range1)[Xu0:]))
            D0 = np.max([max1, max2, max3])
    Xl, Xu = Xl0 + Xl, Xu0 + Xl
    bins1, bin_counts1 = bins[:Xl + 1], bin_counts[:Xl + 1]
    if (bins[Xu] - bins[Xl]) == 0 and (cdf_range[Xu] - cdf_range[Xl]) == 0:
        slope = np.nan
    else:
        slope = (cdf_range[Xu] - cdf_range[Xl])/(bins[Xu] - bins[Xl])
    dx = bins[Xl + 1:Xu + 1] - bins[Xl:Xu]
    cdf_range2 = slope*np.cumsum(dx) + cdf_range[Xl]
    bins2, cdf_range2 = bins[Xl:Xu + 1], np.concatenate([[cdf_range[Xl]], cdf_range2])
    bins3, bin_counts3 = bins[Xu:], bin_counts[Xu:]
    old_counts = np.sum(bin_counts[:Xu])

    lcm_range, void = pava(bins3, bin_counts3, old_counts, N, -1)
    if Xl == 0:
        cdf_range_est = np.concatenate([cdf_range2[:-1], lcm_range])
    else:
        gcm_range, void = pava(bins1, bin_counts1, 0, N, 1)
            
    if len(cdf_range2) == 1 and Xl != 0:
        cdf_range_est = np.concatenate([gcm_range, lcm_range[1:]])
    elif len(cdf_range2) == 2 and Xl != 0:
        cdf_range_est = np.concatenate([gcm_range, lcm_range])
    elif Xl != 0:
        cdf_range_est = np.concatenate([gcm_range, cdf_range2[1:-1], lcm_range])

    '''
    try: 
        plot_ind = np.logical_and(cdf_range <= 0.99, cdf_range >= 0.01)
        plt.plot(bins[plot_ind], cdf_range[plot_ind], 'g-')
        plt.plot(bins[plot_ind], cdf_range_est[plot_ind], 'r-')
        
        plt.show()
        plt.clf()
    except:
        pdb.set_trace()
    print(np.max(np.abs(cdf_range_est - cdf_range)))
    print(D)
    '''

    return(bins, cdf_range_est, D)

def test_multimodality(data, n_bins, not_sensitive = False):
    bins, unimodal_cdf_range, D = compute_d(data, n_bins)
    diffs = []
    if D < 0.001 and not_sensitive or D < 0.0001:
        return(False)
    for i in range(30):
        indices = np.searchsorted(unimodal_cdf_range, np.random.rand(len(data)))
        indices = np.concatenate([indices, np.arange(len(unimodal_cdf_range))])
        void, unimodal_bin_counts = np.unique(indices, return_counts = True)
        unimodal_bin_counts -= np.ones(len(unimodal_bin_counts), dtype = int)
        binned_data = [bins, unimodal_bin_counts]
        void, void, Db = compute_d(binned_data, n_bins, binned = True)
        diffs.append(Db)
    if D > np.max(diffs):
        return(True)
    else:
        return(False)

def test_monotonicity(data_body, data, pcutoff, n_bins, not_sensitive = False, discrete = False):
    if discrete:
        x, y = np.unique(data_body, return_counts = True)
    else:
        x, y = bin_data(data_body, n_bins)
    r, p = spearmanr(x, y)
    if r > 0 and p < 0.001:
        status = "increasing"
        data_flipped = data_body + 2*(np.max(data_body) - data_body)
        data2 = np.concatenate([data_body, data_flipped])
        if discrete:
            is_monotonic =  np.all(y[1:] - y[:-1] > 0)
        else:
            is_monotonic = (test_multimodality(data2, n_bins, not_sensitive) == False)
        data_flipped2 = data + 2*(np.max(data) - data) + 1E-7
        mirrored_data = np.concatenate([data, data_flipped2])
    elif r < 0 and p < 0.001:
        status = "decreasing"
        data_flipped = data_body + 2*(np.min(data_body) - data_body)
        data2 = np.concatenate([data_body, data_flipped])
        if discrete:
            is_monotonic =  np.all(y[1:] - y[:-1] < 0)
        else:
            is_monotonic = (test_multimodality(data2, n_bins, not_sensitive) == False)
        data_flipped2 = data + 2*(np.min(data) - data) - 1E-7
        mirrored_data = np.concatenate([data, data_flipped2])
    else:
        is_monotonic, mirrored_data, status = False, [], "not monotonic"
    return([is_monotonic, mirrored_data, status])

def get_smooth_peak(data):
    num_bins = int(len(data)/(np.max([10, int(len(data)/1000)])))
    region_size = int(0.01*num_bins) + 1
    x, y = bin_data(data, num_bins)
    peak_region = np.argsort(y)[-region_size:]
    argmax_est = np.mean(x[peak_region])
    return(argmax_est)


'''
def get_smooth_peak(all_data):
    p5, p95 = np.percentile(all_data, [5, 95])
    num_bins = int(len(data)/(np.max([10, int(len(data)/1000)])))
    x, y = bin_data(data, num_bins)
    peak_loc = np.argmax(y)
    N = len(y)
    L = int(np.min([peak_loc/5, (N - peak_loc)/5]))
    if L < 5: L = int(np.min([peak_loc/2, (N - peak_loc)/2]))
    ind = np.arange(L, N - L + 1)
    y_smooth = np.zeros(N)
    for i in ind: y_smooth[i] = np.mean(y[i - L: i + L])
    if peak_loc == 478:
        pdb.set_trace()
    argmax_est = data[np.argmin(np.abs(data - x[np.argmax(y_smooth)]))]
    return(argmax_est)


pdb.set_trace()

data = np.concatenate([np.random.laplace(0, 1, 600), np.random.uniform(-10, 10, 400)])
def get_smooth_peak(data):
    num_bins = int(len(data)/(np.max([10, int(len(data)/1000)])))
    x, y = bin_data(data, num_bins)
    peak_loc = np.argmax(y)
    N = len(y)
    L = int(np.min([peak_loc/5, (N - peak_loc)/5]))
    if L < 5: L = int(np.min([peak_loc/2, (N - peak_loc)/2]))
    ind = np.arange(L, N - L + 1)
    y_smooth = np.zeros(N)
    for i in ind: y_smooth[i] = np.mean(y[i - L: i + L])
    argmax_est = data[np.argmin(np.abs(data - x[np.argmax(y_smooth)]))]
    return(argmax_est)

mm_data = pd.read_csv("multimodal_data.txt", delimiter = "\t").to_numpy()
data = mm_data[:, -1]
tree_maker.pyset_trace()
'/
print(test_unimodality(data, 600))

pdb.set_trace()
p_vals = []
old_data = []
for i in range(10):
    M = 150000
    bimodal_data = np.concatenate([np.random.normal(-1, 1, 75000), np.random.normal(1, 1, 75000)])
    old_data.append(bimodal_data)
    bins, unimodal_cdf_range, D = compute_d(bimodal_data, 400)

    diffs = []
    for i in tqdm(range(1000)):
        indices = np.searchsorted(unimodal_cdf_range, np.random.rand(M))
        indices = np.concatenate([indices, np.arange(len(unimodal_cdf_range))])
        void, unimodal_bin_counts = np.unique(indices, return_counts = True)
        unimodal_bin_counts -= np.ones(len(unimodal_bin_counts), dtype = int)
        data = [bins, unimodal_bin_counts]
        void, void, Db = compute_d(data, 400, binned = True)
        diffs.append(Db)
    p = np.sum(np.array(diffs) >= D)/1000
    print(p)
    p_vals.append(p)
plt.hist(np.array(old_data).reshape(-1), bins = 400)
plt.show()
plt.clf()
pdb.set_trace()
print(1)
'''


