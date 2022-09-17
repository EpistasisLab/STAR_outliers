import unittest
import mock
import numpy as np
import os
import sys
import pdb
import platform
from matplotlib import pyplot as plt
from copy import deepcopy as COPY
from scipy import stats

#path finding code start
next_dir, next_folder = os.path.split(os.getcwd())
main_folder = "STAR_outliers"
count = 1
paths = [os.getcwd(), next_dir]
folders = ["", next_folder]
while next_folder != main_folder and count < 4:
    next_dir, next_folder = os.path.split(next_dir)
    paths.append(next_dir)
    folders.append(next_folder)
    count += 1
if count >= 4:
    message = "error: important paths have been renamed or reorganized. "
    message += "If this was intentional, then change the path "
    message += "finding code in test_main_library.py"
    print(message)
os.chdir(paths[count - 1])
sys.path.insert(1, os.getcwd())
#path finding code end
import STAR_outliers_library as main_lib

class test_main_library(unittest.TestCase):
    np.random.seed(0)

    f_name1 = 'STAR_outliers_library.approximate_quantiles'
    def f_alt1(W, percentiles): return(np.percentile(W, percentiles))
    @mock.patch(f_name1, side_effect = f_alt1)
    def test_estimate_tukey_params(self, mock1):
        A, B, g, h = -1.5, 0.4, 1/(np.e), 1/(np.e**3)
        z = np.random.normal(0, 1, 1000000)
        W  = A + B*(1/(g + 1E-10))*(np.exp((g + 1E-10)*z)-1)*np.exp(h*(z**2)/2)
        A2, B2, g2, h2, W_ignored = main_lib.estimate_tukey_params(np.array(W), 99.9, True)
        rounded_estimates = np.round([A2, B2, np.log(g2)/10, np.log(h2)/10], 1)
        correct_vals = np.array([-1.5, 0.4, -0.1, -0.3])
        is_correct = np.all(correct_vals == rounded_estimates)
        self.assertTrue(is_correct, "test_estimate_tukey_params may have a math error")

    f_name1 = 'STAR_outliers_library.approximate_quantiles'
    f_name2 = 'STAR_outliers_library.adjust_median_values'
    def f_alt1(W, percentiles): return(np.percentile(W, percentiles))
    @mock.patch(f_name1, side_effect = f_alt1)
    @mock.patch(f_name2, side_effect = lambda x, Q_vec: x)
    def test_compute_w(self, mock1, mock2):
        x = np.random.uniform(0, 1, 100000000)
        W = main_lib.compute_w(x)
        #should be a standard normal distribution
        moments = np.round([np.mean(W), np.var(W), stats.skew(W), stats.kurtosis(W)], 1)
        is_correct = np.all(np.isclose(moments, [0,1,0,0]))
        self.assertTrue(is_correct, "compute_w may have a math error")

    def test_estimate_tukey_mixture(self):
        A1, B1, g1, h1 = -1.5, 0.2, 1/(np.e), 1/(np.e**3)
        A2, B2, g2, h2 = 1.5, 0.4, 1/(np.e**2), 1/(np.e**4)
        W1 = main_lib.fit_TGH(A1, B1, g1, h1, 200000)
        W2 = main_lib.fit_TGH(A2, B2, g2, h2, 300000)
        W = np.concatenate([W1, W2])
        T1, T2 = main_lib.estimate_tukey_mixture(W, no_penalty = True, q_bounds = [0.01, 99.99],
                                                 max_length = len(W), num_percentiles = 2000,
                                                 custom_weights = 1, unit_testing = True)
        T1, T2 = np.array(T1), np.array(T2)
        T1[2:4], T2[2:4] = np.log(T1[2:4])/10, np.log(T2[2:4])/10
        P1, P2 = np.array([A1, B1, -0.1, -0.3, 0.4]), np.array([A2, B2, -0.2, -0.4, 0.6])
        is_correct = np.all(np.round(T1,1)==P1) and np.all(np.round(T2,1)==P2)
        self.assertTrue(is_correct, "test_estimate_tukey_mixture may have a math error")
        
if __name__ == '__main__':
    unittest.main()
