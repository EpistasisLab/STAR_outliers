import unittest
import mock
import numpy as np
import os
import sys
import pdb
import platform
from copy import deepcopy as COPY
from scipy import stats
from rpy2.robjects.packages import importr
from rpy2.robjects import r

#path finding code start
next_dir = os.getcwd()
main_folder = "outlier_removal"
next_folder = ""
count = 0
paths = [next_dir]
folders = [next_folder]
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

#imports specific libraries from the correct path
import remove_outliers_library as main_lib
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

class test_main_library(unittest.TestCase):
    np.random.seed(0)

       
    f_name1 = 'remove_outliers_library.approximate_quantiles'
    f_name2 = 'remove_outliers_library.plot_test'
    f_name3 = 'remove_outliers_library.remove_worst_continuity_violations'
    def f_alt1(W, percentiles): return(np.percentile(W, percentiles))
    @mock.patch(f_name1, side_effect = f_alt1)
    @mock.patch(f_name2, return_value = 0.99)
    @mock.patch(f_name3, side_effect = lambda W: (W, []))
    def test_estimate_tukey_params(self, mock1, mock2, mock3):
        set_seed = r('set.seed')
        set_seed(0)
        W = TGH.rgh(1000000, float(-1.5), float(0.4), float(0.5), float(0.1))
        A, B, g, h, W_ignored = main_lib.estimate_tukey_params(np.array(W), 99.9)
        good_estimates = [-1.499795576559748, 0.3968692002989624,
                          0.5033067615247887, 0.10111509207361158]
        is_correct = np.all(np.isclose([A, B, g, h], good_estimates))
        self.assertTrue(is_correct, "test_estimate_tukey_params may have a math error")
    
    f_name1 = 'remove_outliers_library.approximate_quantiles'
    f_name2 = 'remove_outliers_library.adjust_median_values'
    def f_alt1(W, percentiles): return(np.percentile(W, percentiles))
    @mock.patch(f_name1, side_effect = f_alt1)
    @mock.patch(f_name2, side_effect = lambda x, Q_vec: x)
    def test_compute_w(self, mock1, mock2):
        np.random.seed(0)
        x = np.random.uniform(0, 1, 1000000)
        W = main_lib.compute_w(x)
        moments = [np.mean(W), np.var(W), stats.skew(W), stats.kurtosis(W)]
        good_estimates = [-0.00204834457408487, 0.9923094167368567,
                          -0.01794709589991156, -0.04215819324267711]
        is_correct = np.all(np.isclose(moments, good_estimates))
        self.assertTrue(is_correct, "compute_w may have a math error")

    f_name1 = 'remove_outliers_library.compute_w'
    f_name2 = 'remove_outliers_library.estimate_tukey_params'
    f_name3 = 'remove_outliers_library.plot_test'
    f_name4 = 'remove_outliers_library.plot_data'
    fake_W = np.sort(np.random.normal(0, 1, 200000))
    fake_params = [0, 1, 0, 0, []]
    @mock.patch(f_name1, return_value = fake_W)
    @mock.patch(f_name2, return_value = fake_params)
    @mock.patch(f_name3, return_value = 0.99)
    @mock.patch(f_name4, side_effect = lambda *args: print())
    def test_attempt_tukey_fit(self, mock1, mock2, mock3, mock4):
        np.random.seed(0)
        set_seed = r('set.seed')
        set_seed(0)
        # leverages the fact that W is a monotonic transformation of
        # the ASO statistic, and is also a monotonic transformation of
        # np.abs(x - np.mean(x)) in symetric distributions. 
        x = np.random.uniform(0, 1, 200000)
        sorted_indices = np.argsort(np.abs(x - 0.5))
        x = x[sorted_indices]
        x_spiked = COPY(x)
        x_spiked = main_lib.attempt_tukey_fit(x, x_spiked, "fake", False, 
                                              "fake", 95, 0.3, [], False, [])
        bounds = [np.nanmin(x_spiked), np.nanmax(x_spiked)]
        good_bounds = [0.0013688675359021518, 0.9986369268602082]
        is_correct1 = np.all(np.isclose(bounds, good_bounds))
        is_correct2 = np.all(len(x_spiked) == len(x))
        message1 = "attempt_tukey_fit did not produce correct outlier bounds"
        message2 = "error: attempt_tukey_fit changed the length of x_spiked"
        self.assertTrue(is_correct1, message1)
        self.assertTrue(is_correct2, message2)

    f_name1 = 'remove_outliers_library.plot_test'
    f_name2 = 'remove_outliers_library.plot_data'
    @mock.patch(f_name1, return_value = 0.99)
    @mock.patch(f_name2, side_effect = lambda *args: print())
    def test_attempt_exponential_fit(self, mock1, mock2):
        np.random.seed(0)
        x = np.random.exponential(1, 200000)
        x2 = np.max(x) - x
        x_spiked = COPY(x)
        x_spiked2 = COPY(x2)
        x_spiked = main_lib.attempt_exponential_fit(x, x_spiked, "fake",
                                                    False, "fake", 0.3,
                                                    [], True, [])
        x_spiked2 = main_lib.attempt_exponential_fit(x2, x_spiked2, "fake",
                                                     False, "fake", 0.3,
                                                     [], True, [])
        bound = np.nanmax(x_spiked)
        bound2 = np.nanmin(x_spiked2)
        good_bound = 5.944200450024489
        good_bound2 = 7.883414460787689
        is_correct = np.isclose(bound, good_bound)
        is_correct2 = np.isclose(bound2, good_bound2)
        message = "attempt_exponential_fit did not produce correct left facing bound"
        message2 = "attempt_exponential_fit did not produce correct right facing bound"
        self.assertTrue(is_correct, message)
        self.assertTrue(is_correct2, message2)
        
if __name__ == '__main__':
    unittest.main()
