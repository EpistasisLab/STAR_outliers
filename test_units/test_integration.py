import numpy as np
import pandas as pd
import pdb
import unittest
import mock
import numpy as np
import os
import sys
import pdb
import platform
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
from STAR_outliers_library import remove_all_outliers

class test_main_library(unittest.TestCase):

    def test_outlier_removal(self):
        np.random.seed(0)
        raw_data = pd.read_csv("all_2018_processed.txt",
                               delimiter = "\t", header = 0).to_numpy()
        new_data = remove_all_outliers("all_2018_processed.txt")[0].to_numpy()
        counts = np.array([len(np.unique(col[np.isnan(col) == False]))
                           for col in raw_data.T])
        raw_data = raw_data[:, counts >= 10]
        new_data = new_data[:, counts >= 10]
        lbs = np.nanmin(new_data, axis = 0)
        ubs = np.nanmax(new_data, axis = 0)
        is_inlier = np.logical_and(raw_data <= ubs, raw_data >= lbs)
        raw_counts = np.sum(is_inlier, axis = 0)
        new_counts = np.sum(np.isnan(new_data) == False, axis = 0)
        error_message = "Not all outliers have been removed and/or "
        error_message += "non-outlier points have been removed"
        self.assertTrue(np.all(raw_counts == new_counts), error_message)

        old_counts = np.sum(np.isnan(raw_data) == False, axis = 0)
        percents = new_counts/old_counts
        bootstraps = np.random.choice(percents, (len(percents), 1000000))
        bs_means = np.mean(bootstraps, axis = 0)
        CI = np.percentile(bs_means, [2.5, 97.5])
        good_CI = np.array([0.99406645, 0.9952961])
        error_message = "The confidence interval for percent of data removed"
        error_message += " is not correct. There may be a mathematical error."
        self.assertTrue(np.all(np.isclose(CI, good_CI)), error_message)
        
if __name__ == '__main__':
    unittest.main()

