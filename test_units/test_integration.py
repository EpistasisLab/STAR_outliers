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
main_folder = "outlier_removal"
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
from remove_outliers_library import remove_all_outliers

class test_main_library(unittest.TestCase):
    np.random.seed(0)

    def test_outlier_removal(self):
        raw_data = pd.read_csv("all_2018_processed.txt",
                               delimiter = "\t", header = 0)
        clean_data = remove_all_outliers("all_2018_processed.txt")[0]
        count_correctness = []
        for colname in np.array(raw_data.columns):
            raw_col = np.round(raw_data[colname], 6)
            clean_col = np.round(clean_data[colname], 6)
            lb, ub = np.min(clean_col), np.max(clean_col)
            is_inlier = np.logical_and(raw_col <= ub, raw_col >= lb)
            clean_count = np.sum(np.isnan(clean_col) == False)
            raw_count = np.sum(np.isnan(raw_col[is_inlier]) == False)
            count_correctness.append(clean_count == raw_count)
        error_message = "Not all outliers have been removed and/or "
        error_message += "non-outlier points have been removed"
        self.assertTrue(np.all(count_correctness), error_message)
        
if __name__ == '__main__':
    unittest.main()

