import numpy as np
import pandas as pd
import pdb

raw_data = pd.read_csv("all_2018_processed.txt", delimiter = "\t", header = 0)
clean_data = pd.read_csv("all_2018_processed_cleaned_data.txt", delimiter = "\t", header = 0)

count_correctness = []
for colname in raw_data.columns:
    raw_col = np.round(raw_data[colname], 6)
    clean_col = np.round(clean_data[colname], 6)
    lb, ub = np.min(clean_col), np.max(clean_col)
    is_inlier = np.logical_and(raw_col <= ub, raw_col >= lb)
    clean_count = np.sum(np.isnan(clean_col) == False)
    raw_count = np.sum(np.isnan(raw_col[is_inlier]) == False)
    count_correctness.append(clean_count == raw_count)
print(np.all(count_correctness))
