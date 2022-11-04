import pandas as pd
import numpy as np 
import os
import pdb
from STAR_outliers_library import remove_all_outliers
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input ', type = str, action = "store", dest = "input")
    parser.add_argument('--index ', type = str, action = "store", dest = "index")
    parser.add_argument('--pcutoff ', type = float, action = "store", dest = "pcutoff")
    parser.add_argument('--seed ', type = float, action = "store", dest = "seed")
    parser.add_argument('--continuity_threshold ', type = int, action = "store", dest = "continuity_threshold")
    
    seed = parser.parse_args().seed
    continuity_threshold = parser.parse_args().continuity_threshold
    if not seed is None:
        np.random.seed(seed = int(seed))
    pcutoff = parser.parse_args().pcutoff
    if pcutoff is None:
        pcutoff = 0.993
    if continuity_threshold is None:
        continuity_threshold  = 60
    input_file_name = parser.parse_args().input
    index_name = parser.parse_args().index
    split_name = input_file_name.split(".")
    message = 'error: input file name must have exactly 1 "." character, '
    message += 'which must preceed the filename extension'
    if len(split_name) != 2:
        print(message)
        exit()
    file_name_prefix = split_name[0]

    output = remove_all_outliers(input_file_name, index_name, pcutoff, continuity_threshold)
    cleaned_data = output[0]
    area_overlap_vals = output[1]
    names = output[2]
    cleaned_field_cols = output[3]
    outlier_info_sets = output[4]

    outlier_info = pd.DataFrame(outlier_info_sets)
    outlier_info.columns = ["name", "percent_inliers",
                            "min_val", "lower_bound",
                            "median", "upper_bound",
                            "max_val"]
    outlier_info.to_csv(file_name_prefix + "_outlier_info.txt",
                        sep = "\t", header = True, index = False)

    all_fits = pd.DataFrame(np.transpose([names, area_overlap_vals]))
    all_fits.to_csv(file_name_prefix + "_all_fits.txt",
                    sep = "\t", header = False, index = False)

    out = file_name_prefix + "_cleaned_data.txt"
    cleaned_data.to_csv(out, sep = "\t", header = True, index = False)

if __name__ == "__main__":
    main()
