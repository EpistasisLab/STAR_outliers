import pandas as pd
import numpy as np 
import os
import pdb
from STAR_outliers_library import remove_all_outliers
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input ', type = str, action = "store", dest = "input")
    parser.add_argument('--bound ', type = float, action = "store", dest = "bound")
    parser.add_argument('--index ', type = str, action = "store", dest = "index")
    parser.add_argument('--pcutoff ', type = float, action = "store", dest = "pcutoff")

    pcutoff = parser.parse_args().pcutoff
    if pcutoff is None:
        pcutoff == 0.993
    input_file_name = parser.parse_args().input
    index_name = parser.parse_args().index
    bound = parser.parse_args().bound
    split_name = input_file_name.split(".")
    message = 'error: input file name must have exactly 1 "." character, '
    message += 'which must preceed the filename extension'
    if len(split_name) != 2:
        print(message)
        exit()
    file_name_prefix = split_name[0]

    output = remove_all_outliers(input_file_name, index_name, bound, pcutoff)
    cleaned_data = output[0]
    r_sq_vals = output[1]
    names = output[2]
    fields_with_poor_fits = output[3]
    poor_r_sq_values = output[4]
    severe_outlier_sets = output[5]
    cleaned_field_cols = output[6]
    outlier_info_sets = output[7]

    outlier_info = pd.DataFrame(outlier_info_sets)
    outlier_info.columns = ["name", "percent_inliers",
                            "min_val", "lower_bound",
                            "median", "upper_bound",
                            "max_val"]
    outlier_info.to_csv(file_name_prefix + "_outlier_info.txt",
                        sep = "\t", header = True, index = False)

    all_fits = pd.DataFrame(np.transpose([names, r_sq_vals]))
    bad_fits = pd.DataFrame(np.transpose([fields_with_poor_fits, poor_r_sq_values]))
    all_fits.to_csv(file_name_prefix + "_all_fits.txt",
                    sep = "\t", header = False, index = False)
    bad_fits.to_csv(file_name_prefix  + "_possible_bad_fits.txt",
                    sep = "\t", header = False, index = False)

    outlier_file = open(file_name_prefix  + "_severe_outliers.txt", "w")
    for name, set in zip(names, severe_outlier_sets):
        outlier_file.write(name + ": " + str(set) + "\n")
    outlier_file.close()

    out = file_name_prefix + "_cleaned_data.txt"
    cleaned_data.to_csv(out, sep = "\t", header = True, index = False)

if __name__ == "__main__":
    main()
