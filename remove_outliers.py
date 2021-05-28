import pandas as pd
import numpy as np 
import os
import pdb
from tqdm import tqdm
from remove_outliers_library import compute_outliers
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--input ', type = str, action = "store", dest = "input")
parser.add_argument('--bound ', type = float, action = "store", dest = "bound")
parser.add_argument('--index ', type = str, action = "store", dest = "index")

input_file_name = parser.parse_args().input
index_name = parser.parse_args().index
bound = parser.parse_args().bound
split_name = input_file_name.split(".")
message = 'error: input file name must have exactly 1 "." character, '
message += 'which must preceed the filename extension'
if len(split_name) != 2:
    print(message)
    exit()
name_prefix = split_name[0]

if not os.path.isdir(name_prefix + "_cols"):
    os.mkdir(name_prefix + "_cols")
present_filenames = np.array(os.listdir(name_prefix + "_cols"))
present_field_names = [name[:-4] for name in present_filenames]
field_names = pd.read_csv(input_file_name, delimiter = "\t", 
                          header = 0, nrows = 0).columns
if not index_name is None:
    field_names = field_names[field_names != index_name]
if len(np.setdiff1d(field_names, present_field_names)) > 0:
    fields = pd.read_csv(input_file_name, delimiter = "\t", header = 0)
    for i, name in enumerate(field_names):
        file_name = name_prefix + "_cols/" + name + ".txt"
        fields[[name]].to_csv(file_name, sep = "\t", header = False, index = False)

if not os.path.isdir(name_prefix + "_cleaned_cols"):
    os.mkdir(name_prefix + "_cleaned_cols")

filenames = np.array(os.listdir(name_prefix + "_cols"))
field_names = [name[:-4] for name in filenames]
path_names = [name_prefix + "_cols/" + name for name in filenames]
field_cols = [pd.read_csv(path, delimiter = "\t", header = None) 
              for path in path_names]
field_cols = [col[0].to_numpy() for col in field_cols]
unique_val_counts = np.array([len(np.unique(field[np.isnan(field) == False])) for field in field_cols])
field_cols_with_outliers = [field_cols[i] for i in np.where(unique_val_counts >= 10)[0]]
field_names_with_outliers = [field_names[i] for i in np.where(unique_val_counts >= 10)[0]]
field_cols_without_outliers = [field_cols[i] for i in np.where(unique_val_counts < 10)[0]]
field_names_without_outliers = [field_names[i] for i in np.where(unique_val_counts < 10)[0]]

bound_not_present = False
if bound is None:
    bound = np.max([np.min([90 + 2.5*(np.log10(len(field_cols[0])) - 2), 99]), 90])
print(bound)

for i in tqdm(range(len(field_cols_with_outliers))):

    field = field_cols_with_outliers[i]
    name = field_names_with_outliers[i]
    cleaned_field = compute_outliers(field, name, name_prefix, bound)
    path = name_prefix + "_cleaned_cols/" + name + ".txt"
    DF = pd.DataFrame(cleaned_field)
    DF.to_csv(path, sep = "\t", header = False, index = False)

for i in range(len(field_cols_without_outliers)):

    field = field_cols_without_outliers[i]
    name = field_names_without_outliers[i]
    path = name_prefix + "_cleaned_cols/" + name + ".txt"
    DF = pd.DataFrame(field)
    DF.to_csv(path, sep = "\t", header = False, index = False)

cleaned_filenames = np.array(os.listdir(name_prefix + "_cleaned_cols"))
cleaned_fieldnames = [name[:-4] for name in cleaned_filenames]
cleaned_path_names = [name_prefix + "_cleaned_cols/" + name 
                      for name in cleaned_filenames]
cleaned_field_cols = [pd.read_csv(path, delimiter = "\t", header = None) 
                      for path in cleaned_path_names]
cleaned_data = pd.concat(cleaned_field_cols, axis = 1)
cleaned_data.columns = cleaned_fieldnames
out = name_prefix + "_cleaned_data.txt"
cleaned_data.to_csv(out, sep = "\t", header = True, index = False)