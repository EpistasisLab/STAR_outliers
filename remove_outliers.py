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


input_file_name = parser.parse_args().input
bound = parser.parse_args().bound
split_name = input_file_name.split(".")
message = 'error: input file name must have exactly 1 "." character, '
message += 'which must preceed the filename extension'
if len(split_name) != 2:
    print(message)
    exit()
name_prefix = split_name[0]
'''
if not os.path.isdir(name_prefix + "_cols"):
    os.mkdir(name_prefix + "_cols")
fields = pd.read_csv(input_file_name, delimiter = "\t", header = 0)
fields = fields[fields.columns[1:]]
field_names = fields.columns
for i, name in enumerate(field_names):
    file_name = name_prefix + "_cols/" + name + ".txt"
    fields[[name]].to_csv(file_name, sep = "\t", header = False, index = False)
'''

if not os.path.isdir(name_prefix + "_cleaned_cols"):
    os.mkdir(name_prefix + "_cleaned_cols")

filenames = np.array(os.listdir(name_prefix + "_cols"))
field_names = [name[:-4] for name in filenames]
path_names = [name_prefix + "_cols/" + name for name in filenames]
field_cols = [pd.read_csv(path, delimiter = "\t", header = None) 
              for path in path_names]
field_cols = [col[0].to_numpy() for col in field_cols]

if bound is None:
    if len(field_cols[0]) < 1000:
        bound = 90
    elif len(field_cols[0]) < 10000:
        bound = 95
    elif len(field_cols[0]) < 100000:
        bound = 97.5
    else:
        bound = 99
print(bound)


for i in tqdm(range(len(field_cols))):

    field = field_cols[i]
    name = field_names[i]
    num_vals = len(np.unique(field[np.isnan(field) == False]))
    if num_vals >= 10 and name not in []:

        cleaned_field = compute_outliers(field, name, name_prefix, bound)
        #signifies technical problem with the data
        if len(cleaned_field) == 0:
            path = name_prefix + "_cleaned_cols/" + filenames[i]
            DF = pd.DataFrame(field)
            DF.to_csv(path, sep = "\t", header = False, index = False)
        else:
            path = name_prefix + "_cleaned_cols/" + filenames[i]
            DF = pd.DataFrame(cleaned_field)
            DF.to_csv(path, sep = "\t", header = False, index = False)
    else:
        path = name_prefix + "_cleaned_cols/" + filenames[i]
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