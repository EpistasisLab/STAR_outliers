import pandas as pd
import numpy as np 
import os
import pdb
from tqdm import tqdm

from remove_outliers import compute_outliers # (x_spiked, name)

if not os.path.isdir("Alena_Orlenko_cleaned_cols"):
    os.mkdir("Alena_Orlenko_cleaned_cols")

filenames = np.array(os.listdir("Alena_Orlenko_cols"))
field_names = [name[:-4] for name in filenames]
path_names = ["Alena_Orlenko_cols/" + name for name in filenames]
field_cols = [pd.read_csv(path, delimiter = "\t", header = None) for path in path_names]

field_cols = [col[0].to_numpy() for col in field_cols]

val_counts_not_included = []
for i in tqdm(range(len(field_cols))):

    field = field_cols[i]
    name = field_names[i]
    num_vals = len(np.unique(field[np.isnan(field) == False]))
    if num_vals >= 10 and name not in []:

        cleaned_field = compute_outliers(field, name)
        #signifies technical problem with the data
        if len(cleaned_field) == 0:
            path = "Alena_Orlenko_cleaned_cols/" + filenames[i]
            DF = pd.DataFrame(field)
            DF.to_csv(path, sep = "\t", header = False, index = False)
        else:
            path = "Alena_Orlenko_cleaned_cols/" + filenames[i]
            DF = pd.DataFrame(cleaned_field)
            DF.to_csv(path, sep = "\t", header = False, index = False)
    else:
        path = "Alena_Orlenko_cleaned_cols/" + filenames[i]
        DF = pd.DataFrame(field)
        DF.to_csv(path, sep = "\t", header = False, index = False)

cleaned_filenames = np.array(os.listdir("Alena_Orlenko_cleaned_cols"))
cleaned_fieldnames = [name[:-4] for name in cleaned_filenames]
cleaned_path_names = ["Alena_Orlenko_cleaned_cols/" + name for name in cleaned_filenames]
cleaned_field_cols = [pd.read_csv(path, delimiter = "\t", header = None) for path in cleaned_path_names]
cleaned_data = pd.concat(cleaned_field_cols, axis = 1)
cleaned_data.columns = cleaned_fieldnames
cleaned_data.to_csv("Alena_Orlenko_cleaned_data.txt", sep = "\t", header = True, index = False)