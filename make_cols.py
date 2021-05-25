import numpy as np
import pandas as pd
import os
import pdb

if not os.path.isdir("Alena_Orlenko_cols"):
    os.mkdir("Alena_Orlenko_cols")

fields = pd.read_csv("Alena_Orlenko_fields.txt", delimiter = "\t", header = 0)
fields = fields[fields.columns[1:]]
field_names = fields.columns
for i, name in enumerate(field_names):
    file_name = "Alena_Orlenko_cols/" + name + ".txt"
    fields[[name]].to_csv(file_name, sep = "\t", header = False, index = False)
