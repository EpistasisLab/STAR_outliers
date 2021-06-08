## About STAR_outliers

STAR_outliers (Skew and Tail-heaviness Adjusted Removal of outliers) is an open source python package that determines which points are outliers relative to their distributions shapes. An exponential tail fit is used to determine outlier status if the distribution behaves in a sufficiently exponential-like manner. Otherwise, the data is transformed and fitted to a four parameteter tukey distribution as described in the paper titled 'Outlier identification for skewed and/or heavy-tailed unimodal multivariate distributions'.

## Instructions to Installing STAR_outliers

1. [Install conda](https://docs.conda.io/en/latest/miniconda.html) if you haven't already installed either Anaconda or Miniconda
2. Open your conda terminal. Type "Anaconda" or "Miniconda" into your search bar and open the terminal.
3. Click the "Anaconda Prompt" app (left) to open the black terminal (right). The terminal's top must say "Anaconda prompt"
4. Enter ```conda create --name outliers python=3.7``` in the terminal to create a new environment called outliers with python version 3.7
5. Enter ```conda activate outliers``` in the terminal to enter your new environment. If that doesn't work, enter ```source activate outliers```
6. Once in your outliers environment (repeat step 5 if you close and reopen the conda terminal), enter ```conda install -c conda-forge matplotlib```
7. after installint matplotlib, enter ```pip install STAR-outliers```
8. Run python ```-m STAR_outliers --input [path_to_input_file]``` to remove outliers from every column.
9. If one column is a sample index, then specify that with ```-m STAR_outliers --input [path_to_input_file] --index [index column name]```
