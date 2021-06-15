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
8. Run ```python -m STAR_outliers --input [path_to_input_file] --pcutoff [fitted distribution probability bound for outliers] --bound [high percentile for parameter estimation]``` to remove outliers from every column.
9. For example, with the ```all_2018_processed.txt``` file, the full input command looks like this: ```python -m STAR_outliers --input all_2018_processed.txt --pcutoff 0.993 --bound 95```. A ```bound``` value of 95 suggests that there will be very few outliers influencing the value of the 95th percentile. We suggest making this higher for larger or cleaner datasets, though we do not reccomend going above 99. A ```pcutoff``` value of 0.993 means that all real datapoints outside of the fitted distributions's main 99.3% probability mass will be declared as outlier, which is analogous to the standard IQR test for outliers. 
10. If one column is a sample index, then specify that with ```path -m STAR_outliers --input [path_to_input_file] --index [index column name] ...```
