## About STAR_outliers :sun_with_face:

STAR_outliers (Skew and Tail-heaviness Adjusted Removal of outliers) is an open source python package that identifies univariate outliers relative to their distributions shapes. Univariate distributions are initially transformed as described in 'Outlier identification for skewed and/or heavy-tailed unimodal multivariate distributions'. Data transformed in this way often fits well to a four parameter tukey distribution, but not if the data is originally monotonic, highly skewed, multimodal, or discrete. STAR_outliers implements several modifications that robustly handle all of these cases.

STAR_outliers handles discreteness by replacing the original tukey parameter estimates (based on only 4 fixed quantiles) with quantile regression that considers quantiles 1 through 99 in intervals of 1, which robustly accounts for single scalars that occupy entire quantile ranges. We handle monotonic data by mirroring the distribution accross the broad side prior to transforming it, and we handle skewed data by splitting it at the peak and treating each side as though it were nmonotonic. Finally, we handle multimodal data by fitting a bimodal tukey mixture model with a custom numerical EM algorithm. 

## Instructions to Installing STAR_outliers :hammer_and_wrench:

1. [Install conda](https://docs.conda.io/en/latest/miniconda.html) if you haven't already installed either Anaconda or Miniconda
2. Open your conda terminal. 
3. Enter ```conda create --name outliers python=3.7``` in the terminal to create a new environment called outliers with python version 3.7]
4. Enter ```conda activate outliers``` in the terminal to enter your new environment. If that doesn't work, enter ```source activate outliers```
5. Enter ```conda install -c conda-forge matplotlib```
6. Enter ```pip install STAR-outliers```
7. Run ```python -m STAR_outliers --input [file_path]``` to remove univariate outliers from every column.
8. If one column is a sample index, then specify that with ```path -m STAR_outliers --input [file_path] --index [index_name]```

## Input :turkey:

## Output :poultry_leg:

## thing1 :computer:

## thing2 :abacus:

## :apple: thing3 :green_apple:

### Example 1: ... :arrow_lower_right:

### Example 2: ... :arrow_heading_down:

### Example 3: ... :twisted_rightwards_arrows:

## supplementary details :european_castle:

Thank you concerned reader (for making it this far)!

But our full analysis is in [another repository](https://github.com/EpistasisLab/regens-analysis)!

## Repository structure

### Folders in the repository :file_cabinet:

  * `images`: contains figures that are either displayed or linked to in this github README
  * `paper`: A directory containing the paper's md file, bib file, and figure
  * `thinning_methods`: All code that was used to select 500000 SNPs from the 1000 genomes project's genotype data

### Files :file_folder:

  * `regens.py`: the main file that runs the regens algorithm
  * `regens_library.py`: functions that the regens algorithm uses repeatedly. 
  * `regens_testers.py`: functions used exclusively for correctness testing and unit testing
  * `setup.py` and `_init_.py`: allows regens to be installed with pip
  * `requirements.txt`: lists REGENS' dependencies
  * `regens_tests_info.md`: Installing REGENS also downloads four folders that test REGENS' functionality. The [regens_tests_info.md](https://github.com/EpistasisLab/regens/blob/main/regens_tests_info.md) file explains what they test.   
  
## Contributing :thumbsup:
If you find any bugs or have any suggestions/questions, please feel free to [post an issue](https://github.com/EpistasisLab/regens/issues/new)! 
Please refer to our [contribution guide](CONTRIBUTING.md) for more details.
Thanks for your support!

## License
MIT + file LICENSE
