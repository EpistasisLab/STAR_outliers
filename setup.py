from setuptools import setup
from setuptools import find_packages

summary = "outlier_removal is an open source python "
summary += "package that determines datapoints' "
summary += "outlier status in a manner that adjusts "
summary += "for skew and kurtosis. An exponential "
summary += "tail fit is used to determine outlier "
summary += "status if the distribution is behaves in "
summary += "a sufficiently exponential-like manner. "
summary += "Otherwise, the data is transformed and "
summary += "fitted to a four parameteter tukey "
summary += "distribution as described in the paper "
summary += "titled 'Outlier identification for skewed "
summary += "and/or heavy-tailed unimodal multivariate "
summary += "distributions'. \n\nplease visit the "
summary += "[github page](https://github.com/Epistasis"
summary += "Lab/outlier_removal) for more information."

ep_val = {"console_scripts": ["remove_outliers = remove_outliers:main"]}
setup(
    long_description = summary,
    long_description_content_type = "text/markdown",
    packages = find_packages(),
    version = "0.1.0",
    python_requires = ">=3.6,<=3.9"
    name = "remove_outliers",
    entry_points = ep_val,
    py_modules=["remove_outliers", "remove_outliers_library",
                "remove_outliers_plotting_library",
                "remove_outliers_polishing_library",
                "remove_outliers_testing_library"],
    install_requires=["numpy",
                      "pandas",
                      "tqdm",
                      "scipy",
                      "matplotlib",
                      "mock"],
)
