---
layout: post
title: Data Analysis Notes - Python
---

## Common terms
* **mean** (np.mean()): average
* **spread**: A measure of how spread out the values in a distribution are
* **variance** (np.var()) : A summary statistic often used to quantify spread
* **standard deviation** (np.std()) : The square root of variance, also used as a measure of spread
* **mode** : the value that appears most often in a set of data. [scipy-func](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.mode.html)
```python
# mode calculation using scipy
a = np.array([1,1,1])
from scipy import stats
stats.mode(a)
```
* **normal distribution** : An idealization of a bell-shaped distribution; also known as a Gaussian distribution
* **uniform distribution**: A distribution in which all values have the same
frequency
* **tail**: The part of a distribution at the high and low extremes
* **outlier**: A value far from the central tendency
* **normalization** : [wiki](https://en.wikipedia.org/wiki/Normalization_(statistics))
* **PMF** : Probability mass function (PMF) a representation of a distribution as a function that maps from values to probabilities. good for small dataset
* **CMF** : cumulative distribution function [wiki](https://en.wikipedia.org/wiki/Cumulative_distribution_function) , [quantile](https://en.wikipedia.org/wiki/Quantile) , [pandas-quantile](http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.quantile.html) , [numpy-percentile](https://docs.scipy.org/doc/numpy-dev/reference/generated/numpy.percentile.html)

## Pearson's R
[**wiki**](https://en.wikipedia.org/wiki/Pearson_correlation_coefficient)

> ranges  from -1 to 1 and represents correlation

```python
# x, y are numpy array or pandas series
def pearson_r(x , y):

    #standardize x and y
    xstd = (x - x.mean())/x.std()
    ystd = (y - y.mean())/y.std()

    pr = (xstd*ystd).mean()

    return pr
```

## Effect size
[**wiki cohen's d**](https://en.wikipedia.org/wiki/Effect_size#Cohen.27s_d)
> describes the size of an effect

```python
def CohenEffectSize(group1, group2):
    """Computes Cohen's effect size for two groups.
    
    group1: Series or DataFrame
    group2: Series or DataFrame
    
    returns: float if the arguments are Series;
             Series if the arguments are DataFrames
    """
    diff = group1.mean() - group2.mean()

    var1 = group1.var()
    var2 = group2.var()
    n1, n2 = len(group1), len(group2)

    pooled_var = (n1 * var1 + n2 * var2) / (n1 + n2)
    d = diff / np.sqrt(pooled_var)
    return d
    
```

## Anlaytical Distributions
> used to fit an empirical distribution (the distribution of actual data)
* exponential distribution
* normal distribution (Gaussian) : charaterstic parameters : mean and std (standard mean = 0, std = 1)
* [Normal probability Plot](https://en.wikipedia.org/wiki/Normal_probability_plot) : A plot of the values in a sample versus random values from a standard normal distribution. used to identify outliers, skewness, kurtosis, a need for transformations, and mixtures
* [log-normal distribution](https://en.wikipedia.org/wiki/Log-normal_distribution)
* [pareto distribution](https://en.wikipedia.org/wiki/Pareto_distribution)

## PDF : probability density function
> derivative of CDF (description?)<br>
> integral of continuous PDF gives expected value
[video](https://www.youtube.com/watch?v=Fvi9A_tEmXQ)
* [KDE](https://en.wikipedia.org/wiki/Kernel_density_estimation) : is an algorithm that takes a sample and finds an appropriately smooth PDF that fits the data. use cases : visualization, Interpolation, Simulation

A framework that relates representations of distribution func-
tions.
from
![PMF-CDF-PDF](public/pmf_cdf_pdf.JPG)

* pearson's median skewness [wiki](https://en.wikipedia.org/wiki/Skewness#Pearson.27s_second_skewness_coefficient_.28median_skewness.29)

> positive, negative or zero.

```python
def PearsonMedianSkewness(xs):
    median = Median(xs)
    mean = RawMoment(xs, 1)
    var = CentralMoment(xs, 2)
    std = np.sqrt(var)
    gp = 3 * (mean - median) / std
    return gp
```

## Multivariate

* scatter plot : with and without jitter
* covariance :measure of the tendency of two variables to vary together
```python
def Cov(xs, ys, meanx=None, meany=None):
    xs = np.asarray(xs)
    ys = np.asarray(ys)

    if meanx is None:
        meanx = np.mean(xs)
    if meany is None:
        meany = np.mean(ys)

    cov = np.dot(xs-meanx, ys-meany) / len(xs)
    return cov
```

* correlation : statistic intended to quantify the strength of the relationship between two variables.
```python
def Corr(xs, ys):
    xs = np.asarray(xs)
    ys = np.asarray(ys)

    meanx, varx = thinkstats2.MeanVar(xs)
    meany, vary = thinkstats2.MeanVar(ys)

    corr = Cov(xs, ys, meanx, meany) / np.sqrt(varx * vary)
    return corr

np.corrcoef(xs, ys)
```

* Pearson's correlation : defined above (Corr) : is not robust in the presence of outliers, and it tends to underestimate the strength of non-linear relationships.
* Spearman's correlation is more robust, and it can handle non-linear relationships as long as they are monotonic. Here's a function that computes Spearman's correlation
```python
import pandas as pd

def SpearmanCorr(xs, ys):
    xranks = pd.Series(xs).rank()
    yranks = pd.Series(ys).rank()
    return Corr(xranks, yranks)
```
* [interseting link](https://en.wikipedia.org/wiki/Correlation_and_dependence) , 
[one more](https://en.wikipedia.org/wiki/Correlation_does_not_imply_causation)


## estimation :later !!!

## hypothesis testing ??

## [chi-squared test](https://en.wikipedia.org/wiki/Chi-squared_test)

## central limit theoram
