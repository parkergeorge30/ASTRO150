import numpy as np
import matplotlib.pyplot as plt
import astropy.io.fits as fits
import math
import pandas as pd

def zeros(x, dtype=int):
    return [dtype(0.0)] * int(x)


def my_sum(x):
    tot = 0.
    for i in x:
        tot += i
    return tot

def my_avg(x):
    return my_sum(x) / len(x)

def my_std(x):
    avg = my_avg(x)
    tot = 0
    for i in x:
        tot += (i - avg) ** 2

    return (tot / len(x)) ** 0.5

def histo(file):

    farr = file.flatten()

    hmin = int(farr.min())
    hmax = int(farr.max())
    #hmax = int(farr.max())

    # initialize bins and empty count x[i]
    hr = np.arange(hmin, hmax + 1)
    hist = zeros(hmax - hmin + 1, dtype=int)

    # count each value in bins
    for value in farr:
        if hmin <= value <= hmax:
            hist[int(value - hmin)] += 1

    return hr, hist

def ascend_str(ls, idxs):
    idx = 0
    idx_slice = slice(idxs[0], idxs[1])  # make slice for desired indices
    while idx < len(ls):
        for i, x in enumerate(ls):
            if i > idx and int(x[idx_slice]) < int(ls[idx][idx_slice]):  # check for lowest integer at given indices
                ls[i], ls[idx] = ls[idx], x

        idx += 1
    return None

# Evan Watson's Function
def get_centroids(wavelengths, intensities, threshold=None, threshold_lim=0.01, scope=20, return_indices=False):
    """
    gets centroid wavelengths for all peaks in intensity above certain threshold
    for a peak to count it has to be largest in scope radius
    meaning largest out of [scope] number of points forward and back

    Usage:
        get_centroids(wavelengths, intensities, threshold=0.01, scope=20)
        get_centroids(wavelengths, intensities, threshold_lim=0.005, scope=5)

    :param wavelengths: array-like of wavelengths
    :param intensities: array-like of intensities
    :param threshold: float, threshold value, adjust in tandem with scope
        if None, automatically determine threshold to be 1.1x mean value of intensities
    :param threshold_lim: float, minimum threshold if automatically determining threshold, default 0.01
    :param scope: int, scope radius, default 20,
        larger scope ensures no false peaks within small bump regions
        smaller scope allows for more peaks to be found (potentially false peaks too)
    :param return_indices: bool, whether or not to return indices of all_centroids (pixel numbers)
    :return: list of centroid wavelengths
    """
    if threshold is None:
        threshold = 1.1 * my_avg(intensities)
        if threshold < threshold_lim:
            threshold = threshold_lim

    n = len(intensities)
    peak_indices = []
    for i, I in enumerate(intensities):
        # dont count end points ("i-n" is negative version of index)
        if i < scope or (i - n) >= -scope:
            continue

        if I > threshold:
            # build like [i+scope, i+(scope-1), ..., i-(scope-1), i-scope]
            vals = [intensities[i - j] for j in range(-scope, scope + 1) if j != 0]

            # make sure intensity is peak of scope
            if I > max(vals):
                peak_indices.append(i)

    centroids = []
    for k, peak_idx in enumerate(peak_indices.copy()):
        # go left until intensity drops below threshold
        left = peak_idx
        while left > 0 and intensities[left - 1] > threshold:
            left -= 1

        # repeat right
        right = peak_idx
        while right < n - 1 and intensities[right + 1] > threshold:
            right += 1

        regional_wavelengths = wavelengths[left:right + 1]
        regional_intensities = intensities[left:right + 1]
        centroid = np.matmul(regional_wavelengths, regional_intensities) / my_sum(regional_intensities)
        if centroid in centroids:
            peak_indices.remove(peak_idx)
        else:
            centroids.append(centroid)

    if return_indices:
        return np.array(peak_indices), np.array(centroids)
    else:
        return np.array(centroids)


def linear_least_squares(x, y):
    """
    Linear Least Square Fit

    :param x: np.array
    :param y: np.array
    :return: m, c tuple representing (slope, intercept)
    """
    n = x.shape[0]
    x_sum = my_sum(x)
    A = np.array([[my_sum(x**2), x_sum], [x_sum, n]])
    B = np.array([[np.matmul(x, y)], [my_sum(y)]])
    A_inv = np.linalg.inv(A)
    O = np.matmul(A_inv, B)
    m = O[0, 0]
    c = O[1, 0]
    return m, c