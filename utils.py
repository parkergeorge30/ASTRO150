import numpy as np
import matplotlib.pyplot as plt
import astropy.io.fits as fits
import math
import pandas as pd

def zeros(x, dtype=int):
    return [dtype(0.0) * int(x)]


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

