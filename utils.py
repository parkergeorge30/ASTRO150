import numpy as np
import astropy.io.fits as fits

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

# [EVAN WATSON]
def get_data(files):
    """
    grabs data from name of fits file

    Usage:
        files = ['data1.fits', 'data2.fits']
        data_array = get_data(files)

    :param files: str or list of strs, name of files to get data from
    :return: array, data array
    """
    # checks to see if files is a list, if not, make it one
    if type(files) is str:
        files = [files]
    n = len(files)

    # if only 1 item in list, just return the data for that str
    if n == 1:
        return fits.getdata(files[0])

    # otherwise initialize empty array and fill it with data
    data = zeros(n, dtype=float)
    for i, file in enumerate(files):
        data[i] = fits.getdata(file)

    return data

def my_bias(x):
    n = len(x)
    tot = zeros(n, dtype=float)

    for i, file in enumerate(x):
        arr = fits.getdata(file)
        tot[i] = arr
    avg = my_avg(tot)
    return avg

def bias_sub(x, bias, num_arrays=None):
    n = len(x)
    tot = zeros(n, dtype=float)

    if num_arrays == 1:
        return x - bias

    for i, data in enumerate(x):
        tot[i] = data - bias
    return tot

def normalize_flats(data):
    n = len(data)
    tot = zeros(n, dtype=float)

    for i, data in enumerate(data):
        tot[i] = data/(np.median(data))

    return my_avg(tot)

def norm(data, norm_flat):
    return data/norm_flat

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
        # don't count end points ("i-n" is negative version of index)
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