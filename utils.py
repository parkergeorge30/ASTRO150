import astropy.io.fits as fits
import astropy.units as u
import numpy as np
from astroquery.vizier import Vizier
from astropy.coordinates import SkyCoord
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.ndimage import label

#############
### LAB 1 ###
#############

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

def ascend_str(ls, idxs):
    idx = 0
    idx_slice = slice(idxs[0], idxs[1])  # make slice for desired indices
    while idx < len(ls):
        for i, x in enumerate(ls):
            if i > idx and int(x[idx_slice]) < int(ls[idx][idx_slice]):  # check for lowest integer at given indices
                ls[i], ls[idx] = ls[idx], x

        idx += 1
    return None

# [EVAN WATSON]: Sets the negative values in a file array to zero for cleaner value control
def set_negatives_to_zero_nd(tensor):
    """
    sets negative values to 0 inplace for a rank n tensor
    """
    # check for rank 1
    ele = tensor[0]
    if isinstance(ele, np.ndarray):
        # not inside rank 1 yet so recursively loop with self call
        for sub in tensor:
            set_negatives_to_zero_nd(sub)
    else:
        # we are inside the rank 1 now
        for i, val in enumerate(tensor):
            if val < 0:                         # if less than zero
                tensor[i] = 0                   # set to zero
    return None

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

#############
### LAB 2 ###
#############

def normalize_flats(data):
    n = len(data)
    tot = zeros(n, dtype=float)

    for i, data in enumerate(data):
        tot[i] = data/(np.median(data))

    return np.median(tot, axis=0)

def norm(data, norm_flat):
    return data/(norm_flat + 1e-2)

# [EVAN WATSON]
def get_centroids(wavelengths, intensities, threshold=None, threshold_lim=0.01, scope=20, return_indices=True):
    """
    gets centroid wavelengths for all peaks in intensity above certain threshold
    for a peak to count it has to be largest in scope radius
    meaning largest out of [scope] number of points forward and back

    also returns variance of centroids

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
    :return: if return_indices, (peaks, centroids, error), otherwise (centroids, error)
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
    centroid_errors = []
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
        # matmul acts as sum of element-wise product with 1d arrays
        centroid = np.matmul(regional_wavelengths, regional_intensities) / my_sum(regional_intensities)

        # Prevent repeats and keep same length
        if centroid in centroids:
            peak_indices.remove(peak_idx)
        else:
            centroids.append(centroid)
            # error prop for centroid, var = sigma^2
            variance = np.matmul(regional_intensities,
                                 (regional_wavelengths - centroid) ** 2) / my_sum(regional_intensities) ** 2
            centroid_errors.append(variance)

    if return_indices:
        return np.array(peak_indices), np.array(centroids), np.array(centroid_errors)
    else:
        return np.array(centroids), np.array(centroid_errors)

# [EVAN WATSON]
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

#############
### LAB 3 ###
#############

# [EVAN WATSON]
def remove_bad_cols(x, bad_cols):
    """
    removes bad columns from x by setting them to background

    :param x: 2d array, data with bad columns
    :param bad_cols: int or list, index or indices of bad columns
    :return: 2d array, frame with bad columns set to background
    """
    x[:, bad_cols] = np.median(x)
    return x

# [EVAN WATSON]
def get_hdr_data(file, entry):
    """
    get header value for a file

    Usage:
        exp = get_hdr_data('data1.fits', 'EXPTIME')

    :param file: str, name of file
    :param entry: str, name of header entry storing desired value
    :return: object, value of header entry
    """
    with fits.open(file) as hdu:
        hdr = hdu[0].header

    return hdr[entry]

# [EVAN WATSON]
def load_headers_all_files(headers, data_files=None, data_dir=None):
    """
    load values for each header entry for all files in a 2D array
    rows being headers, and cols being data_file

    :param headers: list, list of header entries
    :param data_files: list, list of data files
    :param data_dir: str, optional prefix for all data files
    :return: np.array, 2d array of values for each header for each file
    """
    if data_dir is None or not isinstance(data_dir, str):
        data_dir = globals().get('data_dir')
        if data_dir is None:
            data_dir = ""

    if data_files is None:
        data_files = globals().get('data_files')
        if data_files is None:
            raise ValueError("No available 'data_files' has been defined.")

    # All loaded headers will be saved here. np.full() creates a NumPy array of a given shape (first argument)
    # filled with a constant value (second argument, empty string in this case). "dtype = object" will allow
    # the array to store data of any type (some headers may be numbers, not strings).
    output = np.full([len(headers), len(data_files)], "", dtype=object)

    # Now read the headers from all files
    for i, hdr in enumerate(headers):
        for j, file in enumerate(data_files):
            output[i, j] = get_hdr_data(data_dir + file, hdr)

    return output

# [EVAN WATSON]
def load_frame_overscan_remove_bias_subtract(filename, bias, overscan=32, bad_col_idx=None):
    """
    load frame and subtract the bias and remove overscan

    :param filename: str or list, name or list of names of data_file
    :param bias: array, bias frame
    :param overscan: int, overscan value
    :param bad_col_idx: int, index of bad column
    :return: 2d or 3d np.array, clean frame or frames
    """
    if isinstance(filename, str):
        frame = get_data(filename)
        image = frame - bias
        set_negatives_to_zero_nd(image)

        # remove overscan (right black bar region)
        image = image[:, :int(np.shape(image)[1] - overscan)]

        # set bad col to background noise
        if isinstance(bad_col_idx, int):
            image[:, bad_col_idx] = np.median(image)

        return image

    elif isinstance(filename, list):
        images = zeros(len(filename), np.float32)
        for i, fname in enumerate(filename):
            frame = get_data(fname)
            images[i] = frame - bias
            set_negatives_to_zero_nd(images[i])

            # remove overscan (right black bar region)
            images[i] = images[i][:, :int(np.shape(images[i])[1] - overscan)]
        return images

# [EVAN WATSON]
def load_reduced_science_frame(filename, flat, bias):
    """
    load reduced science frame

    :param filename: str or list, path to frame file
    :param flat: array, cleaned flat frame
    :param bias: array, bias frame
    :return: array, normalized reduced frame
    """
    # bias sub data
    data_clean = load_frame_overscan_remove_bias_subtract(filename, bias)

    # norm with clean flat
    norm = data_clean / (flat + 1e-2)

    return norm

# [EVAN WATSON]
def plot_im(ax, data, xlabel='x (px)', ylabel='y (px)', title='', **imshow_kwargs):
    """
    Plot an image using matplotlib imshow.

    :param ax: plt.Axes, axes to plot on
    :param data: array, data to imshow
    :param xlabel: str, x label
    :param ylabel: str, y label
    :param title: str, title
    :param imshow_kwargs: iterable, imshow kwargs including colorbar axis kwargs 'pad' and 'size'
    :return: None, plots data using imshow
    """
    # extract cax kwargs from imshow_kwargs
    cax_keys = ['pad', 'size']
    cax_kwargs = {key: imshow_kwargs.pop(key) for key in cax_keys if key in imshow_kwargs}
    cax_kwargs.setdefault('size', "5%")
    cax_kwargs.setdefault('pad', 0.05)

    fig = ax.get_figure()
    ax.imshow(data, **imshow_kwargs)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size=cax_kwargs['size'], pad=cax_kwargs['pad'])
    fig.colorbar(ax.images[0], cax=cax)

def find_star_centroids(data, threshold=0.025):
    """
    Reads a 2D FITS file, detects stars above a threshold, and calculates centroids.

    Parameters:
        data (arr): Path to the FITS file.
        threshold (float): Intensity threshold for star detection.

    Returns:
        list of tuples: [(x1, y1), (x2, y2), ...] where (x, y) are centroid coordinates.
        list of Intensities: [I1, I2, ...].
    """
    # Identify bright regions (potential stars)
    mask = data > threshold
    labeled, num_features = label(mask)

    coords = []
    bright_count = []
    for i in range(1, num_features + 1):
        indices = np.argwhere(labeled == i)

        # Compute intensity-weighted centroid
        intensities = data[labeled == i]
        total_intensity = np.sum(intensities)
        x = np.sum(indices[:, 1] * intensities) / total_intensity
        y = np.sum(indices[:, 0] * intensities) / total_intensity
        if x > 995 or x < 50: # Make sure we are not in the bright corner or in the dead zone to the left of the rotated frame
            continue

        coords.append((y, x))
        bright_count.append(total_intensity)
    return coords, bright_count

# [EVAN WATSON]
def collapse_subsets(arr_list):
    """
    remove all subsets from a list, only keep the super sets (sets that are not subsets of another set)

    :param arr_list: list or array-likes
    :return: list of super sets present in arr_list
    """
    # convert arrays to sets
    sets = [set(arr) for arr in arr_list]
    keep = [True] * len(arr_list)

    # check for subsets
    for i, s in enumerate(sets):
        for j, t in enumerate(sets):
            if i != j:
                # if s is a strict (not duplicate) subset of t, mark it to be removed.
                if s.issubset(t) and len(s) < len(t):
                    keep[i] = False
                    break

    # remove duplicates
    filtered = []
    for flag, arr in zip(keep, arr_list):
        if flag and arr not in filtered:
            filtered.append(arr)

    return filtered

# [EVAN WATSON]
def sky_query(dataframe, filename, fov_width='6.3m', fov_height='6.3m', plate_scale=0.368, magnitude_limit=18):
    """
    vizier query the sky for objects around center of a file

    :param dataframe: pandas.DataFrame, dataframe containing the data of file, must have columns
                            ['FILE NAME'] ['RA'] ['DEC'] ['DATE-BEG']
    :param filename: str, name of file in dataframe
    :param fov_width: str, width of field of view in arcs, '6.3m' is 6.3 arcmin
    :param fov_height: str, height of field of view in arcs, '6.3m' is 6.3 arcmin
    :param plate_scale: float, CCD plate scale in arcseconds per pixel
    :param magnitude_limit: float, R2 magnitude limit
    :return: ra, dec arrays of queried objects
    """
    ra_center, dec_center, yr = dataframe.loc[dataframe['FILE NAME'] == filename, ['RA', 'DEC', 'DATE-BEG']].values[0]

    center_coord = SkyCoord(ra=ra_center, dec=dec_center, unit=(u.hour, u.deg), frame='fk5')

    vizier = Vizier(column_filters={"R2mag": f"<{magnitude_limit}"})
    result_table = vizier.query_region(center_coord, width=fov_width, height=fov_height, catalog="USNO-B1")

    # Extract required data from obtained query results
    ra_cat = np.array(result_table[0]["RAJ2000"])  # this is the stars' RA in the 2000 epoch
    dec_cat = np.array(result_table[0]["DEJ2000"])  # this is the stars' Dec in the 2000 epoch
    pm_ra = np.array(result_table[0]["pmRA"])  # this is the RA proper motion of the stars
    pm_dec = np.array(result_table[0]["pmDE"])  # this is the Dec proper motion of the stars
    mag = np.array(result_table[0]["R2mag"])

    # convert mas/yr to deg/yr
    pm_ra = pm_ra / 1000 / 3600
    pm_dec = pm_dec / 1000 / 3600

    # time in years since epoch (2000)
    dt = yr - 2000

    # add proper motion to epoch coordinates
    ra_cat = ra_cat + pm_ra * dt
    dec_cat = dec_cat + pm_dec * dt

    # get relative to center coord and change to arcsec
    ra_cat = (ra_cat - center_coord.ra.value) * 3600
    dec_cat = (dec_cat - center_coord.dec.value) * 3600

    # convert from as to px using plate scale
    ra_cat /= plate_scale
    dec_cat /= plate_scale

    # centering (image is 1024x1024)
    ra_cat += 512
    dec_cat += 512

    return ra_cat, dec_cat