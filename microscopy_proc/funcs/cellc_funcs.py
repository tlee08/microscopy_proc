import logging

import cupy as cp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from cupyx.scipy.ndimage import gaussian_filter as cupy_gaussian_filter
from cupyx.scipy.ndimage import label as cupy_label
from cupyx.scipy.ndimage import maximum_filter as cupy_maximum_filter
from cupyx.scipy.ndimage import white_tophat as cupy_white_tophat
from skimage.segmentation import watershed

from microscopy_proc.utils.cp_utils import (
    clear_cuda_memory_decorator,
    numpy_2_cupy_decorator,
)


@clear_cuda_memory_decorator
@numpy_2_cupy_decorator()
def tophat_filter(arr: np.ndarray, sigma=10) -> np.ndarray:
    """
    Top hat is calculated as:

    ```
    res = img - max_filter(min_filter(img, sigma), sigma)
    ```
    """
    logging.debug("Perform white top-hat filter")
    res = cupy_white_tophat(arr, sigma)
    logging.debug("ReLu")
    res = cp.maximum(res, 0)
    # Returning
    return res


@clear_cuda_memory_decorator
@numpy_2_cupy_decorator()
def dog_filter(arr: np.ndarray, sigma1=1.0, sigma2=2.0) -> np.ndarray:
    logging.debug("Making gaussian blur 1")
    gaus1 = cupy_gaussian_filter(arr, sigma=sigma1)
    logging.debug("Making gaussian blur 2")
    gaus2 = cupy_gaussian_filter(arr, sigma=sigma2)
    logging.debug("Subtracting gaussian blurs")
    res = gaus1 - gaus2
    logging.debug("ReLu")
    res = cp.maximum(res, 0)
    # Returning
    return res


@clear_cuda_memory_decorator
@numpy_2_cupy_decorator()
def gaussian_subtraction_filter(arr: np.ndarray, sigma=1.0) -> np.ndarray:
    logging.debug("Calculate local Gaussian blur")
    gaus = cupy_gaussian_filter(arr, sigma=sigma)
    logging.debug("Apply the adaptive filter")
    res = arr - gaus
    logging.debug("ReLu")
    res = cp.maximum(res, 0)
    # Returning
    return res


@clear_cuda_memory_decorator
@numpy_2_cupy_decorator()
def intensity_cutoff(arr: np.ndarray, min_=None, max_=None) -> np.ndarray:
    """
    Performing cutoffs on a 3D tensor.
    """
    logging.debug("Making cutoffs")
    res = arr
    if min_ is not None:
        res = cp.maximum(res, min_)
    if max_ is not None:
        res = cp.minimum(res, max_)
    # Returning
    return res


@clear_cuda_memory_decorator
@numpy_2_cupy_decorator(out_type=np.uint8)
def otsu_thresholding(arr: np.ndarray) -> np.ndarray:
    """
    Perform Otsu's thresholding on a 3D tensor.
    """
    logging.debug("Calculate histogram")
    hist, bin_edges = cp.histogram(arr, bins=256)
    logging.debug("Normalize histogram")
    prob_hist = hist / hist.sum()
    logging.debug("Compute cumulative sum and cumulative mean")
    cum_sum = cp.cumsum(prob_hist, dim=0)
    cum_mean = cp.cumsum(prob_hist * cp.arange(256), dim=0)
    logging.debug("Compute global mean")
    global_mean = cum_mean[-1]
    logging.debug(
        "Compute between class variance for all thresholds and find the threshold that maximizes it"
    )
    numerator = (global_mean * cum_sum - cum_mean) ** 2
    denominator = cum_sum * (1.0 - cum_sum)
    logging.debug("Avoid division by zero")
    denominator = cp.where(denominator == 0, float("inf"), denominator)
    between_class_variance = numerator / denominator
    logging.debug("Find the threshold that maximizes the between class variance")
    optimal_threshold = cp.argmax(between_class_variance)
    logging.debug("Apply threshold")
    res = (arr > optimal_threshold).float()
    # Returning
    return res


@clear_cuda_memory_decorator
@numpy_2_cupy_decorator(out_type=np.uint8)
def mean_thresholding(arr: np.ndarray, offset_sd: float = 0.0) -> np.ndarray:
    """
    Perform adaptive thresholding on a 3D tensor on GPU.
    """
    logging.debug("Get mean and std of ONLY non-zero values")
    arr0 = arr[arr > 0]
    mu = arr0.mean().get()
    sd = arr0.std().get()
    logging.debug("Apply the threshold")
    res = arr > mu + offset_sd * sd
    # Returning
    return res


@clear_cuda_memory_decorator
@numpy_2_cupy_decorator(out_type=np.uint8)
def manual_thresholding(arr: np.ndarray, val: int):
    """
    Perform manual thresholding on a tensor.
    """
    logging.debug("Applying the threshold")
    res = arr > val
    # Returning
    return res


@clear_cuda_memory_decorator
@numpy_2_cupy_decorator(out_type=np.uint32)
def label_objects(arr: np.ndarray) -> np.ndarray:
    """
    Label objects in a 3D tensor.
    """
    logging.debug("Label objects")
    res, counts = cupy_label(arr)
    logging.debug("Returning")
    return res


@clear_cuda_memory_decorator
def get_sizes(arr: np.ndarray) -> pd.Series:
    """
    Get statistics from a labeled 3D tensor.

    `arr` is a 3D tensor of labels.
    """
    # Choosing numpy or cupy
    xp = cp
    logging.debug("Getting sizes of labels")
    arr = xp.asarray(arr)
    arr0 = arr[arr > 0]
    ids, counts = cp.unique(arr0, return_counts=True)
    # Returning
    return pd.Series(
        counts.get().astype(np.uint32),
        index=pd.Index(ids.get().astype(np.uint32), name="label"),
        name="size",
    )


@clear_cuda_memory_decorator
@numpy_2_cupy_decorator(in_type=cp.uint32, out_type=np.uint32)
def labels_map(arr: np.ndarray, vect: pd.Series) -> np.ndarray:
    """
    NOTE: assumes the `vect` index is incrementing from 1
    """
    # Choosing numpy or cupy
    xp = cp
    logging.debug("Get vector of sizes (including 0)")
    vect_sizes = xp.concatenate([xp.asarray([0]), xp.asarray(vect.values)]).astype(
        xp.uint32
    )
    logging.debug("Convert arr of labels to arr of sizes")
    res = vect_sizes[arr]
    # Returning
    return res


def visualise_stats(vect: pd.Series):
    """
    Visualise statistics.
    """
    logging.debug("Making histogram")
    fig, ax = plt.subplots()
    sns.histplot(
        x=vect,
        log_scale=True,
        ax=ax,
    )


@clear_cuda_memory_decorator
@numpy_2_cupy_decorator(out_type=np.uint32)
def filter_large_objects(
    arr: np.ndarray, vect: pd.Series, min_size=None, max_size=None
):
    # Choosing numpy or cupy
    xp = cp
    logging.debug("Getting array of small and large object to filter out")
    min_size = min_size if min_size is not None else 0
    max_size = max_size if max_size is not None else vect.max()
    filt_objs = xp.asarray(vect[(vect < min_size) | (vect > max_size)].index.values)
    logging.debug("Filter out objects (by setting them to 0)")
    arr[xp.isin(arr, filt_objs)] = 0
    # Returning
    return arr


@clear_cuda_memory_decorator
@numpy_2_cupy_decorator(in_type=cp.int32, out_type=np.uint8)
def get_local_maxima(arr: np.ndarray, sigma=10):
    """
    NOTE: there can be multiple maxima per label
    """
    logging.debug("Making max filter for raw arr (holds the maximum in given area)")
    max_arr = cupy_maximum_filter(arr, sigma)
    logging.debug("Add 1 (so we separate the max pixel from the max_filter)")
    arr = arr + 1
    logging.debug("Getting local maxima (where arr - max_arr == 1)")
    res = arr - max_arr == 1
    # Returning
    return res


@clear_cuda_memory_decorator
@numpy_2_cupy_decorator(in_type=cp.int32, out_type=np.uint8)
def mask(arr: np.ndarray, mask: np.ndarray):
    # Choosing numpy or cupy
    xp = cp
    logging.debug("Masking for only maxima within mask")
    res = arr * xp.asarray(mask > 0)
    # Returning
    return res


@clear_cuda_memory_decorator
def watershed_segm(arr_raw: np.ndarray, arr_maxima: np.ndarray, arr_mask: np.ndarray):
    """
    NOTE: NOT GPU accelerated

    Expects `arr_maxima` to have unique labels for each maxima.
    """
    # Choosing numpy or cupy
    xp = np
    logging.debug("Converting other arrays to chosen type")
    arr_raw = xp.asarray(arr_raw)
    arr_maxima = xp.asarray(arr_maxima)
    arr_mask = xp.asarray(arr_mask)
    logging.debug("Padding everything with a 1 pixel empty border")
    arr_raw = xp.pad(arr_raw, pad_width=1, mode="constant", constant_values=0)
    arr_maxima = xp.pad(arr_maxima, pad_width=1, mode="constant", constant_values=0)
    arr_mask = xp.pad(arr_mask, pad_width=1, mode="constant", constant_values=0)
    logging.debug("Watershed segmentation")
    res = watershed(
        image=-arr_raw,
        markers=arr_maxima,
        mask=arr_mask > 0,
    )
    logging.debug("Unpadding")
    res = res[1:-1, 1:-1, 1:-1]
    # Returning
    return res


@clear_cuda_memory_decorator
def region_to_coords(arr_region: np.ndarray):
    """
    Get coordinates of regions in 3D tensor.

    Keeps only the first row (i.e cell) for each label.
    """
    # Choosing numpy or cupy
    xp = cp
    logging.debug("Converting to cupy arrays")
    arr_region = xp.asarray(arr_region)
    logging.debug("Getting coordinates of regions")
    z, y, x = xp.where(arr_region)
    logging.debug("Getting IDs of regions (from coords)")
    ids = arr_region[z, y, x]
    logging.debug("Making dataframe")
    df = pd.DataFrame(
        {
            "z": z.get().astype(np.uint16),
            "y": y.get().astype(np.uint16),
            "x": x.get().astype(np.uint16),
        },
        index=pd.Index(ids.get().astype(np.uint32), name="label"),
    )
    # Returning
    return df


@clear_cuda_memory_decorator
def maxima_to_coords(arr_maxima: np.ndarray):
    """
    Get coordinates of maxima in 3D tensor.

    Expects `arr_maxima` to have unique labels for each maxima.

    Keeps only the first row (i.e cell) for each label.
    """
    df = region_to_coords(arr_maxima)
    logging.debug("Keeping only first row per label (some maxima may be contiguous)")
    df = df.groupby("label").first()
    # Returning
    return df
