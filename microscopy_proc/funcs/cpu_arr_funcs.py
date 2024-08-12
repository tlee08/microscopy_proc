import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# from prefect import task
from scipy import ndimage as sc_ndimage
from skimage.segmentation import watershed

from microscopy_proc.constants import DEPTH


class CpuArrFuncs:
    xp = np
    xdimage = sc_ndimage

    @classmethod
    # @task
    def tophat_filt(cls, arr: np.ndarray, sigma: int = 10) -> np.ndarray:
        """
        Top hat is calculated as:

        ```
        res = arr - max_filter(min_filter(arr, sigma), sigma)
        ```
        """
        arr = cls.xp.asarray(arr).astype(cls.xp.float32)
        logging.debug("Perform white top-hat filter")
        logging.debug(f"TYPE: {arr.dtype} {type(arr)}")
        res = cls.xdimage.white_tophat(arr, sigma)
        logging.debug("ReLu")
        res = cls.xp.maximum(res, 0)
        # Returning
        return res.astype(cls.xp.uint16)

    @classmethod
    # @task
    def dog_filt(cls, arr: np.ndarray, sigma1=1, sigma2=2) -> np.ndarray:
        arr = cls.xp.asarray(arr).astype(cls.xp.float32)
        logging.debug("Making gaussian blur 1")
        gaus1 = cls.xdimage.gaussian_filter(arr, sigma=sigma1)
        logging.debug("Making gaussian blur 2")
        gaus2 = cls.xdimage.gaussian_filter(arr, sigma=sigma2)
        logging.debug("Subtracting gaussian blurs")
        res = gaus1 - gaus2
        logging.debug("ReLu")
        res = cls.xp.maximum(res, 0)
        # Returning
        return res.astype(cls.xp.uint16)

    @classmethod
    # @task
    def gauss_subt_filt(cls, arr: np.ndarray, sigma=10) -> np.ndarray:
        arr = cls.xp.asarray(arr).astype(cls.xp.float32)
        logging.debug("Calculate local Gaussian blur")
        gaus = cls.xdimage.gaussian_filter(arr, sigma=sigma)
        logging.debug("Apply the adaptive filter")
        res = arr - gaus
        logging.debug("ReLu")
        res = cls.xp.maximum(res, 0)
        # Returning
        return res.astype(cls.xp.uint16)

    @classmethod
    # @task
    def intensity_cutoff(cls, arr: np.ndarray, min_=None, max_=None) -> np.ndarray:
        """
        Performing cutoffs on a 3D tensor.
        """
        arr = cls.xp.asarray(arr)
        logging.debug("Making cutoffs")
        res = arr
        if min_ is not None:
            res = cls.xp.maximum(res, min_)
        if max_ is not None:
            res = cls.xp.minimum(res, max_)
        # Returning
        return res

    @classmethod
    # @task
    def otsu_thresh(cls, arr: np.ndarray) -> np.ndarray:
        """
        Perform Otsu's thresholding on a 3D tensor.
        """
        arr = cls.xp.asarray(arr)
        logging.debug("Calculate histogram")
        hist, bin_edges = cls.xp.histogram(arr, bins=256)
        logging.debug("Normalize histogram")
        prob_hist = hist / hist.sum()
        logging.debug("Compute cumulative sum and cumulative mean")
        cum_sum = cls.xp.cumsum(prob_hist)
        cum_mean = cls.xp.cumsum(prob_hist * cls.xp.arange(256))
        logging.debug("Compute global mean")
        global_mean = cum_mean[-1]
        logging.debug(
            "Compute between class variance for all thresholds and find the threshold that maximizes it"
        )
        numerator = (global_mean * cum_sum - cum_mean) ** 2
        denominator = cum_sum * (1.0 - cum_sum)
        logging.debug("Avoid division by zero")
        denominator = cls.xp.where(denominator == 0, float("inf"), denominator)
        between_class_variance = numerator / denominator
        logging.debug("Find the threshold that maximizes the between class variance")
        optimal_threshold = cls.xp.argmax(between_class_variance)
        logging.debug("Apply threshold")
        res = arr > optimal_threshold
        # Returning
        return res.astype(cls.xp.uint8)

    @classmethod
    # @task
    def mean_thresh(cls, arr: np.ndarray, offset_sd: float = 0.0) -> np.ndarray:
        """
        Perform adaptive thresholding on a 3D tensor on GPU.
        """
        arr = cls.xp.asarray(arr)
        logging.debug("Get mean and std of ONLY non-zero values")
        arr0 = arr[arr > 0]
        mu = arr0.mean()
        sd = arr0.std()
        logging.debug("Apply the threshold")
        res = arr > mu + offset_sd * sd
        # Returning
        return res.astype(cls.xp.uint8)

    @classmethod
    # @task
    def manual_thresh(cls, arr: np.ndarray, val: int):
        """
        Perform manual thresholding on a tensor.
        """
        arr = cls.xp.asarray(arr)
        logging.debug("Applying the threshold")
        res = arr >= val
        # Returning
        return res.astype(cls.xp.uint8)

    @classmethod
    # @task
    def label_with_ids(cls, arr: np.ndarray) -> np.ndarray:
        """
        Label objects in a 3D tensor.
        """
        arr = cls.xp.asarray(arr).astype(cls.xp.uint8)
        logging.debug("Labelling contiguous objects uniquely")
        res, _ = cls.xdimage.label(arr)
        logging.debug("Returning")
        return res.astype(cls.xp.uint32)

    @classmethod
    # @task
    def ids_to_sizes(cls, arr: np.ndarray) -> np.ndarray:
        """
        Convert labels to sizes.
        """
        arr = cls.xp.asarray(arr)
        logging.debug("Getting vector of ids and sizes (not incl. 0)")
        ids, counts = cls.xp.unique(arr[arr > 0], return_counts=True)
        # NOTE: assumes ids are incrementing from 1
        counts = cls.xp.concatenate([cls.xp.asarray([0]), counts])
        logging.debug("Converting arr intensity to sizes")
        res = counts[arr]
        logging.debug("Returning")
        return res.astype(cls.xp.uint16)

    @classmethod
    # @task
    def label_with_sizes(cls, arr: np.ndarray) -> np.ndarray:
        """
        Label objects in a 3D tensor.
        """
        arr = cls.label_with_ids(arr)
        res = cls.ids_to_sizes(arr)
        return res

    @classmethod
    # @task
    def visualise_stats(cls, arr: np.ndarray):
        """
        Visualise statistics.

        NOTE: expects arr to be a 3D tensor of a property
        (e.g. size).
        """
        logging.debug("Converting arr to vector of the ids")
        ids = arr[arr > 0]
        logging.debug("Making histogram")
        fig, ax = plt.subplots()
        sns.histplot(
            x=ids,
            log_scale=True,
            ax=ax,
        )
        return fig

    @classmethod
    # @task
    def filt_by_size(cls, arr: np.ndarray, smin=None, smax=None):
        """
        Assumes `arr` is array of objects labelled with their size.
        """
        arr = cls.xp.asarray(arr)
        logging.debug("Getting filter of small and large object to filter out")
        smin = smin if smin is not None else 0
        smax = smax if smax is not None else arr.max()
        filt_objs = (arr < smin) | (arr > smax)
        logging.debug("Filter out objects (by setting them to 0)")
        arr[filt_objs] = 0
        # Returning
        return arr

    @classmethod
    # @task
    def get_local_maxima(
        cls,
        arr: np.ndarray,
        sigma: int = 10,
        arr_mask: np.ndarray = None,
        block_info=None,
    ):
        """
        Getting local maxima (no connectivity) in a 3D tensor.
        If there is a connected region of maxima, then only the centre point is kept.

        If `arr_mask` is provided, then only maxima within the mask are kept.
        """
        arr = cls.xp.asarray(arr)
        logging.debug("Making max filter for raw arr (holds the maximum in given area)")
        max_arr = cls.xdimage.maximum_filter(arr, sigma)
        logging.debug("Add 1 (so we separate the max pixel from the max_filter)")
        arr = arr + 1
        logging.debug("Getting local maxima (where arr - max_arr == 1)")
        res = arr - max_arr == 1
        # If a mask is given, then keep only the maxima within the mask
        if arr_mask is not None:
            logging.debug("Mask provided. Maxima will only be found within regions.")
            arr_mask = (cls.xp.asarray(arr_mask) > 0).astype(cls.xp.uint8)
            res = res * arr_mask
        # Returning
        return res

    @classmethod
    # @task
    def mask(cls, arr: np.ndarray, arr_mask: np.ndarray):
        arr = cls.xp.asarray(arr)
        arr_mask = cls.xp.asarray(arr_mask).astype(cls.xp.uint8)
        logging.debug("Masking for only maxima within mask")
        res = arr * (arr_mask > 0)
        # Returning
        return res

    @classmethod
    # @task
    def wshed_segm(
        cls, arr_raw: np.ndarray, arr_maxima: np.ndarray, arr_mask: np.ndarray
    ):
        """
        NOTE: NOT GPU accelerated

        Expects `arr_maxima` to have unique labels for each maxima.
        """
        logging.debug("Watershed segmentation")
        res = watershed(
            image=-arr_raw,
            markers=arr_maxima,
            mask=arr_mask > 0,
        )
        # Returning
        return res

    @classmethod
    # @task
    def wshed_segm_sizes(
        cls, arr_raw: np.ndarray, arr_maxima: np.ndarray, arr_mask: np.ndarray
    ):
        """
        NOTE: NOT GPU accelerated
        """
        # Labelling contiguous maxima with unique labels
        arr_maxima = cls.label_with_ids(arr_maxima)
        # Watershed segmentation
        arr_wshed = cls.wshed_segm(arr_raw, arr_maxima, arr_mask)
        # Getting sizes of watershed regions
        res = cls.ids_to_sizes(arr_wshed)
        # Returning
        return res

    @classmethod
    # @task
    def get_coords(cls, arr: np.ndarray):
        """
        Get coordinates of regions in 3D tensor.

        TODO: Keep only the first row (i.e cell) for each label (groupby).
        """
        logging.debug("Getting coordinates of regions")
        z, y, x = np.where(arr)
        logging.debug("Getting IDs of regions (from coords)")
        ids = arr[z, y, x]
        logging.debug("Making dataframe")
        df = pd.DataFrame(
            {"z": z, "y": y, "x": x},
            index=pd.Index(ids.astype(np.uint32), name="label"),
        ).astype(np.uint16)
        df["size"] = -1  # TODO: placeholder
        df["sum_itns"] = -1  # TODO: placeholder
        # df["max_itns"] = -1  # TODO: placeholder
        # Returning
        return df

    @classmethod
    # @task
    def get_cells(
        cls,
        arr_raw: np.ndarray,
        arr_overlap: np.ndarray,
        arr_maxima: np.ndarray,
        arr_mask: np.ndarray,
        d: int = DEPTH,
    ):
        """
        Get the cells from the maxima labels and the watershed segmentation
        (with corresponding labels).
        """
        # Asserting arr sizes
        assert arr_raw.shape == tuple(i - 2 * d for i in arr_overlap.shape)
        logging.debug("Trimming maxima labels array to raw array dimensions using `d`")
        slicer = slice(d, -d) if d > 0 else slice(None)
        arr_maxima = arr_maxima[slicer, slicer, slicer]
        logging.debug("Getting unique labels in arr_maxima")
        arr_maxima_l = cls.label_with_ids(arr_maxima)
        logging.debug("Converting to DataFrame of coordinates and sizes")
        # NOTE: getting first coord of each unique label
        ids_m, ind = cls.xp.unique(arr_maxima_l, return_index=True)
        z, y, x = cls.xp.unravel_index(ind, arr_maxima_l.shape)
        df = (
            pd.DataFrame(
                {"z": arr_cp2np(z), "y": arr_cp2np(y), "x": arr_cp2np(x)},
                index=pd.Index(arr_cp2np(ids_m).astype(np.uint32), name="label"),
            )
            .drop(index=0)
            .astype(np.uint16)
        )
        logging.debug("Watershed of arr_overlap, seeds arr_maxima, mask arr_mask")
        arr_maxima_l = np.pad(
            arr_cp2np(arr_maxima_l), d, mode="constant", constant_values=0
        )
        arr_wshed = cls.wshed_segm(arr_overlap, arr_maxima_l, arr_mask)
        logging.debug("Making vector of region sizes (corresponding to maxima)")
        ids_w, counts = cls.xp.unique(arr_wshed[arr_wshed > 0], return_counts=True)
        ids_w = arr_cp2np(ids_w).astype(np.uint32)
        counts = arr_cp2np(counts).astype(np.uint32)
        logging.debug("Getting sum intensity for each cell (wshed)")
        sum_itns = cls.xp.bincount(
            x=cls.xp.asarray(arr_wshed[arr_wshed > 0].ravel()),
            weights=cls.xp.asarray(arr_overlap[arr_wshed > 0].ravel()),
            minlength=len(ids_w),
        )
        sum_itns = arr_cp2np(sum_itns[sum_itns > 0])
        logging.debug("Adding sizes and intensities to DataFrame")
        idx = pd.Index(ids_w, name="label")
        df["size"] = pd.Series(counts, index=idx)
        df["sum_itns"] = pd.Series(sum_itns, index=idx)
        # df["max_itns"] = pd.Series(max_itns, index=idx)
        # Filtering out rows with NaNs in z, y, or x columns
        df = df[df[["z", "y", "x"]].isna().mean(axis=1) == 0]
        # Returning
        return df


def arr_cp2np(arr):
    try:
        return arr.get()
    except Exception:
        return arr
