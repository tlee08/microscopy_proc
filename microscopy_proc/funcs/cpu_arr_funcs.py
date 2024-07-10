import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import ndimage as sc_ndimage
from skimage.segmentation import watershed


class CpuArrFuncs:
    xp = np
    xdimage = sc_ndimage

    @classmethod
    def tophat_filt(cls, arr: np.ndarray, sigma: int = 10) -> np.ndarray:
        """
        Top hat is calculated as:

        ```
        res = img - max_filter(min_filter(img, sigma), sigma)
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
    def label_objects_with_ids(cls, arr: np.ndarray) -> np.ndarray:
        """
        Label objects in a 3D tensor.
        """
        arr = cls.xp.asarray(arr).astype(cls.xp.uint8)
        logging.debug("Labelling contiguous objects uniquely")
        res, _ = cls.xdimage.label(arr)
        logging.debug("Returning")
        return res.astype(cls.xp.uint32)

    @classmethod
    def label_objects_with_sizes(cls, arr: np.ndarray) -> np.ndarray:
        """
        Label objects in a 3D tensor.
        """
        arr = cls.xp.asarray(arr).astype(cls.xp.uint8)
        logging.debug("Labelling contiguous objects uniquely")
        arr, _ = cls.xdimage.label(arr)
        logging.debug("Getting vector of ids and sizes (not incl. 0)")
        ids, counts = cls.xp.unique(arr[arr > 0], return_counts=True)
        # NOTE: assumes ids is perfectly incrementing from 1
        counts = cls.xp.concatenate([cls.xp.asarray([0]), counts])
        logging.debug("Converting arr intensity to sizes")
        res = counts[arr]
        logging.debug("Returning")
        return res.astype(cls.xp.uint16)

    @classmethod
    def get_sizes(cls, arr: np.ndarray) -> pd.Series:
        """
        Get statistics from a labeled 3D tensor.

        `arr` is a 3D tensor of labels.
        """
        logging.debug("Getting sizes of labels")
        arr = cls.xp.asarray(arr)
        ids, counts = cls.xp.unique(arr[arr > 0], return_counts=True)
        # Returning
        return pd.Series(
            counts.astype(np.uint32),
            index=pd.Index(ids.astype(np.uint32), name="label"),
            name="size",
        )

    @classmethod
    def labels_map(cls, arr: np.ndarray, vect: pd.Series) -> np.ndarray:
        """
        NOTE: assumes the `vect` index is incrementing from 1
        """
        arr = cls.xp.asarray(arr)
        logging.debug("Get vector of sizes (including 0)")
        vect_sizes = cls.xp.concatenate(
            [cls.xp.asarray([0]), cls.xp.asarray(vect.values)]
        ).astype(cls.xp.uint32)
        logging.debug("Convert arr of labels to arr of sizes")
        res = vect_sizes[arr]
        # Returning
        return res

    @classmethod
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
    def filter_by_size(cls, arr: np.ndarray, smin=None, smax=None):
        """
        Assumes
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
    def get_local_maxima(cls, arr: np.ndarray, sigma=10):
        """
        NOTE: there can be multiple maxima per label
        """
        arr = cls.xp.asarray(arr)
        logging.debug("Making max filter for raw arr (holds the maximum in given area)")
        max_arr = cls.xdimage.maximum_filter(arr, sigma)
        logging.debug("Add 1 (so we separate the max pixel from the max_filter)")
        arr = arr + 1
        logging.debug("Getting local maxima (where arr - max_arr == 1)")
        res = arr - max_arr == 1
        # Returning
        return res.astype(cls.xp.uint8)

    @classmethod
    def mask(cls, arr: np.ndarray, arr_mask: np.ndarray):
        arr = cls.xp.asarray(arr)
        arr_mask = cls.xp.asarray(arr_mask).astype(cls.xp.uint8)
        logging.debug("Masking for only maxima within mask")
        res = arr * (arr_mask > 0)
        # Returning
        return res

    @classmethod
    def watershed_segm(
        cls, arr_raw: np.ndarray, arr_maxima: np.ndarray, arr_mask: np.ndarray
    ):
        """
        NOTE: NOT GPU accelerated

        Expects `arr_maxima` to have unique labels for each maxima.
        """
        logging.debug("Labelling maxima objects")
        arr_maxima, _ = cls.xdimage.label(arr_maxima)
        logging.debug("Padding everything with a 1 pixel empty border")
        arr_raw = cls.xp.pad(arr_raw, pad_width=1, mode="constant", constant_values=0)
        arr_maxima = cls.xp.pad(
            arr_maxima, pad_width=1, mode="constant", constant_values=0
        )
        arr_mask = cls.xp.pad(arr_mask, pad_width=1, mode="constant", constant_values=0)
        logging.debug("Watershed segmentation")
        res = watershed(
            image=-arr_raw,
            markers=arr_maxima,
            mask=arr_mask > 0,
        )
        logging.debug("Unpadding")
        res = res[1:-1, 1:-1, 1:-1].astype(np.uint32)
        # Returning
        return res

    @classmethod
    def region_to_coords(cls, arr: np.ndarray):
        """
        Get coordinates of regions in 3D tensor.

        Keeps only the first row (i.e cell) for each label.
        """
        logging.debug("Getting coordinates of regions")
        z, y, x = np.where(arr)
        logging.debug("Getting IDs of regions (from coords)")
        ids = arr[z, y, x]
        logging.debug("Making dataframe")
        df = pd.DataFrame(
            {
                "z": z.astype(np.uint16),
                "y": y.astype(np.uint16),
                "x": x.astype(np.uint16),
            },
            index=pd.Index(ids.astype(np.uint32), name="label"),
        )
        # Returning
        return df

    @classmethod
    def maxima_to_coords(cls, arr: np.ndarray):
        """
        Get coordinates of maxima in 3D tensor.

        Expects `arr_maxima` to have unique labels for each maxima.

        Keeps only the first row (i.e cell) for each label.
        """
        df = cls.region_to_coords(arr)
        logging.debug(
            "Keeping only first row per label (some maxima may be contiguous)"
        )
        df = df.groupby("label").first()
        # Returning
        return df
