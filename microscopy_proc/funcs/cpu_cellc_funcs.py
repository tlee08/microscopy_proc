import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# from prefect import task
from scipy import ndimage as sc_ndimage
from skimage.segmentation import watershed

from microscopy_proc.constants import CELL_IDX_NAME, DEPTH, CellColumns, Coords


class CpuCellcFuncs:
    xp = np
    xdimage = sc_ndimage

    @classmethod
    # @task
    def tophat_filt(cls, ar: np.ndarray, sigma: int = 10) -> np.ndarray:
        """
        Top hat is calculated as:

        ```
        res = ar - max_filter(min_filter(ar, sigma), sigma)
        ```
        """
        ar = cls.xp.asarray(ar).astype(cls.xp.float32)
        logging.debug("Perform white top-hat filter")
        logging.debug(f"TYPE: {ar.dtype} {type(ar)}")
        res = cls.xdimage.white_tophat(ar, sigma)
        logging.debug("ReLu")
        res = cls.xp.maximum(res, 0)
        # Returning
        return res.astype(cls.xp.uint16)

    @classmethod
    # @task
    def dog_filt(cls, ar: np.ndarray, sigma1=1, sigma2=2) -> np.ndarray:
        ar = cls.xp.asarray(ar).astype(cls.xp.float32)
        logging.debug("Making gaussian blur 1")
        gaus1 = cls.xdimage.gaussian_filter(ar, sigma=sigma1)
        logging.debug("Making gaussian blur 2")
        gaus2 = cls.xdimage.gaussian_filter(ar, sigma=sigma2)
        logging.debug("Subtracting gaussian blurs")
        res = gaus1 - gaus2
        logging.debug("ReLu")
        res = cls.xp.maximum(res, 0)
        # Returning
        return res.astype(cls.xp.uint16)

    @classmethod
    # @task
    def gauss_blur_filt(cls, ar: np.ndarray, sigma=10) -> np.ndarray:
        ar = cls.xp.asarray(ar).astype(cls.xp.float32)
        logging.debug("Calculate Gaussian blur")
        res = cls.xdimage.gaussian_filter(ar, sigma=sigma)
        # Returning
        return res.astype(cls.xp.uint16)

    @classmethod
    # @task
    def gauss_subt_filt(cls, ar: np.ndarray, sigma=10) -> np.ndarray:
        ar = cls.xp.asarray(ar).astype(cls.xp.float32)
        logging.debug("Calculate local Gaussian blur")
        gaus = cls.xdimage.gaussian_filter(ar, sigma=sigma)
        logging.debug("Apply the adaptive filter")
        res = ar - gaus
        logging.debug("ReLu")
        res = cls.xp.maximum(res, 0)
        # Returning
        return res.astype(cls.xp.uint16)

    @classmethod
    # @task
    def intensity_cutoff(cls, ar: np.ndarray, min_=None, max_=None) -> np.ndarray:
        """
        Performing cutoffs on a 3D tensor.
        """
        ar = cls.xp.asarray(ar)
        logging.debug("Making cutoffs")
        res = ar
        if min_ is not None:
            res = cls.xp.maximum(res, min_)
        if max_ is not None:
            res = cls.xp.minimum(res, max_)
        # Returning
        return res

    @classmethod
    # @task
    def otsu_thresh(cls, ar: np.ndarray) -> np.ndarray:
        """
        Perform Otsu's thresholding on a 3D tensor.
        """
        ar = cls.xp.asarray(ar)
        logging.debug("Calculate histogram")
        hist, bin_edges = cls.xp.histogram(ar, bins=256)
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
        res = ar > optimal_threshold
        # Returning
        return res.astype(cls.xp.uint8)

    @classmethod
    # @task
    def mean_thresh(cls, ar: np.ndarray, offset_sd: float = 0.0) -> np.ndarray:
        """
        Perform adaptive thresholding on a 3D tensor on GPU.
        """
        ar = cls.xp.asarray(ar)
        logging.debug("Get mean and std of ONLY non-zero values")
        ar0 = ar[ar > 0]
        mu = ar0.mean()
        sd = ar0.std()
        logging.debug("Apply the threshold")
        res = ar > mu + offset_sd * sd
        # Returning
        return res.astype(cls.xp.uint8)

    @classmethod
    # @task
    def manual_thresh(cls, ar: np.ndarray, val: int):
        """
        Perform manual thresholding on a tensor.
        """
        ar = cls.xp.asarray(ar)
        logging.debug("Applying the threshold")
        res = ar >= val
        # Returning
        return res.astype(cls.xp.uint8)

    @classmethod
    # @task
    def label_with_ids(cls, ar: np.ndarray) -> np.ndarray:
        """
        Label objects in a 3D tensor.
        """
        ar = cls.xp.asarray(ar).astype(cls.xp.uint8)
        logging.debug("Labelling contiguous objects uniquely")
        res, _ = cls.xdimage.label(ar)
        logging.debug("Returning")
        return res.astype(cls.xp.uint32)

    @classmethod
    # @task
    def ids2sizes(cls, ar: np.ndarray) -> np.ndarray:
        """
        Convert labels to sizes.
        """
        ar = cls.xp.asarray(ar)
        logging.debug("Getting vector of ids and sizes (not incl. 0)")
        ids, counts = cls.xp.unique(ar[ar > 0], return_counts=True)
        # NOTE: assumes ids are incrementing from 1
        counts = cls.xp.concatenate([cls.xp.asarray([0]), counts])
        logging.debug("Converting arr intensity to sizes")
        res = counts[ar]
        logging.debug("Returning")
        return res.astype(cls.xp.uint16)

    @classmethod
    # @task
    def label_with_sizes(cls, ar: np.ndarray) -> np.ndarray:
        """
        Label objects in a 3D tensor.
        """
        ar = cls.label_with_ids(ar)
        res = cls.ids2sizes(ar)
        return res

    @classmethod
    # @task
    def visualise_stats(cls, ar: np.ndarray):
        """
        Visualise statistics.

        NOTE: expects arr to be a 3D tensor of a property
        (e.g. size).
        """
        logging.debug("Converting arr to vector of the ids")
        ids = ar[ar > 0]
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
    def filt_by_size(cls, ar: np.ndarray, smin=None, smax=None):
        """
        Assumes `arr` is array of objects labelled with their size.
        """
        ar = cls.xp.asarray(ar)
        logging.debug("Getting filter of small and large object to filter out")
        smin = smin if smin is not None else 0
        smax = smax if smax is not None else ar.max()
        filt_objs = (ar < smin) | (ar > smax)
        logging.debug("Filter out objects (by setting them to 0)")
        ar[filt_objs] = 0
        # Returning
        return ar

    @classmethod
    # @task
    def get_local_maxima(
        cls,
        ar: np.ndarray,
        sigma: int = 10,
        mask_ar: None | np.ndarray = None,
    ):
        """
        Getting local maxima (no connectivity) in a 3D tensor.
        If there is a connected region of maxima, then only the centre point is kept.

        If `mask_ar` is provided, then only maxima within the mask are kept.
        """
        ar = cls.xp.asarray(ar)
        logging.debug("Making max filter for raw ar (holds the maximum in given area)")
        max_ar = cls.xdimage.maximum_filter(ar, sigma)
        logging.debug("Add 1 (so we separate the max pixel from the max_filter)")
        ar = ar + 1
        logging.debug("Getting local maxima (where ar - max_arr == 1)")
        res = ar - max_ar == 1
        # If a mask is given, then keep only the maxima within the mask
        if mask_ar is not None:
            logging.debug(
                "Mask provided. Maxima will only be found within mask regions."
            )
            mask_ar = (cls.xp.asarray(mask_ar) > 0).astype(cls.xp.uint8)
            res = res * mask_ar
        # Returning
        return res

    @classmethod
    # @task
    def mask(cls, ar: np.ndarray, mask_ar: np.ndarray):
        ar = cls.xp.asarray(ar)
        mask_ar = cls.xp.asarray(mask_ar).astype(cls.xp.uint8)
        logging.debug("Masking for only maxima within mask")
        res = ar * (mask_ar > 0)
        # Returning
        return res

    @classmethod
    # @task
    def wshed_segm(cls, raw_ar: np.ndarray, maxima_ar: np.ndarray, mask_ar: np.ndarray):
        """
        NOTE: NOT GPU accelerated

        Expects `maxima_ar` to have unique labels for each maxima.
        """
        logging.debug("Watershed segmentation")
        res = watershed(
            image=-raw_ar,
            markers=maxima_ar,
            mask=mask_ar > 0,
        )
        # Returning
        return res

    @classmethod
    # @task
    def wshed_segm_sizes(
        cls, raw_ar: np.ndarray, maxima_ar: np.ndarray, mask_ar: np.ndarray
    ):
        """
        NOTE: NOT GPU accelerated
        """
        # Labelling contiguous maxima with unique labels
        maxima_ar = cls.label_with_ids(maxima_ar)
        # Watershed segmentation
        wshed_ar = cls.wshed_segm(raw_ar, maxima_ar, mask_ar)
        # Getting sizes of watershed regions
        res = cls.ids2sizes(wshed_ar)
        # Returning
        return res

    @classmethod
    # @task
    def get_coords(cls, ar: np.ndarray):
        """
        Get coordinates of regions in 3D tensor.

        TODO: Keep only the first row (i.e cell) for each label (groupby).
        """
        logging.debug("Getting coordinates of regions")
        z, y, x = np.where(ar)
        logging.debug("Getting IDs of regions (from coords)")
        ids = ar[z, y, x]
        logging.debug("Making dataframe")
        df = pd.DataFrame(
            {Coords.Z.value: z, Coords.Y.value: y, Coords.X.value: x},
            index=pd.Index(ids.astype(np.uint32), name=CELL_IDX_NAME),
        ).astype(np.uint16)
        df["size"] = -1  # TODO: placeholder
        df["sum_intensity"] = -1  # TODO: placeholder
        # df["max_intensity"] = -1  # TODO: placeholder
        # Returning
        return df

    @classmethod
    # @task
    def get_cells(
        cls,
        raw_ar: np.ndarray,
        overlap_ar: np.ndarray,
        maxima_ar: np.ndarray,
        mask_ar: np.ndarray,
        depth: int = DEPTH,
    ):
        """
        Get the cells from the maxima labels and the watershed segmentation
        (with corresponding labels).
        """
        # Asserting arr sizes match between raw_ar, overlap_ar, and depth
        assert raw_ar.shape == tuple(i - 2 * depth for i in overlap_ar.shape)
        logging.debug("Trimming maxima labels array to raw array dimensions using `d`")
        slicer = slice(depth, -depth) if depth > 0 else slice(None)
        maxima_ar = maxima_ar[slicer, slicer, slicer]
        logging.debug("Getting unique labels in maxima_ar")
        maxima_l_ar = cls.label_with_ids(maxima_ar)
        logging.debug("Converting to DataFrame of coordinates and sizes")
        # NOTE: getting first coord of each unique label
        ids_m, ind = cls.xp.unique(maxima_l_ar, return_index=True)
        z, y, x = cls.xp.unravel_index(ind, maxima_l_ar.shape)
        df = (
            pd.DataFrame(
                {
                    Coords.Z.value: cp2np(z),
                    Coords.Y.value: cp2np(y),
                    Coords.X.value: cp2np(x),
                },
                index=pd.Index(cp2np(ids_m).astype(np.uint32), name=CELL_IDX_NAME),
            )
            .drop(index=0)
            .astype(np.uint16)
        )
        logging.debug("Watershed of overlap_ar, seeds maxima_ar, mask mask_ar")
        # NOTE: padding maxima_l_ar because we previously trimmed maxima_ar
        maxima_l_ar = np.pad(
            cp2np(maxima_l_ar), depth, mode="constant", constant_values=0
        )
        wshed_ar = cls.wshed_segm(overlap_ar, maxima_l_ar, mask_ar)
        logging.debug("Making vector of region sizes (corresponding to maxima)")
        ids_w, counts = cls.xp.unique(wshed_ar[wshed_ar > 0], return_counts=True)
        ids_w = cp2np(ids_w).astype(np.uint32)
        counts = cp2np(counts).astype(np.uint32)
        logging.debug("Getting sum intensity for each cell (wshed)")
        sum_intensity = cls.xp.bincount(
            cls.xp.asarray(wshed_ar[wshed_ar > 0].ravel()),
            weights=cls.xp.asarray(overlap_ar[wshed_ar > 0].ravel()),
            minlength=len(ids_w),
        )
        # NOTE: excluding 0 valued elements means sum_intensity matches with ids_w
        sum_intensity = cp2np(sum_intensity[sum_intensity > 0])
        logging.debug("Adding sizes and intensities to DataFrame")
        idx = pd.Index(ids_w, name=CELL_IDX_NAME)
        df[CellColumns.COUNT.value] = 1
        df[CellColumns.VOLUME.value] = pd.Series(counts, index=idx)
        df[CellColumns.SUM_INTENSITY.value] = pd.Series(sum_intensity, index=idx)
        # df["max_intensity"] = pd.Series(max_intensity, index=idx)
        # Filtering out rows with NaNs in z, y, or x columns (i.e. no na values)
        df = df[
            df[[Coords.Z.value, Coords.Y.value, Coords.X.value]].isna().sum(axis=1) == 0
        ]
        # Returning
        return df


def cp2np(arr) -> np.ndarray:
    try:
        return arr.get()
    except Exception:
        return arr
