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
    def gauss_blur_filt(cls, arr: np.ndarray, sigma=10) -> np.ndarray:
        arr = cls.xp.asarray(arr).astype(cls.xp.float32)
        logging.debug("Calculate Gaussian blur")
        res = cls.xdimage.gaussian_filter(arr, sigma=sigma)
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
    def mask2ids(cls, arr: np.ndarray) -> np.ndarray:
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
    def ids2volumes(cls, arr: np.ndarray) -> np.ndarray:
        """
        Convert array of label values to
        contiguous volume (i.e. count) values.
        """
        arr = cls.xp.asarray(arr)
        logging.debug("Getting vector of ids and volumes (not incl. 0)")
        ids, counts = cls.xp.unique(arr[arr > 0], return_counts=True)
        # NOTE: assumes ids are incrementing from 1
        counts = cls.xp.concatenate([cls.xp.asarray([0]), counts])
        logging.debug("Converting arr intensity to volumes")
        res = counts[arr]
        logging.debug("Returning")
        return res.astype(cls.xp.uint16)

    @classmethod
    # @task
    def label_with_volumes(cls, arr: np.ndarray) -> np.ndarray:
        """
        Label objects in a 3D tensor.
        """
        arr = cls.mask2ids(arr)
        res = cls.ids2volumes(arr)
        return res

    @classmethod
    # @task
    def visualise_stats(cls, arr: np.ndarray):
        """
        Visualise statistics.

        NOTE: expects arr to be a 3D tensor of a property
        (e.g. volume).
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
    def volume_filter(cls, arr: np.ndarray, smin=None, smax=None):
        """
        Assumes `arr` is array of objects labelled with their volumes.
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
        mask_arr: None | np.ndarray = None,
    ):
        """
        Getting local maxima (no connectivity) in a 3D tensor.
        If there is a connected region of maxima, then only the centre point is kept.

        If `mask_arr` is provided, then only maxima within the mask are kept.
        """
        arr = cls.xp.asarray(arr)
        logging.debug("Making max filter for raw arr (holds the maximum in given area)")
        max_arr = cls.xdimage.maximum_filter(arr, sigma)
        logging.debug("Add 1 (so we separate the max pixel from the max_filter)")
        arr = arr + 1
        logging.debug("Getting local maxima (where arr - max_arr == 1)")
        res = arr - max_arr == 1
        # If a mask is given, then keep only the maxima within the mask
        if mask_arr is not None:
            logging.debug(
                "Mask provided. Maxima will only be found within mask regions."
            )
            mask_arr = (cls.xp.asarray(mask_arr) > 0).astype(cls.xp.uint8)
            res = res * mask_arr
        # Returning
        return res

    @classmethod
    # @task
    def mask(cls, arr: np.ndarray, mask_arr: np.ndarray):
        arr = cls.xp.asarray(arr)
        mask_arr = cls.xp.asarray(mask_arr).astype(cls.xp.uint8)
        logging.debug("Masking for only maxima within mask")
        res = arr * (mask_arr > 0)
        # Returning
        return res

    @classmethod
    # @task
    def wshed_segm(
        cls, raw_arr: np.ndarray, maxima_arr: np.ndarray, mask_arr: np.ndarray
    ):
        """
        NOTE: NOT GPU accelerated

        Expects `maxima_arr` to have unique labels for each maxima.
        """
        logging.debug("Watershed segmentation")
        res = watershed(
            image=-raw_arr,
            markers=maxima_arr,
            mask=mask_arr > 0,
        )
        # Returning
        return res

    @classmethod
    # @task
    def wshed_segm_volumes(
        cls, raw_arr: np.ndarray, maxima_arr: np.ndarray, mask_arr: np.ndarray
    ):
        """
        NOTE: NOT GPU accelerated
        """
        # Labelling contiguous maxima with unique labels
        maxima_arr = cls.mask2ids(maxima_arr)
        # Watershed segmentation
        wshed_arr = cls.wshed_segm(raw_arr, maxima_arr, mask_arr)
        # Getting volumes of watershed regions
        res = cls.ids2volumes(wshed_arr)
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
            {
                Coords.Z.value: z,
                Coords.Y.value: y,
                Coords.X.value: x,
            },
            index=pd.Index(ids.astype(np.uint32), name=CELL_IDX_NAME),
        ).astype(np.uint16)
        df[CellColumns.VOLUME.value] = -1  # TODO: placeholder
        df[CellColumns.SUM_INTENSITY.value] = -1  # TODO: placeholder
        # df[CellColumns.MAX_INTENSITY.value] = -1  # TODO: placeholder
        # Returning
        return df

    @classmethod
    # @task
    def get_cells(
        cls,
        raw_arr: np.ndarray,
        overlap_arr: np.ndarray,
        maxima_arr: np.ndarray,
        mask_arr: np.ndarray,
        depth: int = DEPTH,
    ):
        """
        Get the cells from the maxima labels and the watershed segmentation
        (with corresponding labels).
        """
        # Asserting arr sizes match between arr_raw, arr_overlap, and depth
        # NOTE: we NEED raw_arr as the first da.Array to get chunking coord offsets correct
        assert raw_arr.shape == tuple(i - 2 * depth for i in overlap_arr.shape)
        logging.debug("Trimming maxima labels array to raw array dimensions using `d`")
        slicer = slice(depth, -depth) if depth > 0 else slice(None)
        maxima_arr = maxima_arr[slicer, slicer, slicer]
        logging.debug("Getting unique labels in maxima_arr")
        maxima_l_arr = cls.mask2ids(maxima_arr)
        logging.debug("Converting to DataFrame of coordinates and measures")
        # NOTE: getting first coord of each unique label
        # NOTE: np.unique auto flattens arr so reshaping it back with np.unravel_index
        ids_m, ind = cls.xp.unique(maxima_l_arr, return_index=True)
        z, y, x = cls.xp.unravel_index(ind, maxima_l_arr.shape)
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
        logging.debug("Watershed of overlap_arr, seeds maxima_arr, mask mask_arr")
        # NOTE: padding maxima_l_arr because we previously trimmed maxima_arr
        maxima_l_arr = np.pad(
            cp2np(maxima_l_arr), depth, mode="constant", constant_values=0
        )
        wshed_arr = cls.wshed_segm(overlap_arr, maxima_l_arr, mask_arr)
        logging.debug("Making vector of watershed region volumes")
        ids_w, counts = cls.xp.unique(wshed_arr[wshed_arr > 0], return_counts=True)
        ids_w = cp2np(ids_w).astype(np.uint32)
        counts = cp2np(counts).astype(np.uint32)
        logging.debug("Getting sum intensity for each cell (wshed)")
        sum_intensity = cls.xp.bincount(
            cls.xp.asarray(wshed_arr[wshed_arr > 0].ravel()),
            weights=cls.xp.asarray(overlap_arr[wshed_arr > 0].ravel()),
            minlength=len(ids_w),
        )
        # NOTE: excluding 0 valued elements means sum_intensity matches with ids_w
        sum_intensity = cp2np(sum_intensity[sum_intensity > 0])
        logging.debug("Adding cell measures to DataFrame")
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
