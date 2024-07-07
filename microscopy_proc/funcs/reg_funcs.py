
import logging

import cupy as cp
import dask.delayed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tifffile
from cupyx.scipy.ndimage import gaussian_filter as cupy_gaussian_filter
from cupyx.scipy.ndimage import label as cupy_label
from cupyx.scipy.ndimage import maximum_filter as cupy_maximum_filter
from cupyx.scipy.ndimage import white_tophat as cupy_white_tophat
from skimage.segmentation import watershed

from microscopy_proc.utils.cp_utils import clear_cuda_memory_decorator, numpy_2_cupy_decorator


def zoom_arr(arr: np.ndarray, z_slice, y_slice, x_slice) -> np.ndarray:
    res = arr[::z_slice, ::y_slice, ::x_slice]
    return res

