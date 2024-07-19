import logging

import numpy as np
import tifffile

from microscopy_proc.funcs.gpu_arr_funcs import GpuArrFuncs

logging.basicConfig(level=logging.DEBUG)

# Pipeline

# Step 0: read image
arr_raw = tifffile.imread("cropped abcd_larger.tif").astype(np.uint16)

# Step 1: Top-hat filter (background subtraction)
arr_bgsub = GpuArrFuncs.tophat_filt(arr_raw, 10.0)
tifffile.imwrite("1_bgsub.tif", arr_bgsub)

# Step 2: Difference of Gaussians (edge detection)
arr_dog = GpuArrFuncs.dog_filt(arr_bgsub, 2.0, 5.0)
tifffile.imwrite("2_dog.tif", arr_dog)

# Step 3: Gaussian subtraction with large sigma
# (adaptive thresholding - different from top-hat filter)
arr_adaptv = GpuArrFuncs.gauss_subt_filt(arr_dog, 101)
tifffile.imwrite("3_adaptive_filt.tif", arr_adaptv)

# Step 4: Mean thresholding with standard deviation offset
# NOTE: visually inspect sd offset to use
arr_threshd = GpuArrFuncs.manual_thresh(arr_adaptv, 30.0)
tifffile.imwrite("4_thresh.tif", arr_threshd)

# Step 5: Label objects
arr_labels = GpuArrFuncs.label_with_ids(arr_threshd)
tifffile.imwrite("5_labels.tif", arr_labels)

# Step 6a: Get sizes of labelled objects (as df)
arr_sizes = GpuArrFuncs.label_with_sizes(arr_threshd)
tifffile.imwrite("6_sizes.tif", arr_sizes)
# Step 6b: Making sizes on arr (for checking)
df_sizes = GpuArrFuncs.get_sizes(arr_labels)
df_sizes.to_parquet("6_sizes.parquet")
# Step 6c: Visualise statistics (for checking)
GpuArrFuncs.visualise_stats(df_sizes)

# Step 7: Filter out large objects (likely outlines, not cells)
# TODO: Need to manually set min_size and max_size
arr_labels_filt = GpuArrFuncs.filt_by_size(arr_sizes, min_size=None, max_size=3000)
tifffile.imwrite("7_labels_filt.tif", arr_labels_filt)

# Step 8: Get maxima of image masked by labels
arr_maxima = GpuArrFuncs.get_local_maxima(arr_raw, 10)
arr_maxima = GpuArrFuncs.mask(arr_maxima, arr_labels_filt)
tifffile.imwrite("8_maxima.tif", arr_maxima)

# Step 9: Making labels from maxima (i.e different ID for each maxima)
arr_watershed = GpuArrFuncs.watershed_segm(arr_raw, arr_maxima, arr_labels_filt)
tifffile.imwrite("10_watershed.tif", arr_watershed)

# Step 11: Get coords of maxima and get corresponding sizes from watershed
df_cells = GpuArrFuncs.region_to_coords(arr_watershed)
df_cells.to_parquet("11_cells.parquet")
