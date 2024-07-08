import logging

import numpy as np
import tifffile

from microscopy_proc.funcs.cellc_funcs import (
    dog_filter,
    filter_large_objects,
    gaussian_subtraction_filter,
    get_local_maxima,
    get_sizes,
    label_objects,
    labels_map,
    mask,
    mean_thresholding,
    region_to_coords,
    tophat_filter,
    visualise_stats,
    watershed_segm,
)

logging.basicConfig(level=logging.DEBUG)

# Pipeline

# Step 0: read image
arr_raw = tifffile.imread("cropped abcd_larger.tif").astype(np.uint16)

# Step 1: Top-hat filter (background subtraction)
arr_bgsub = tophat_filter(arr_raw, 10.0)
tifffile.imwrite("1_bgsub.tif", arr_bgsub)

# Step 2: Difference of Gaussians (edge detection)
arr_dog = dog_filter(arr_bgsub, 2.0, 5.0)
tifffile.imwrite("2_dog.tif", arr_dog)

# Step 3: Gaussian subtraction with large sigma
# (adaptive thresholding - different from top-hat filter)
arr_adaptv = gaussian_subtraction_filter(arr_dog, 101)
tifffile.imwrite("3_adaptive_filt.tif", arr_adaptv)

# Step 4: Mean thresholding with standard deviation offset
# NOTE: visually inspect sd offset to use
arr_threshd = mean_thresholding(arr_adaptv, 0.0)
tifffile.imwrite("4_thresh.tif", arr_threshd)

# Step 5: Label objects
arr_labels = label_objects(arr_threshd)
tifffile.imwrite("5_labels.tif", arr_labels)

# Step 6a: Get sizes of labelled objects (as df)
df_sizes = get_sizes(arr_labels)
df_sizes.to_parquet("6_sizes.parquet")
# Step 6b: Making sizes on arr (for checking)
arr_sizes = labels_map(arr_labels, df_sizes)
tifffile.imwrite("6_sizes.tif", arr_sizes)
# Step 6c: Visualise statistics (for checking)
visualise_stats(df_sizes)

# Step 7: Filter out large objects (likely outlines, not cells)
# TODO: Need to manually set min_size and max_size
arr_labels_filt = filter_large_objects(
    arr_labels, df_sizes, min_size=None, max_size=3000
)
tifffile.imwrite("7_labels_filt.tif", arr_labels_filt)

# Step 8: Get maxima of image masked by labels
arr_maxima = get_local_maxima(arr_raw, 10)
arr_maxima = mask(arr_maxima, arr_labels_filt)
tifffile.imwrite("8_maxima.tif", arr_maxima)

# Step 9: Making labels from maxima (i.e different ID for each maxima)
arr_maxima_labels = label_objects(arr_maxima)
tifffile.imwrite("9_maxima_labels.tif", arr_maxima_labels)

# Step 10: Watershed segmentation
arr_watershed = watershed_segm(arr_raw, arr_maxima_labels, arr_labels_filt)
tifffile.imwrite("10_watershed.tif", arr_watershed)

# Step 11: Get coords of maxima and get corresponding sizes from watershed
df_cells = region_to_coords_df(arr_watershed)
df_cells["size"] = get_sizes(arr_watershed)
df_cells.to_parquet("11_cells.parquet")
