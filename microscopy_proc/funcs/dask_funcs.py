import numpy as np

from microscopy_proc.funcs.cellc_funcs import (
    dog_filter,
    filter_large_objects,
    gaussian_subtraction_filter,
    get_local_maxima,
    get_sizes,
    label_objects,
    mask,
    maxima_to_coords_df,
    mean_thresholding,
    region_to_coords_df,
    tophat_filter,
)


def get_maxima_block(block):
    res = block
    res = tophat_filter(res, 10.0)
    res = dog_filter(res, 2.0, 5.0)
    res = gaussian_subtraction_filter(res, 101)
    res = mean_thresholding(res, 0.0)
    res = label_objects(res)
    df_sizes = get_sizes(res)
    res = filter_large_objects(res, df_sizes, min_size=None, max_size=3000)
    res_maxima = get_local_maxima(block, 10)
    res_maxima = mask(res_maxima, res)
    res_maxima = label_objects(res_maxima)
    # res_watershed = watershed_segm(block, res_maxima, res)

    return res


def get_region_block(block):
    res = block
    res = tophat_filter(res, 10.0)
    res = dog_filter(res, 2.0, 5.0)
    res = gaussian_subtraction_filter(res, 101)
    res = mean_thresholding(res, 0.0)
    res = label_objects(res)
    df_sizes = get_sizes(res)
    res = filter_large_objects(res, df_sizes, min_size=None, max_size=3000)

    return res


def get_maxima_block_from_region(block_raw, block_region):
    res = get_local_maxima(block_raw, 10)
    res = mask(res, block_region)
    res = label_objects(res)
    # res_watershed = watershed_segm(block, res_maxima, res)

    return res


def calc_inds(arr):
    """
    Calculate indices for each block in a Dask array.
    """
    return np.meshgrid(*[np.cumsum([0, *i[:-1]]) for i in arr.chunks], indexing="ij")


def get_region_df_block(block):
    """
    Expects block to be labelled maxima
    """
    df_cells = region_to_coords_df(block)
    return df_cells


def get_maxima_df_block(arr, z_offset, y_offset, x_offset):
    df = maxima_to_coords_df(arr)
    df["z"] += z_offset
    df["y"] += y_offset
    df["x"] += x_offset
    return df
