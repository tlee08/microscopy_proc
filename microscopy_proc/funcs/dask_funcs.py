
from cellc_funcs import tophat_filter, dog_filter, gaussian_subtraction_filter, mean_thresholding, label_objects, get_sizes, labels_map, visualise_stats, filter_large_objects, get_local_maxima, mask, watershed_segm, region_to_coords_df, maxima_to_coords_df

def get_maxima_block(block):
    res = block
    res = tophat_filter(res, 10.0)
    res = dog_filter(res, 2.0, 5.0)
    res = gaussian_subtraction_filter(res, 101)
    res = mean_thresholding(res, 0.0)
    res = label_objects(res)
    df_sizes = get_sizes(res)
    # arr_sizes = labels_map(arr_labels, df_sizes)
    # visualise_stats(df_sizes)
    res = filter_large_objects(res, df_sizes, min_size=None, max_size=3000)
    res_maxima = get_local_maxima(block, 10)
    res_maxima = mask(res_maxima, res)
    res_maxima = label_objects(res_maxima)
    # res_watershed = watershed_segm(block, res_maxima, res)

    # Can choose either res, res_maxima, or res_watershed
    return res


def get_region_block(block):
    res = block
    res = tophat_filter(res, 10.0)
    res = dog_filter(res, 2.0, 5.0)
    res = gaussian_subtraction_filter(res, 101)
    res = mean_thresholding(res, 0.0)
    res = label_objects(res)
    df_sizes = get_sizes(res)
    # arr_sizes = labels_map(arr_labels, df_sizes)
    # visualise_stats(df_sizes)
    res = filter_large_objects(res, df_sizes, min_size=None, max_size=3000)

    # Can choose either res, res_maxima, or res_watershed
    return res


def get_maxima_block_from_region(block_raw, block_region):
    res = get_local_maxima(block_raw, 10)
    res = mask(res, block_region)
    res = label_objects(res)
    # res_watershed = watershed_segm(block, res_maxima, res)

    # Can choose either res, res_maxima, or res_watershed
    return res


def calc_inds(arr):
    """
    Calculate indices for each block in a Dask array.
    """
    return np.meshgrid(*[np.cumsum([0, *i[:-1]]) for i in arr.chunks], indexing="ij")


@dask.delayed
def get_region_df_block(block):
    """
    Expects block to be labelled maxima
    """
    df_cells = region_to_coords_df(block)
    return df_cells


@dask.delayed
def get_maxima_df_block(arr, z_offset, y_offset, x_offset):
    df = maxima_to_coords_df(arr)
    df["z"] += z_offset
    df["y"] += y_offset
    df["x"] += x_offset
    return df
