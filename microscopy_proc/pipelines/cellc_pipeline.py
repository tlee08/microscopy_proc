import os

import dask.array as da
from dask.distributed import Client
from dask_cuda import LocalCUDACluster

from microscopy_proc.funcs.cellc_funcs import (
    dog_filter,
    gaussian_subtraction_filter,
    label_objects_with_ids,
    label_objects_with_sizes,
    mean_thresholding,
    region_to_coords,
    tophat_filter,
    visualise_stats,
)
from microscopy_proc.funcs.dask_funcs import (
    block_to_coords,
    get_maxima_block_from_region,
)

if __name__ == "__main__":
    # Filenames
    # in_fp = "/home/linux1/Desktop/A-1-1/abcd.tif"
    out_dir = "/home/linux1/Desktop/A-1-1/large_cellcount"

    # Defining process params
    depth = 50
    chunks = (500, 1000, 1000)

    # Making Dask cluster and client
    cluster = LocalCUDACluster()
    client = Client(cluster)
    print(client.dashboard_link)

    # Step 0: read image
    arr_raw = da.from_zarr(os.path.join(out_dir, "raw.zarr"), chunks=chunks)

    # Step 1: Top-hat filter (background subtraction)
    arr_bgsub = arr_raw.map_overlap(lambda i: tophat_filter(i, 5.0), depth=depth)

    # Step 2: Difference of Gaussians (edge detection)
    arr_dog = arr_bgsub.map_overlap(lambda i: dog_filter(i, 2.0, 4.0), depth=depth)

    # Step 3: Gaussian subtraction with large sigma
    # (adaptive thresholding - different from top-hat filter)
    arr_adaptv = arr_dog.map_overlap(
        lambda i: gaussian_subtraction_filter(i, 101), depth=depth
    )

    # Step 4: Mean thresholding with standard deviation offset
    # NOTE: visually inspect sd offset to use
    arr_threshd = arr_adaptv.map_overlap(
        lambda i: mean_thresholding(i, 0.0), depth=depth
    )

    # Step 5: Label objects
    arr_labels = label_objects_with_ids(arr_threshd)

    # Step 6: Object sizes
    arr_sizes = label_objects_with_sizes(arr_labels)
    # Step 6a: Visualising
    fig = visualise_stats(arr_sizes)

    # Making maxima arr
    arr_raw = da.from_zarr(raw_arr_fp)
    arr_region = da.from_zarr(region_arr_fp)
    arr_maxima = da.map_overlap(
        get_maxima_block_from_region, arr_raw, arr_region, depth=depth
    )

    exit()

    # Converting from region arr to coords df
    cell_coords = block_to_coords(region_to_coords, arr_region)
    cell_coords.to_parquet(region_df_fp)
    # Converting from maxima arr to coords df
    cell_coords = block_to_coords(region_to_coords, arr_maxima)
    cell_coords.to_parquet(maxima_df_fp)

    # Closing client
    client.close()
