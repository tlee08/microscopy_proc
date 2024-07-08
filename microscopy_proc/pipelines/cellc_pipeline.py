import os

import dask
import dask.array as da
from dask.distributed import Client
from dask_cuda import LocalCUDACluster

from microscopy_proc.funcs.cellc_funcs import (
    dog_filter,
    filter_by_size,
    gaussian_subtraction_filter,
    get_local_maxima,
    label_objects_with_sizes,
    manual_threshold,
    mask,
    region_to_coords,
    tophat_filter,
)
from microscopy_proc.funcs.dask_funcs import block_to_coords, disk_cache

if __name__ == "__main__":
    # Filenames
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
    arr_bgsub = disk_cache(arr_bgsub, os.path.join(out_dir, "1_bgsub.zarr"))

    # Step 2: Difference of Gaussians (edge detection)
    arr_dog = arr_bgsub.map_overlap(lambda i: dog_filter(i, 2.0, 4.0), depth=depth)
    arr_dog = disk_cache(arr_dog, os.path.join(out_dir, "2_dog.zarr"))

    # Step 3: Gaussian subtraction with large sigma
    # (adaptive thresholding - different from top-hat filter)
    arr_adaptv = arr_dog.map_overlap(
        lambda i: gaussian_subtraction_filter(i, 101), depth=depth
    )
    arr_adaptv = disk_cache(arr_adaptv, os.path.join(out_dir, "3_adaptive_filt.zarr"))

    # Step 4: Mean thresholding with standard deviation offset
    # NOTE: visually inspect sd offset to use
    arr_adaptv0 = arr_adaptv[arr_adaptv > 0]
    (t_p,) = dask.compute(arr_adaptv0.mean() + 0.0 * arr_adaptv0.std())
    arr_threshd = arr_adaptv.map_blocks(lambda i: manual_threshold(i, t_p))
    arr_threshd = disk_cache(arr_threshd, os.path.join(out_dir, "4_thresh.zarr"))

    # Step 5: Object sizes
    arr_sizes = arr_threshd.map_overlap(
        lambda i: label_objects_with_sizes(i), depth=depth
    )
    arr_sizes = disk_cache(arr_sizes, os.path.join(out_dir, "5_sizes.zarr"))

    # Step 6: Filter out large objects (likely outlines, not cells)
    # TODO: Need to manually set min_size and max_size
    arr_filt = arr_sizes.map_blocks(lambda i: filter_by_size(i, smin=None, smax=3000))
    arr_filt = arr_filt.map_blocks(lambda i: manual_threshold(i, 1))
    arr_filt = disk_cache(arr_filt, os.path.join(out_dir, "6_filt.zarr"))

    # Step 7: Get maxima of image masked by labels
    arr_maxima = arr_raw.map_overlap(lambda i: get_local_maxima(i, 10), depth=depth)
    arr_maxima = da.map_blocks(mask, arr_maxima, arr_filt)
    arr_maxima = disk_cache(arr_maxima, os.path.join(out_dir, "7_maxima.zarr"))

    # Step 8: Watershed segmentation
    # arr_watershed = da.map_blocks(watershed_segm, arr_raw, arr_maxima, arr_filt)
    # arr_watershed = disk_cache(arr_watershed, os.path.join(out_dir, "8_watershed.zarr"))

    # Step 9b: Get coords of maxima and get corresponding sizes from watershed
    cell_coords = block_to_coords(region_to_coords, arr_filt)
    cell_coords.to_parquet(os.path.join(out_dir, "9_regions.parquet"))
    # Step 9a: Get coords of maxima and get corresponding sizes from watershed
    cell_coords = block_to_coords(region_to_coords, arr_maxima)
    cell_coords.to_parquet(os.path.join(out_dir, "9_maximas.parquet"))

    # Closing client
    client.close()
