import os

import dask.array as da
import numpy as np
from dask.distributed import Client, LocalCluster

from microscopy_proc.funcs.cellc_funcs import (
    dog_filter,
    filter_by_size,
    gaussian_subtraction_filter,
    get_local_maxima,
    label_objects_with_sizes,
    manual_threshold,
    # tophat_filter,
    mask,
    region_to_coords,
)
from microscopy_proc.utils.dask_utils import block_to_coords, disk_cache, my_trim

if __name__ == "__main__":
    # Filenames
    in_fp = "/home/linux1/Desktop/A-1-1/abcd.tif"
    out_dir = "/home/linux1/Desktop/A-1-1/large_cellcount"

    #########################
    # TIFF TO ZARR
    #########################

    # tiff_to_zarr(in_fp, os.path.join(out_dir, "raw.zarr"), chunks=PROC_CHUNKS)

    # #########################
    # # OVERLAP
    # #########################

    # # Making Dask cluster and client
    # cluster = LocalCluster(n_workers=6, threads_per_worker=4)
    # client = Client(cluster)
    # print(client.dashboard_link)

    # # Read raw arr
    # arr_raw = da.from_zarr(os.path.join(out_dir, "raw.zarr"), chunks=PROC_CHUNKS)

    # # Make overlapping blocks
    # arr_overlap = da.overlap.overlap(arr_raw, depth=S_DEPTH, boundary="reflect")
    # arr_overlap = disk_cache(arr_overlap, os.path.join(out_dir, "0_overlap.zarr"))

    # # Closing client
    # client.close()
    # cluster.close()

    #########################
    # HEAVY GPU PROCESSING
    #########################

    # Making Dask cluster and client
    # cluster = LocalCUDACluster()
    cluster = LocalCluster(processes=False, threads_per_worker=1)
    client = Client(cluster)
    print(client.dashboard_link)

    # # Step 0: Read overlapped image
    # arr_overlap = da.from_zarr(os.path.join(out_dir, "0_overlap.zarr"))
    arr_bgrm = da.from_zarr(os.path.join(out_dir, "1_bgrm.zarr"))

    # # # Step 1: Top-hat filter (background subtraction)
    # arr_bgrm = arr_overlap.map_blocks(lambda i: tophat_filter(i, 5.0))
    # arr_bgrm = disk_cache(arr_bgrm, os.path.join(out_dir, "1_bgrm.zarr"))

    # # Step 2: Difference of Gaussians (edge detection)
    arr_dog = arr_bgrm.map_blocks(lambda i: dog_filter(i, 2.0, 4.0))
    arr_dog = disk_cache(arr_dog, os.path.join(out_dir, "2_dog.zarr"))

    # Step 3: Gaussian subtraction with large sigma for adaptive thresholding
    arr_adaptv = arr_dog.map_blocks(lambda i: gaussian_subtraction_filter(i, 101))
    arr_adaptv = disk_cache(arr_adaptv, os.path.join(out_dir, "3_adaptv.zarr"))

    # Step 4: Mean thresholding with standard deviation offset
    # NOTE: visually inspect sd offset
    arr_adaptv_mean = (
        arr_adaptv.sum() / (np.prod(arr_adaptv.shape) - (arr_adaptv == 0).sum())
    ).compute()
    print(arr_adaptv_mean)
    t_p = 30
    arr_threshd = arr_adaptv.map_blocks(lambda i: manual_threshold(i, t_p))
    arr_threshd = disk_cache(arr_threshd, os.path.join(out_dir, "4_thresh.zarr"))

    # Step 5: Object sizes
    arr_sizes = arr_threshd.map_blocks(label_objects_with_sizes)
    arr_sizes = disk_cache(arr_sizes, os.path.join(out_dir, "5_sizes.zarr"))

    # Step 6: Filter out large objects (likely outlines, not cells)
    # TODO: Need to manually set min_size and max_size
    arr_filt = arr_sizes.map_blocks(lambda i: filter_by_size(i, smin=None, smax=3000))
    arr_filt = arr_filt.map_blocks(lambda i: manual_threshold(i, 1))
    arr_filt = disk_cache(arr_filt, os.path.join(out_dir, "6_filt.zarr"))

    # Step 7: Get maxima of image masked by labels
    arr_maxima = arr_overlap.map_blocks(lambda i: get_local_maxima(i, 10))
    arr_maxima = da.map_blocks(mask, arr_maxima, arr_filt)
    arr_maxima = disk_cache(arr_maxima, os.path.join(out_dir, "7_maxima.zarr"))

    # Step 8: Watershed segmentation
    # arr_watershed = da.map_blocks(watershed_segm, arr_overlap, arr_maxima, arr_filt)
    # arr_watershed = disk_cache(arr_watershed, os.path.join(out_dir, "8_watershed.zarr"))

    # Closing client
    client.close()
    cluster.close()

    #########################
    # TRIMMING OVERLAPS
    #########################

    # Making Dask cluster and client
    cluster = LocalCluster(processes=False)
    client = Client(cluster)
    print(client.dashboard_link)

    arr_raw = da.from_zarr(os.path.join(out_dir, "raw.zarr"))
    arr_filt = da.from_zarr(os.path.join(out_dir, "6_filt.zarr"))
    arr_maxima = da.from_zarr(os.path.join(out_dir, "7_maxima.zarr"))

    # Step 9a: trimming overlaps
    arr_filt_f = arr_filt.map_blocks(my_trim, chunks=arr_raw.chunks)
    # arr_filt_f = da.overlap.trim_overlap(arr_filt, depth=S_DEPTH)
    arr_filt_f = disk_cache(arr_filt_f, os.path.join(out_dir, "9_filt_f.zarr"))

    # Step 9a: trimming overlaps
    arr_maxima_f = arr_maxima.map_blocks(my_trim, chunks=arr_raw.chunks)
    # arr_maxima_f = da.overlap.trim_overlap(arr_maxima, depth=S_DEPTH)
    arr_maxima_f = disk_cache(arr_maxima_f, os.path.join(out_dir, "9_maxima_f.zarr"))

    # Closing client
    client.close()
    cluster.close()

    #########################
    # ARR TO COORDS
    #########################

    # Making Dask cluster and client
    cluster = LocalCluster(n_workers=8, threads_per_worker=1)
    client = Client(cluster)
    print(client.dashboard_link)

    # Step 10b: Get coords of maxima and get corresponding sizes from watershed
    cell_coords = block_to_coords(region_to_coords, arr_filt_f)
    cell_coords.to_parquet(os.path.join(out_dir, "10_regions.parquet"))
    # Step 10a: Get coords of maxima and get corresponding sizes from watershed
    cell_coords = block_to_coords(region_to_coords, arr_maxima_f)
    cell_coords.to_parquet(os.path.join(out_dir, "10_maximas.parquet"))

    # Closing client
    client.close()
    cluster.close()
