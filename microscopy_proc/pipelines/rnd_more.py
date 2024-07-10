# %%
import logging
import os

import dask.array as da
import numpy as np
from dask.distributed import Client
from dask_cuda import LocalCUDACluster

from microscopy_proc.funcs.gpu_arr_funcs import GpuArrFuncs
from microscopy_proc.utils.dask_utils import disk_cache

# from dask_cuda import LocalCUDACluster

logging.basicConfig(level=logging.DEBUG)


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

    # cluster = LocalCluster(n_workers=1, threads_per_worker=1)
    cluster = LocalCUDACluster()
    client = Client(cluster)
    print(client.dashboard_link)

    # # Step 0: Read overlapped image
    arr_overlap = da.from_zarr(os.path.join(out_dir, "0_overlap.zarr"))

    # Step 1: Top-hat filter (background subtraction)
    arr_bgrm = arr_overlap.map_blocks(lambda i: GpuArrFuncs.tophat_filt(i, 5))
    arr_bgrm = disk_cache(arr_bgrm, os.path.join(out_dir, "1_bgrm.zarr"))

    # Step 2: Difference of Gaussians (edge detection)
    arr_dog = arr_bgrm.map_blocks(lambda i: GpuArrFuncs.dog_filt(i, 2, 4))
    arr_dog = disk_cache(arr_dog, os.path.join(out_dir, "2_dog.zarr"))

    # Step 3: Gaussian subtraction with large sigma for adaptive thresholding
    arr_adaptv = arr_dog.map_blocks(lambda i: GpuArrFuncs.gauss_subt_filt(i, 101))
    arr_adaptv = disk_cache(arr_adaptv, os.path.join(out_dir, "3_adaptv.zarr"))

    # Step 4: Mean thresholding with standard deviation offset
    # NOTE: visually inspect sd offset
    arr_adaptv_mean = (
        arr_adaptv.sum() / (np.prod(arr_adaptv.shape) - (arr_adaptv == 0).sum())
    ).compute()
    print(arr_adaptv_mean)
    t_p = 30
    arr_threshd = arr_adaptv.map_blocks(lambda i: GpuArrFuncs.manual_threshold(i, t_p))
    arr_threshd = disk_cache(arr_threshd, os.path.join(out_dir, "4_thresh.zarr"))

    # # Step 5: Label objects
    # arr_labels = label_objects(arr_threshd)
    # tifffile.imwrite("5_labels.tif", arr_labels)

    # # Step 6a: Get sizes of labelled objects (as df)
    # df_sizes = get_sizes(arr_labels)
    # df_sizes.to_parquet("6_sizes.parquet")
    # # Step 6b: Making sizes on arr (for checking)
    # arr_sizes = labels_map(arr_labels, df_sizes)
    # tifffile.imwrite("6_sizes.tif", arr_sizes)
    # # Step 6c: Visualise statistics (for checking)
    # visualise_stats(df_sizes)

    # # Step 7: Filter out large objects (likely outlines, not cells)
    # # TODO: Need to manually set min_size and max_size
    # arr_labels_filt = filter_large_objects(
    #     arr_labels, df_sizes, min_size=None, max_size=3000
    # )
    # tifffile.imwrite("7_labels_filt.tif", arr_labels_filt)

    # # Step 8: Get maxima of image masked by labels
    # arr_maxima = get_local_maxima(arr_raw, 10)
    # arr_maxima = mask(arr_maxima, arr_labels_filt)
    # tifffile.imwrite("8_maxima.tif", arr_maxima)

    # # Step 9: Making labels from maxima (i.e different ID for each maxima)
    # arr_maxima_labels = label_objects(arr_maxima)
    # tifffile.imwrite("9_maxima_labels.tif", arr_maxima_labels)

    # # Step 10: Watershed segmentation
    # arr_watershed = watershed_segm(arr_raw, arr_maxima_labels, arr_labels_filt)
    # tifffile.imwrite("10_watershed.tif", arr_watershed)

    # # Step 11: Get coords of maxima and get corresponding sizes from watershed
    # df_cells = region_to_coords_df(arr_watershed)
    # df_cells["size"] = get_sizes(arr_watershed)
    # df_cells.to_parquet("11_cells.parquet")
