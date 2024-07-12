import os

import dask.array as da
import numpy as np
from dask.distributed import Client, LocalCluster
from dask_cuda import LocalCUDACluster

from microscopy_proc.constants import PROC_CHUNKS, S_DEPTH
from microscopy_proc.funcs.gpu_arr_funcs import GpuArrFuncs
from microscopy_proc.utils.dask_utils import block_to_coords, disk_cache, my_trim, cluster_proc_dec
from microscopy_proc.funcs.io_funcs import tiffs_to_zarr, btiff_to_zarr


@cluster_proc_dec(lambda: LocalCluster())
def tiff_to_zarr(in_fp, out_dir):
    if os.path.isdir(in_fp):
        tiffs_to_zarr(
            [os.path.join(in_fp, f) for f in os.listdir(in_fp)],
            os.path.join(out_dir, "raw.zarr"),
            chunks=PROC_CHUNKS,
        )
    elif os.path.isfile(in_fp):
        btiff_to_zarr(
            in_fp,
            os.path.join(out_dir, "raw.zarr"),
            chunks=PROC_CHUNKS,
        )
    else:
        raise ValueError("Input file path does not exist.")

@cluster_proc_dec(lambda: LocalCluster(n_workers=1, threads_per_worker=2))
def img_overlap_pipeline(out_dir):
    # Read raw arr
    arr_raw = da.from_zarr(os.path.join(out_dir, "raw.zarr"), chunks=PROC_CHUNKS)

    # Make overlapping blocks
    arr_overlap = da.overlap.overlap(arr_raw, depth=S_DEPTH, boundary="reflect")
    arr_overlap = disk_cache(arr_overlap, os.path.join(out_dir, "0_overlap.zarr"))


@cluster_proc_dec(lambda: LocalCUDACluster())
def img_proc_pipeline(out_dir):
    # Step 0: Read overlapped image
    arr_overlap = da.from_zarr(os.path.join(out_dir, "0_overlap.zarr"))

    # Step 1: Top-hat filter (background subtraction)
    arr_bgrm = arr_overlap.map_blocks(lambda i: GpuArrFuncs.tophat_filt(i, 10))
    arr_bgrm = disk_cache(arr_bgrm, os.path.join(out_dir, "1_bgrm.zarr"))

    # # Step 2: Difference of Gaussians (edge detection)
    arr_dog = arr_bgrm.map_blocks(lambda i: GpuArrFuncs.dog_filt(i, 2, 5))
    arr_dog = disk_cache(arr_dog, os.path.join(out_dir, "2_dog.zarr"))

    # Step 3: Gaussian subtraction with large sigma for adaptive thresholding
    arr_adaptv = arr_dog.map_blocks(lambda i: GpuArrFuncs.gauss_subt_filt(i, 101))
    arr_adaptv = disk_cache(arr_adaptv, os.path.join(out_dir, "3_adaptv.zarr"))

    # Step 4: Mean thresholding with standard deviation offset
    # NOTE: visually inspect sd offset
    t_p = (
        arr_adaptv.sum() / (np.prod(arr_adaptv.shape) - (arr_adaptv == 0).sum())
    ).compute()
    print(t_p)
    t_p = 30
    arr_threshd = arr_adaptv.map_blocks(lambda i: GpuArrFuncs.manual_thresh(i, t_p))
    arr_threshd = disk_cache(arr_threshd, os.path.join(out_dir, "4_threshd.zarr"))

    # Step 5: Object sizes
    arr_sizes = arr_threshd.map_blocks(GpuArrFuncs.label_with_sizes)
    arr_sizes = disk_cache(arr_sizes, os.path.join(out_dir, "5_sizes.zarr"))

    # Step 6: Filter out large objects (likely outlines, not cells)
    # TODO: Need to manually set min_size and max_size
    arr_filt = arr_sizes.map_blocks(lambda i: GpuArrFuncs.filt_by_size(i, None, 3000))
    arr_filt = arr_filt.map_blocks(lambda i: GpuArrFuncs.manual_thresh(i, 1))
    arr_filt = disk_cache(arr_filt, os.path.join(out_dir, "6_filt.zarr"))

    # Step 7: Get maxima of image masked by labels
    arr_maxima = arr_overlap.map_blocks(lambda i: GpuArrFuncs.get_local_maxima(i, 10))
    arr_maxima = da.map_blocks(GpuArrFuncs.mask, arr_maxima, arr_filt)
    arr_maxima = disk_cache(arr_maxima, os.path.join(out_dir, "7_maxima.zarr"))

    # Step 8: Watershed segmentation
    # arr_watershed = da.map_blocks(watershed_segm, arr_overlap, arr_maxima, arr_filt)
    # arr_watershed = disk_cache(arr_watershed, os.path.join(out_dir, "8_watershed.zarr"))


@cluster_proc_dec(lambda: LocalCluster())
def img_trim_pipeline(out_dir):
    arr_filt = da.from_zarr(os.path.join(out_dir, "6_filt.zarr"))
    arr_maxima = da.from_zarr(os.path.join(out_dir, "7_maxima.zarr"))

    # Step 9a: trimming overlaps
    arr_filt_f = my_trim(arr_filt)
    arr_filt_f = disk_cache(arr_filt_f, os.path.join(out_dir, "9_filt_f.zarr"))

    # Step 9a: trimming overlaps
    arr_maxima_f = my_trim(arr_maxima)
    arr_maxima_f = disk_cache(arr_maxima_f, os.path.join(out_dir, "9_maxima_f.zarr"))


@cluster_proc_dec(lambda: LocalCluster())
def img_to_coords_pipeline(out_dir):
    arr_filt_f = da.from_zarr(os.path.join(out_dir, "9_filt_f.zarr"))
    arr_maxima_f = da.from_zarr(os.path.join(out_dir, "9_maxima_f.zarr"))

    # Step 10b: Get coords of maxima and get corresponding sizes from watershed
    cell_coords = block_to_coords(GpuArrFuncs.region_to_coords, arr_filt_f)
    cell_coords.to_parquet(os.path.join(out_dir, "10_region.parquet"))
    # Step 10a: Get coords of maxima and get corresponding sizes from watershed
    cell_coords = block_to_coords(GpuArrFuncs.region_to_coords, arr_maxima_f)
    cell_coords.to_parquet(os.path.join(out_dir, "10_maxima.parquet"))


if __name__ == "__main__":
    # Filenames
    in_fp = "/home/linux1/Desktop/A-1-1/example"
    # in_fp = "/home/linux1/Desktop/A-1-1/cropped abcd_larger.tif"
    out_dir = "/home/linux1/Desktop/A-1-1/large_cellcount"

    os.makedirs(out_dir, exist_ok=True)

    tiff_to_zarr(in_fp, out_dir)

    img_overlap_pipeline(out_dir)

    img_proc_pipeline(out_dir)

    img_trim_pipeline(out_dir)

    img_to_coords_pipeline(out_dir)
