import dask.array as da
import numpy as np
from dask.distributed import LocalCluster
from dask_cuda import LocalCUDACluster
from prefect import flow

from microscopy_proc.constants import PROC_CHUNKS, S_DEPTH
from microscopy_proc.funcs.gpu_arr_funcs import GpuArrFuncs
from microscopy_proc.utils.dask_utils import (
    block_to_coords,
    cluster_proc_dec,
    disk_cache,
    my_trim,
)
from microscopy_proc.utils.proj_org_utils import get_proj_fp_dict, make_proj_dirs


@cluster_proc_dec(lambda: LocalCluster(n_workers=1, threads_per_worker=2))
@flow
def img_overlap_pipeline(proj_fp_dict):
    # Read raw arr
    arr_raw = da.from_zarr(proj_fp_dict["raw"], chunks=PROC_CHUNKS)
    # Make overlapping blocks
    arr_overlap = da.overlap.overlap(arr_raw, depth=S_DEPTH, boundary="reflect")
    arr_overlap = disk_cache(arr_overlap, proj_fp_dict["overlap"])


@cluster_proc_dec(lambda: LocalCUDACluster())
@flow
def img_proc_pipeline(
    proj_fp_dict,
    tophat_sigma=10,
    dog_sigma1=2,
    dog_sigma2=5,
    gauss_sigma=101,
    thresh_p=30,
    min_size=None,
    max_size=3000,
    maxima_sigma=10,
):
    # Step 0: Read overlapped image
    arr_overlap = da.from_zarr(proj_fp_dict["overlap"])

    # Step 1: Top-hat filter (background subtraction)
    arr_bgrm = arr_overlap.map_blocks(
        lambda i: GpuArrFuncs.tophat_filt(i, tophat_sigma)
    )
    arr_bgrm = disk_cache(arr_bgrm, proj_fp_dict["bgrm"])

    # Step 2: Difference of Gaussians (edge detection)
    arr_dog = arr_bgrm.map_blocks(
        lambda i: GpuArrFuncs.dog_filt(i, dog_sigma1, dog_sigma2)
    )
    arr_dog = disk_cache(arr_dog, proj_fp_dict["dog"])

    # Step 3: Gaussian subtraction with large sigma for adaptive thresholding
    arr_adaptv = arr_dog.map_blocks(
        lambda i: GpuArrFuncs.gauss_subt_filt(i, gauss_sigma)
    )
    arr_adaptv = disk_cache(arr_adaptv, proj_fp_dict["adaptv"])

    # Step 4: Mean thresholding with standard deviation offset
    # Visually inspect sd offset
    t_p = (
        arr_adaptv.sum() / (np.prod(arr_adaptv.shape) - (arr_adaptv == 0).sum())
    ).compute()
    print(t_p)
    arr_threshd = arr_adaptv.map_blocks(
        lambda i: GpuArrFuncs.manual_thresh(i, thresh_p)
    )
    arr_threshd = disk_cache(arr_threshd, proj_fp_dict["threshd"])

    # Step 5: Object sizes
    arr_sizes = arr_threshd.map_blocks(GpuArrFuncs.label_with_sizes)
    arr_sizes = disk_cache(arr_sizes, proj_fp_dict["sizes"])

    # Step 6: Filter out large objects (likely outlines, not cells)
    # TODO: Need to manually set min_size and max_size
    arr_filt = arr_sizes.map_blocks(
        lambda i: GpuArrFuncs.filt_by_size(i, min_size, max_size)
    )
    arr_filt = arr_filt.map_blocks(lambda i: GpuArrFuncs.manual_thresh(i, 1))
    arr_filt = disk_cache(arr_filt, proj_fp_dict["filt"])

    # Step 7: Get maxima of image masked by labels
    arr_maxima = arr_overlap.map_blocks(
        lambda i: GpuArrFuncs.get_local_maxima(i, maxima_sigma)
    )
    arr_maxima = da.map_blocks(GpuArrFuncs.mask, arr_maxima, arr_filt)
    arr_maxima = disk_cache(arr_maxima, proj_fp_dict["maxima"])

    # Step 8: Watershed segmentation
    # arr_watershed = da.map_blocks(watershed_segm, arr_overlap, arr_maxima, arr_filt)
    # arr_watershed = disk_cache(arr_watershed, proj_fp_dict["watershed"])


@cluster_proc_dec(lambda: LocalCluster())
@flow
def img_trim_pipeline(proj_fp_dict):
    # Read overlapped filtered and maxima images
    arr_filt = da.from_zarr(proj_fp_dict["filt"])
    arr_maxima = da.from_zarr(proj_fp_dict["maxima"])
    # Step 9a: trimming filtered regions overlaps
    arr_filt_f = my_trim(arr_filt)
    arr_filt_f = disk_cache(arr_filt_f, proj_fp_dict["filt_final"])
    # Step 9a: trimming maxima points overlaps
    arr_maxima_f = my_trim(arr_maxima)
    arr_maxima_f = disk_cache(arr_maxima_f, proj_fp_dict["maxima_final"])


@cluster_proc_dec(lambda: LocalCluster())
@flow
def img_to_coords_pipeline(proj_fp_dict):
    # Read filtered and maxima images (trimmed - orig space)
    arr_filt_f = da.from_zarr(proj_fp_dict["filt_final"])
    arr_maxima_f = da.from_zarr(proj_fp_dict["maxima_final"])
    # Step 10b: Get coords of maxima and get corresponding sizes from watershed
    cell_coords = block_to_coords(GpuArrFuncs.region_to_coords, arr_filt_f)
    cell_coords.to_parquet(proj_fp_dict["region_df"])
    # Step 10a: Get coords of maxima and get corresponding sizes from watershed
    cell_coords = block_to_coords(GpuArrFuncs.region_to_coords, arr_maxima_f)
    cell_coords.to_parquet(proj_fp_dict["maxima_df"])


if __name__ == "__main__":
    # Filenames
    proj_dir = "/home/linux1/Desktop/A-1-1/large_cellcount"

    proj_fp_dict = get_proj_fp_dict(proj_dir)
    make_proj_dirs(proj_dir)

    img_overlap_pipeline(proj_fp_dict)

    img_proc_pipeline(
        proj_fp_dict=proj_fp_dict,
        tophat_sigma=10,
        dog_sigma1=1,
        dog_sigma2=4,
        gauss_sigma=101,
        thresh_p=30,
        min_size=None,
        max_size=3000,
        maxima_sigma=10,
    )

    # img_trim_pipeline(proj_fp_dict)

    # img_to_coords_pipeline(proj_fp_dict)
