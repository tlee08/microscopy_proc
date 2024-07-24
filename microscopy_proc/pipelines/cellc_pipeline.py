import logging

import dask.array as da
import numpy as np
from dask.distributed import LocalCluster
from dask_cuda import LocalCUDACluster

# from prefect import flow
from microscopy_proc.constants import PROC_CHUNKS, S_DEPTH
from microscopy_proc.funcs.cpu_arr_funcs import CpuArrFuncs
from microscopy_proc.funcs.gpu_arr_funcs import GpuArrFuncs
from microscopy_proc.utils.dask_utils import (
    block_to_coords,
    cluster_proc_dec,
    disk_cache,
    my_trim,
)
from microscopy_proc.utils.proj_org_utils import get_proj_fp_dict, make_proj_dirs

logging.basicConfig(level=logging.DEBUG)


@cluster_proc_dec(lambda: LocalCluster(n_workers=1, threads_per_worker=2))
# @flow
def img_overlap_pipeline(proj_fp_dict):
    # Read raw arr
    arr_raw = da.from_zarr(proj_fp_dict["raw"], chunks=PROC_CHUNKS)
    # Make overlapping blocks
    arr_overlap = da.overlap.overlap(arr_raw, depth=S_DEPTH, boundary="reflect")
    arr_overlap = disk_cache(arr_overlap, proj_fp_dict["overlap"])


@cluster_proc_dec(lambda: LocalCUDACluster())
# @flow
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
    arr_bgrm = da.map_blocks(GpuArrFuncs.tophat_filt, arr_overlap, tophat_sigma)
    arr_bgrm = disk_cache(arr_bgrm, proj_fp_dict["bgrm"])

    # Step 2: Difference of Gaussians (edge detection)
    arr_dog = da.map_blocks(GpuArrFuncs.dog_filt, arr_bgrm, dog_sigma1, dog_sigma2)
    arr_dog = disk_cache(arr_dog, proj_fp_dict["dog"])

    # Step 3: Gaussian subtraction with large sigma for adaptive thresholding
    arr_adaptv = da.map_blocks(GpuArrFuncs.gauss_subt_filt, arr_dog, gauss_sigma)
    arr_adaptv = disk_cache(arr_adaptv, proj_fp_dict["adaptv"])

    # Step 4: Mean thresholding with standard deviation offset
    # Visually inspect sd offset
    t_p = (
        arr_adaptv.sum() / (np.prod(arr_adaptv.shape) - (arr_adaptv == 0).sum())
    ).compute()
    print(t_p)
    arr_threshd = da.map_blocks(GpuArrFuncs.manual_thresh, arr_adaptv, thresh_p)
    arr_threshd = disk_cache(arr_threshd, proj_fp_dict["threshd"])

    # Step 5: Object sizes
    arr_sizes = da.map_blocks(GpuArrFuncs.label_with_sizes, arr_threshd)
    arr_sizes = disk_cache(arr_sizes, proj_fp_dict["sizes"])

    # Step 6: Filter out large objects (likely outlines, not cells)
    # TODO: Need to manually set min_size and max_size
    arr_filt = da.map_blocks(GpuArrFuncs.filt_by_size, arr_sizes, min_size, max_size)
    arr_filt = da.map_blocks(GpuArrFuncs.manual_thresh, arr_filt, 1)
    arr_filt = disk_cache(arr_filt, proj_fp_dict["filt"])

    # Step 7: Get maxima of image masked by labels
    arr_maxima = da.map_blocks(GpuArrFuncs.get_local_maxima, arr_overlap, maxima_sigma)
    arr_maxima = da.map_blocks(GpuArrFuncs.mask, arr_maxima, arr_filt)
    arr_maxima = disk_cache(arr_maxima, proj_fp_dict["maxima"])

    # Converting maxima to unique labels
    arr_maxima_labels = da.map_blocks(GpuArrFuncs.label_with_ids, arr_maxima)
    arr_maxima_labels = disk_cache(arr_maxima_labels, proj_fp_dict["maxima_labels"])

    # Step 8: Watershed segmentation
    # arr_watershed = da.map_blocks(watershed_segm, arr_overlap, arr_maxima, arr_filt)
    # arr_watershed = disk_cache(arr_watershed, proj_fp_dict["watershed"])


@cluster_proc_dec(lambda: LocalCluster(n_workers=6, threads_per_worker=1))
def img_get_cell_sizes(proj_fp_dict):
    # Loading in arrs
    arr_overlap = da.from_zarr(proj_fp_dict["overlap"])
    arr_filt = da.from_zarr(proj_fp_dict["filt"])
    arr_maxima_labels = da.from_zarr(proj_fp_dict["maxima_labels"])

    # Step 8: Watershed segmentation
    arr_watershed = da.map_blocks(
        CpuArrFuncs.watershed_segm, arr_overlap, arr_maxima_labels, arr_filt
    )
    arr_watershed = disk_cache(arr_watershed, proj_fp_dict["watershed"])

    # # With sizes
    # arr_watershed_sizes = arr_watershed.map_blocks(CpuArrFuncs.label_ids_to_sizes)
    # arr_watershed_sizes = disk_cache(
    #     arr_watershed_sizes, proj_fp_dict["watershed_sizes"]
    # )
    # # NOTE: And filter any cell over 1000??


@cluster_proc_dec(lambda: LocalCluster())
# @flow
def img_trim_pipeline(proj_fp_dict):
    # Read overlapped filtered and maxima images
    # Step 9a: trimming filtered regions overlaps
    arr_filt = da.from_zarr(proj_fp_dict["filt"])
    arr_filt_f = my_trim(arr_filt)
    disk_cache(arr_filt_f, proj_fp_dict["filt_final"])
    # Step 9b: trimming maxima points overlaps
    arr_maxima = da.from_zarr(proj_fp_dict["maxima"])
    arr_maxima_f = my_trim(arr_maxima)
    disk_cache(arr_maxima_f, proj_fp_dict["maxima_final"])
    # Step 9c: trimming watershed points overlaps
    arr_watershed = da.from_zarr(proj_fp_dict["watershed"])
    arr_watershed_f = my_trim(arr_watershed)
    disk_cache(arr_watershed_f, proj_fp_dict["watershed_final"])


@cluster_proc_dec(lambda: LocalCluster())
def img_to_cells_pipeline(proj_fp_dict):
    """
    Uses the overlapped images to get cell coords and corresponding sizes.
    Only considers cells within the boundary.
    """
    # Loading in arrs
    arr_raw = da.from_zarr(proj_fp_dict["raw"])
    arr_maxima_labels = da.from_zarr(proj_fp_dict["maxima_labels"])
    arr_watershed = da.from_zarr(proj_fp_dict["watershed"])
    # Getting anything inside boundary
    cell_coords = block_to_coords(
        CpuArrFuncs.get_cells, [arr_raw, arr_maxima_labels, arr_watershed]
    )
    cell_coords.to_parquet(proj_fp_dict["cells_raw_df"], overwrite=True)


@cluster_proc_dec(lambda: LocalCluster())
# @flow
def img_to_coords_pipeline(proj_fp_dict):
    # Read filtered and maxima images (trimmed - orig space)
    arr_maxima_f = da.from_zarr(proj_fp_dict["maxima_final"])
    # Step 10a: Get coords of maxima and get corresponding sizes from watershed
    cell_coords = block_to_coords(GpuArrFuncs.region_to_coords, [arr_maxima_f])
    cell_coords.to_parquet(proj_fp_dict["maxima_df"], overwrite=True)


if __name__ == "__main__":
    # Filenames
    proj_dir = "/home/linux1/Desktop/A-1-1/large_cellcount"
    # proj_dir = "/home/linux1/Desktop/A-1-1/cellcount"

    proj_fp_dict = get_proj_fp_dict(proj_dir)
    make_proj_dirs(proj_dir)

    img_overlap_pipeline(proj_fp_dict)

    img_proc_pipeline(
        proj_fp_dict=proj_fp_dict,
        tophat_sigma=10,
        dog_sigma1=1,
        dog_sigma2=4,
        gauss_sigma=101,
        thresh_p=32,
        min_size=100,
        max_size=10000,
        maxima_sigma=10,
    )

    img_get_cell_sizes(proj_fp_dict)

    img_trim_pipeline(proj_fp_dict)

    img_to_cells_pipeline(proj_fp_dict)
