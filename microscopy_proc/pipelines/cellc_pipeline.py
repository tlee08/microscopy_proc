import dask.array as da
import numpy as np
from dask.distributed import LocalCluster
from dask_cuda import LocalCUDACluster

# from prefect import flow
from microscopy_proc.constants import DEPTH, PROC_CHUNKS
from microscopy_proc.funcs.cpu_arr_funcs import CpuArrFuncs as Cf
from microscopy_proc.funcs.gpu_arr_funcs import GpuArrFuncs as Gf
from microscopy_proc.utils.dask_utils import (
    block_to_coords,
    cluster_proc_contxt,
    disk_cache,
    my_trim,
)
from microscopy_proc.utils.proj_org_utils import get_proj_fp_dict, make_proj_dirs

# logging.basicConfig(level=logging.DEBUG)


# @flow
def img_overlap_pipeline(proj_fp_dict):
    with cluster_proc_contxt(LocalCluster(n_workers=1, threads_per_worker=2)):
        # Read raw arr
        arr_raw = da.from_zarr(proj_fp_dict["raw"], chunks=PROC_CHUNKS)
        # Make overlapping blocks
        arr_overlap = da.overlap.overlap(arr_raw, depth=DEPTH, boundary="reflect")
        arr_overlap = disk_cache(arr_overlap, proj_fp_dict["overlap"])


# @flow
def img_proc_pipeline(
    proj_fp_dict,
    tophat_sigma=10,
    dog_sigma1=2,
    dog_sigma2=5,
    gauss_sigma=101,
    thresh_p=30,
    min_threshd=1,
    max_threshd=3000,
    min_wshed=1,
    max_wshed=1000,
    maxima_sigma=10,
):
    with cluster_proc_contxt(LocalCUDACluster()):
        # Step 0: Read overlapped image
        arr_overlap = da.from_zarr(proj_fp_dict["overlap"])

        # Step 1: Top-hat filter (background subtraction)
        arr_bgrm = da.map_blocks(Gf.tophat_filt, arr_overlap, tophat_sigma)
        arr_bgrm = disk_cache(arr_bgrm, proj_fp_dict["bgrm"])

        # Step 2: Difference of Gaussians (edge detection)
        arr_dog = da.map_blocks(Gf.dog_filt, arr_bgrm, dog_sigma1, dog_sigma2)
        arr_dog = disk_cache(arr_dog, proj_fp_dict["dog"])

        # Step 3: Gaussian subtraction with large sigma for adaptive thresholding
        arr_adaptv = da.map_blocks(Gf.gauss_subt_filt, arr_dog, gauss_sigma)
        arr_adaptv = disk_cache(arr_adaptv, proj_fp_dict["adaptv"])

    with cluster_proc_contxt(LocalCluster()):
        # Step 4: Mean thresholding with standard deviation offset
        # Visually inspect sd offset
        t_p = (
            arr_adaptv.sum() / (np.prod(arr_adaptv.shape) - (arr_adaptv == 0).sum())
        ).compute()
        print(t_p)
        arr_threshd = da.map_blocks(Cf.manual_thresh, arr_adaptv, thresh_p)
        arr_threshd = disk_cache(arr_threshd, proj_fp_dict["threshd"])

    with cluster_proc_contxt(LocalCUDACluster()):
        # Step 5: Object sizes
        arr_sizes = da.map_blocks(Gf.label_with_sizes, arr_threshd)
        arr_sizes = disk_cache(arr_sizes, proj_fp_dict["threshd_sizes"])

    with cluster_proc_contxt(LocalCluster()):
        # Step 6: Filter out large objects (likely outlines, not cells)
        arr_threshd_filt = da.map_blocks(
            Cf.filt_by_size, arr_sizes, min_threshd, max_threshd
        )
        arr_threshd_filt = disk_cache(arr_threshd_filt, proj_fp_dict["threshd_filt"])

    with cluster_proc_contxt(LocalCUDACluster()):
        # Step 7: Get maxima of image masked by labels
        arr_maxima = da.map_blocks(
            Gf.get_local_maxima, arr_overlap, maxima_sigma, arr_threshd_filt
        )
        arr_maxima = disk_cache(arr_maxima, proj_fp_dict["maxima"])

    with cluster_proc_contxt(LocalCluster(n_workers=4, threads_per_worker=1)):
        # Step 8: Watershed segmentation sizes
        arr_wshed_sizes = da.map_blocks(
            Cf.wshed_segm_sizes, arr_overlap, arr_maxima, arr_threshd_filt
        )
        arr_wshed_sizes = disk_cache(arr_wshed_sizes, proj_fp_dict["wshed_sizes"])

    with cluster_proc_contxt(LocalCluster()):
        # Step 9: Filter out large watershed objects (again likely outlines, not cells)
        arr_wshed_filt = da.map_blocks(
            Cf.filt_by_size, arr_wshed_sizes, min_wshed, max_wshed
        )
        arr_wshed_filt = disk_cache(arr_wshed_filt, proj_fp_dict["wshed_filt"])

        # Step 10: Trimming filtered regions overlaps
        # Trimming maxima points overlaps
        arr_maxima_f = my_trim(arr_maxima, d=DEPTH)
        disk_cache(arr_maxima_f, proj_fp_dict["maxima_final"])
        # Trimming filtered regions overlaps
        arr_threshd_final = my_trim(arr_threshd_filt, d=DEPTH)
        disk_cache(arr_threshd_final, proj_fp_dict["threshd_final"])
        # Trimming watershed sizes overlaps
        arr_wshed_final = my_trim(arr_wshed_sizes, d=DEPTH)
        disk_cache(arr_wshed_final, proj_fp_dict["wshed_final"])

    with cluster_proc_contxt(LocalCluster(n_workers=4, threads_per_worker=1)):
        # Getting maxima coords and corresponding watershed sizes in table
        cells_df = block_to_coords(
            Cf.get_cells, arr_overlap, arr_maxima, arr_wshed_filt, 10
        )
        # Filtering out by size
        cells_df = cells_df.query(f"size >= {min_wshed} and size <= {max_wshed}")
        # Saving to parquet
        cells_df.to_parquet(proj_fp_dict["cells_raw_df"], overwrite=True)


def img_to_coords_pipeline(proj_fp_dict):
    with cluster_proc_contxt(LocalCluster(n_workers=4, threads_per_worker=1)):
        # Read filtered and maxima images (trimmed - orig space)
        arr_maxima_f = da.from_zarr(proj_fp_dict["maxima_final"])
        # Step 10a: Get coords of maxima and get corresponding sizes from watershed
        coords_df = block_to_coords(Gf.get_coords, arr_maxima_f)
        coords_df.to_parquet(proj_fp_dict["maxima_df"], overwrite=True)


if __name__ == "__main__":
    # Filenames
    # proj_dir = "/home/linux1/Desktop/A-1-1/large_cellcount"
    proj_dir = "/home/linux1/Desktop/A-1-1/cellcount"

    proj_fp_dict = get_proj_fp_dict(proj_dir)
    make_proj_dirs(proj_dir)

    # img_overlap_pipeline(proj_fp_dict)

    img_proc_pipeline(
        proj_fp_dict=proj_fp_dict,
        tophat_sigma=10,
        dog_sigma1=1,
        dog_sigma2=4,
        gauss_sigma=101,
        thresh_p=32,
        min_threshd=100,
        max_threshd=10000,
        min_wshed=1,
        max_wshed=1000,
        maxima_sigma=10,
    )

    # img_to_coords_pipeline(proj_fp_dict)
