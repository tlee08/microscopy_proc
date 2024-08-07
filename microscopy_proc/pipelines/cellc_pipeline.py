import dask.array as da
from dask.distributed import LocalCluster

# from prefect import flow
from microscopy_proc.constants import PROC_CHUNKS, S_DEPTH
from microscopy_proc.funcs.cpu_arr_funcs import CpuArrFuncs
from microscopy_proc.funcs.gpu_arr_funcs import GpuArrFuncs
from microscopy_proc.utils.dask_utils import (
    block_to_coords,
    cluster_proc_contxt,
    disk_cache,
)
from microscopy_proc.utils.proj_org_utils import get_proj_fp_dict, make_proj_dirs

# logging.basicConfig(level=logging.DEBUG)


# @flow
def img_overlap_pipeline(proj_fp_dict):
    with cluster_proc_contxt(LocalCluster(n_workers=1, threads_per_worker=2)):
        # Read raw arr
        arr_raw = da.from_zarr(proj_fp_dict["raw"], chunks=PROC_CHUNKS)
        # Make overlapping blocks
        arr_overlap = da.overlap.overlap(arr_raw, depth=S_DEPTH, boundary="reflect")
        arr_overlap = disk_cache(arr_overlap, proj_fp_dict["overlap"])


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
    # with cluster_proc_contxt(LocalCUDACluster()):
    #     # Step 0: Read overlapped image
    #     arr_overlap = da.from_zarr(proj_fp_dict["overlap"])

    #     # Step 1: Top-hat filter (background subtraction)
    #     arr_bgrm = da.map_blocks(GpuArrFuncs.tophat_filt, arr_overlap, tophat_sigma)
    #     arr_bgrm = disk_cache(arr_bgrm, proj_fp_dict["bgrm"])

    #     # Step 2: Difference of Gaussians (edge detection)
    #     arr_dog = da.map_blocks(GpuArrFuncs.dog_filt, arr_bgrm, dog_sigma1, dog_sigma2)
    #     arr_dog = disk_cache(arr_dog, proj_fp_dict["dog"])

    #     # Step 3: Gaussian subtraction with large sigma for adaptive thresholding
    #     arr_adaptv = da.map_blocks(GpuArrFuncs.gauss_subt_filt, arr_dog, gauss_sigma)
    #     arr_adaptv = disk_cache(arr_adaptv, proj_fp_dict["adaptv"])

    # with cluster_proc_contxt(LocalCluster()):
    #     # Step 4: Mean thresholding with standard deviation offset
    #     # Visually inspect sd offset
    #     t_p = (
    #         arr_adaptv.sum() / (np.prod(arr_adaptv.shape) - (arr_adaptv == 0).sum())
    #     ).compute()
    #     print(t_p)
    #     arr_threshd = da.map_blocks(CpuArrFuncs.manual_thresh, arr_adaptv, thresh_p)
    #     arr_threshd = disk_cache(arr_threshd, proj_fp_dict["threshd"])

    # with cluster_proc_contxt(LocalCUDACluster()):
    #     # Step 5: Object sizes
    #     arr_sizes = da.map_blocks(GpuArrFuncs.label_with_sizes, arr_threshd)
    #     arr_sizes = disk_cache(arr_sizes, proj_fp_dict["sizes"])

    # with cluster_proc_contxt(LocalCluster()):
    #     # Step 6: Filter out large objects (likely outlines, not cells)
    #     # TODO: Manually set min_size and max_size
    #     arr_filt = da.map_blocks(
    #         CpuArrFuncs.filt_by_size, arr_sizes, min_size, max_size
    #     )
    #     arr_filt = da.map_blocks(CpuArrFuncs.manual_thresh, arr_filt, 1)
    #     arr_filt = disk_cache(arr_filt, proj_fp_dict["filt"])

    # with cluster_proc_contxt(LocalCUDACluster()):
    #     # Step 7: Get maxima of image masked by labels
    #     arr_maxima = da.map_blocks(
    #         GpuArrFuncs.get_local_maxima, arr_overlap, maxima_sigma, arr_filt
    #     )
    #     arr_maxima = disk_cache(arr_maxima, proj_fp_dict["maxima"])

    with cluster_proc_contxt(LocalCluster(n_workers=4, threads_per_worker=1)):
        arr_overlap = da.from_zarr(proj_fp_dict["overlap"])
        arr_maxima = da.from_zarr(proj_fp_dict["maxima"])
        arr_filt = da.from_zarr(proj_fp_dict["filt"])
        # Step 8: Watershed segmentation sizes
        arr_wshed_sizes = da.map_blocks(
            CpuArrFuncs.wshed_segm_sizes, arr_overlap, arr_maxima, arr_filt
        )
        arr_wshed_sizes = disk_cache(arr_wshed_sizes, proj_fp_dict["wshed_sizes"])

    # with cluster_proc_contxt(LocalCluster()):
    #     # Step 9: Trimming filtered regions overlaps
    #     # Trimming maxima points overlaps
    #     arr_maxima = da.from_zarr(proj_fp_dict["maxima"])
    #     arr_maxima_f = my_trim(arr_maxima, d=S_DEPTH)
    #     disk_cache(arr_maxima_f, proj_fp_dict["maxima_final"])
    #     # Trimming filtered regions overlaps
    #     arr_filt = da.from_zarr(proj_fp_dict["filt"])
    #     arr_filt_f = my_trim(arr_filt, d=S_DEPTH)
    #     disk_cache(arr_filt_f, proj_fp_dict["filt_final"])
    #     # Trimming watershed sizes overlaps
    #     arr_wshed = da.from_zarr(proj_fp_dict["wshed_sizes"])
    #     arr_wshed_f = my_trim(arr_wshed, d=S_DEPTH)
    #     disk_cache(arr_wshed_f, proj_fp_dict["wshed_sizes_final"])


def img_to_cells_pipeline(proj_fp_dict):
    """
    Uses the overlapped images to get cell coords and corresponding sizes.
    Only considers cells within the boundary.
    """
    with cluster_proc_contxt(LocalCluster(n_workers=5, threads_per_worker=1)):
        # with cluster_proc_contxt(LocalCUDACluster()):
        # Loading in arrs
        arr_overlap = da.from_zarr(proj_fp_dict["overlap"])
        arr_maxima = da.from_zarr(proj_fp_dict["maxima"])
        arr_filt = da.from_zarr(proj_fp_dict["filt"])
        # Getting maxima coords and corresponding watershed sizes in table
        cells_df = block_to_coords(
            CpuArrFuncs.get_cells, arr_overlap, arr_maxima, arr_filt, 10
        )
        cells_df.to_parquet(proj_fp_dict["cells_raw_df"] + "B.parquet", overwrite=True)


def img_to_coords_pipeline(proj_fp_dict):
    with cluster_proc_contxt(LocalCluster(n_workers=4, threads_per_worker=1)):
        # Read filtered and maxima images (trimmed - orig space)
        arr_maxima_f = da.from_zarr(proj_fp_dict["maxima_final"])
        # Step 10a: Get coords of maxima and get corresponding sizes from watershed
        coords_df = block_to_coords(GpuArrFuncs.get_coords, arr_maxima_f)
        coords_df.to_parquet(proj_fp_dict["maxima_df"], overwrite=True)


if __name__ == "__main__":
    # Filenames
    proj_dir = "/home/linux1/Desktop/A-1-1/large_cellcount"
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
        min_size=100,
        max_size=10000,
        maxima_sigma=10,
    )

    img_to_cells_pipeline(proj_fp_dict)

    # img_to_coords_pipeline(proj_fp_dict)
