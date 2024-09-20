import dask.array as da
import dask.dataframe as dd
from dask.distributed import LocalCluster
from dask_cuda import LocalCUDACluster

# from prefect import flow
from microscopy_proc.constants import DEPTH, PROC_CHUNKS
from microscopy_proc.funcs.cpu_arr_funcs import CpuArrFuncs as Cf
from microscopy_proc.funcs.gpu_arr_funcs import GpuArrFuncs as Gf
from microscopy_proc.utils.config_params_model import ConfigParamsModel
from microscopy_proc.utils.dask_utils import (
    block2coords,
    cluster_proc_contxt,
    da_overlap,
    da_trim,
    disk_cache,
)
from microscopy_proc.utils.proj_org_utils import (
    get_proj_fp_dict,
    init_params,
    make_proj_dirs,
)


# @flow
def img_overlap_pipeline(proj_fp_dict, chunks=PROC_CHUNKS, d=DEPTH):
    with cluster_proc_contxt(LocalCluster(n_workers=1, threads_per_worker=4)):
        arr_raw = da.from_zarr(proj_fp_dict["raw"], chunks=chunks)
        arr_overlap = da_overlap(arr_raw, d=d)
        arr_overlap = disk_cache(arr_overlap, proj_fp_dict["overlap"])


# @flow
def img_proc_pipeline(proj_fp_dict, **kwargs):
    # Update registration params json
    rp = ConfigParamsModel.update_params_file(proj_fp_dict["config_params"], **kwargs)
    # Reading raw image
    arr_raw = da.from_zarr(proj_fp_dict["raw"])
    # Reading overlapped image
    arr_overlap = da.from_zarr(proj_fp_dict["overlap"])

    with cluster_proc_contxt(LocalCUDACluster()):
        # Step 1: Top-hat filter (background subtraction)
        arr_bgrm = da.map_blocks(Gf.tophat_filt, arr_overlap, rp.tophat_sigma)
        arr_bgrm = disk_cache(arr_bgrm, proj_fp_dict["bgrm"])

        # Step 2: Difference of Gaussians (edge detection)
        arr_dog = da.map_blocks(Gf.dog_filt, arr_bgrm, rp.dog_sigma1, rp.dog_sigma2)
        arr_dog = disk_cache(arr_dog, proj_fp_dict["dog"])

        # Step 3: Gaussian subtraction with large sigma for adaptive thresholding
        arr_adaptv = da.map_blocks(Gf.gauss_subt_filt, arr_dog, rp.gauss_sigma)
        arr_adaptv = disk_cache(arr_adaptv, proj_fp_dict["adaptv"])

    with cluster_proc_contxt(LocalCluster()):
        # Step 4: Mean thresholding with standard deviation offset
        # # Visually inspect sd offset
        # t_p = arr_adaptv.sum() / (np.prod(arr_adaptv.shape) - (arr_adaptv == 0).sum())
        # t_p = t_p.compute()
        # logging.debug(t_p)
        arr_threshd = da.map_blocks(Cf.manual_thresh, arr_adaptv, rp.thresh_p)
        arr_threshd = disk_cache(arr_threshd, proj_fp_dict["threshd"])

    with cluster_proc_contxt(LocalCluster(n_workers=6, threads_per_worker=1)):
        # Step 5: Object sizes
        arr_sizes = da.map_blocks(Cf.label_with_sizes, arr_threshd)
        arr_sizes = disk_cache(arr_sizes, proj_fp_dict["threshd_sizes"])

    with cluster_proc_contxt(LocalCluster()):
        # Step 6: Filter out large objects (likely outlines, not cells)
        arr_threshd_filt = da.map_blocks(
            Cf.filt_by_size, arr_sizes, rp.min_threshd, rp.max_threshd
        )
        arr_threshd_filt = disk_cache(arr_threshd_filt, proj_fp_dict["threshd_filt"])

    with cluster_proc_contxt(LocalCUDACluster()):
        # Step 7: Get maxima of image masked by labels
        arr_maxima = da.map_blocks(
            Gf.get_local_maxima, arr_overlap, rp.maxima_sigma, arr_threshd_filt
        )
        arr_maxima = disk_cache(arr_maxima, proj_fp_dict["maxima"])

    with cluster_proc_contxt(LocalCluster(n_workers=3, threads_per_worker=1)):
        # n_workers=2
        # Step 8: Watershed segmentation sizes
        arr_wshed_sizes = da.map_blocks(
            Cf.wshed_segm_sizes, arr_overlap, arr_maxima, arr_threshd_filt
        )
        arr_wshed_sizes = disk_cache(arr_wshed_sizes, proj_fp_dict["wshed_sizes"])

    with cluster_proc_contxt(LocalCluster()):
        # Step 9: Filter out large watershed objects (again likely outlines, not cells)
        arr_wshed_filt = da.map_blocks(
            Cf.filt_by_size, arr_wshed_sizes, rp.min_wshed, rp.max_wshed
        )
        arr_wshed_filt = disk_cache(arr_wshed_filt, proj_fp_dict["wshed_filt"])

        # Step 10: Trimming filtered regions overlaps
        # Trimming maxima points overlaps
        arr_maxima_f = da_trim(arr_maxima, d=rp.d)
        disk_cache(arr_maxima_f, proj_fp_dict["maxima_final"])
        # Trimming filtered regions overlaps
        arr_threshd_final = da_trim(arr_threshd_filt, d=rp.d)
        disk_cache(arr_threshd_final, proj_fp_dict["threshd_final"])
        # Trimming watershed sizes overlaps
        arr_wshed_final = da_trim(arr_wshed_sizes, d=rp.d)
        disk_cache(arr_wshed_final, proj_fp_dict["wshed_final"])

    with cluster_proc_contxt(LocalCluster(n_workers=2, threads_per_worker=1)):
        # n_workers=2
        # Getting maxima coords and corresponding watershed sizes in table
        cells_df = block2coords(
            Cf.get_cells, arr_raw, arr_overlap, arr_maxima, arr_wshed_filt, rp.d
        )
        # Filtering out by size
        cells_df = cells_df.query(f"size >= {rp.min_wshed} and size <= {rp.max_wshed}")
        # Saving to parquet
        cells_df.to_parquet(proj_fp_dict["cells_raw_df"], overwrite=True)


def img2coords_pipeline(proj_fp_dict):
    with cluster_proc_contxt(LocalCluster(n_workers=6, threads_per_worker=1)):
        # Read filtered and maxima images (trimmed - orig space)
        arr_maxima_f = da.from_zarr(proj_fp_dict["maxima_final"])
        # Step 10a: Get coords of maxima and get corresponding sizes from watershed
        coords_df = block2coords(Gf.get_coords, arr_maxima_f)
        coords_df.to_parquet(proj_fp_dict["maxima_df"], overwrite=True)


def cells_df_smb_field_patch(fp):
    """
    Patch for some error that makes another column called
    smb-share:server. I don't know where the error is happening
    but this is a patch to fix it after the fact.
    """
    cells_df = dd.read_parquet(fp)
    if "smb-share:server" in cells_df.columns:
        print("IN")
        cells_df = cells_df.drop(columns=["smb-share:server"])
        cells_df.to_parquet(fp, overwrite=True)


if __name__ == "__main__":
    # Filenames
    proj_dir = "/home/linux1/Desktop/A-1-1/large_cellcount"
    # proj_dir = "/home/linux1/Desktop/A-1-1/cellcount"

    proj_fp_dict = get_proj_fp_dict(proj_dir)
    make_proj_dirs(proj_dir)

    # Making params json
    init_params(proj_fp_dict)

    img_overlap_pipeline(proj_fp_dict, chunks=PROC_CHUNKS, d=DEPTH)

    img_proc_pipeline(
        proj_fp_dict=proj_fp_dict,
        d=DEPTH,
        tophat_sigma=10,
        dog_sigma1=1,
        dog_sigma2=4,
        gauss_sigma=101,
        thresh_p=60,
        min_threshd=100,
        max_threshd=9000,
        maxima_sigma=10,
        min_wshed=1,
        max_wshed=700,
    )

    cells_df_smb_field_patch(proj_fp_dict["cells_raw_df"])

    # img2coords_pipeline(proj_fp_dict)
