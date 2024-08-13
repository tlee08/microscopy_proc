import dask.array as da
import tifffile

from microscopy_proc.funcs.elastix_funcs import transformation_img

# logging.basicConfig(level=logging.DEBUG)
# from prefect import flow
from microscopy_proc.utils.config_params_model import ConfigParamsModel
from microscopy_proc.utils.proj_org_utils import (
    get_proj_fp_dict,
    make_proj_dirs,
)

# TODO: JUST TAKE THE BRAIN MASK OF THE DOWNSAMPLED, TRIMMED IMAGE!!

if __name__ == "__main__":
    # Filenames
    proj_dir = "/home/linux1/Desktop/A-1-1/large_cellcount"
    # proj_dir = "/home/linux1/Desktop/A-1-1/cellcount"
    # proj_dir = "/run/user/1000/gvfs/smb-share:server=shared.sydney.edu.au,share=research-data/PRJ-BowenLab/Experiments/2024/Other/2024_whole_brain_clearing_TS/KNX_Aggression_cohort_1_analysed_images/B15_agg_2.5x_1xzoom_03072024"

    proj_fp_dict = get_proj_fp_dict(proj_dir)
    make_proj_dirs(proj_dir)

    # Update registration params json
    rp = ConfigParamsModel.update_params_file(proj_fp_dict["config_params"])
    # Reading raw image
    arr_raw = da.from_zarr(proj_fp_dict["raw"])

    # with cluster_proc_contxt(LocalCUDACluster()):
    #     # Step 0: Read raw image
    #     arr_raw = da.from_zarr(proj_fp_dict["raw"])

    #     arr_adaptv = da.map_blocks(Gf.gauss_blur_filt, arr_raw, 5)
    #     arr_adaptv = disk_cache(arr_adaptv, proj_fp_dict["adaptv"] + "_blur.zarr")

    # with cluster_proc_contxt(LocalCluster()):
    #     arr_threshd = da.map_blocks(Cf.manual_thresh, arr_adaptv, 450)
    #     arr_threshd = disk_cache(arr_threshd, proj_fp_dict["threshd"] + "_mask.zarr")

    # with cluster_proc_contxt(LocalCluster()):
    #     # Reading
    #     arr_threshd = da.from_zarr(proj_fp_dict["threshd"] + "_mask.zarr")
    #     # Rough downsample
    #     arr_downsmpl1 = downsmpl_rough_arr(
    #         arr_threshd, rp.z_rough, rp.y_rough, rp.x_rough
    #     )
    #     arr_downsmpl1 = arr_downsmpl1.compute()
    #     # Saving
    #     tifffile.imwrite(proj_fp_dict["downsmpl1"] + "_mask.tif", arr_downsmpl1)

    # # Fine downsample
    # arr_downsmpl2 = downsmpl_fine_arr(arr_downsmpl1, rp.z_fine, rp.y_fine, rp.x_fine)
    # # Saving
    # tifffile.imwrite(proj_fp_dict["downsmpl2"] + "_mask.tif", arr_downsmpl2)

    # # Trim
    # arr_trimmed = arr_downsmpl2[slice(*rp.z_trim), slice(*rp.y_trim), slice(*rp.x_trim)]
    # # Saving
    # tifffile.imwrite(proj_fp_dict["trimmed"] + "_mask.tif", arr_trimmed)

    # # Convert 3d tensor to (z, y, x) coords
    # coords_df = Gf.get_coords(arr_trimmed)
    # dd.from_pandas(coords_df).to_parquet(proj_fp_dict["cells_raw_df"] + "_mask.parquet")

    # Transformix
    arr_masked_trfm = tifffile.imread(proj_fp_dict["trimmed"])
    x = transformation_img(
        proj_fp_dict["annot"],
        proj_fp_dict["regresult"],
    )
    tifffile.imwrite(proj_fp_dict["regresult"] + "_mask_annot.tif", x)

    # Transformix
    # coords_df = dd.read_parquet(
    #     proj_fp_dict["cells_raw_df"] + "_mask.parquet"
    # ).compute()
    # coords_trfm = transformation_coords(
    #     coords_df,
    #     proj_fp_dict["ref"],
    #     proj_fp_dict["regresult"],
    # )
    # dd.from_pandas(coords_trfm).to_parquet(
    #     proj_fp_dict["cells_raw_df"] + "_mask_trfm.parquet"
    # )

    # with cluster_proc_contxt(LocalCluster()):
    #     df = dd.read_parquet(proj_fp_dict["cells_raw_df"]).compute()
    #     coords_to_points(
    #         df,
    #         tifffile.imread(proj_fp_dict["ref"]).shape,
    #         proj_fp_dict["points_check"] + "mask.zarr",
    #     )
    #     x = da.from_zarr(proj_fp_dict["points_check"] + "mask.zarr")
    #     tifffile.imwrite(proj_fp_dict["points_check"] + "mask.tif", x.compute())
