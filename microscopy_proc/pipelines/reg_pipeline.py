import shutil

import dask.array as da
import tifffile
from dask.distributed import LocalCluster

from microscopy_proc.funcs.elastix_funcs import registration

# from prefect import flow
from microscopy_proc.funcs.reg_funcs import (
    downsmpl_fine_arr,
    downsmpl_rough_arr,
    reorient_arr,
)
from microscopy_proc.utils.config_params_model import ConfigParamsModel
from microscopy_proc.utils.dask_utils import cluster_proc_contxt
from microscopy_proc.utils.proj_org_utils import (
    ProjFpModel,
    RefFpModel,
    get_proj_fp_model,
    get_ref_fp_model,
    init_configs,
    make_proj_dirs,
)


# @flow
def ref_prepare_pipeline(
    ref_fp_dict: RefFpModel,
    pfm: ProjFpModel,
    **kwargs,
):
    # Update registration params json
    configs = ConfigParamsModel.update_params_file(pfm.config_params, **kwargs)
    # Making atlas images
    for fp_i, fp_o in [
        (ref_fp_dict.ref, pfm.ref),
        (ref_fp_dict.annot, pfm.annot),
    ]:
        # Reading
        arr = tifffile.imread(fp_i)
        # Reorienting
        arr = reorient_arr(arr, configs.ref_orient_ls)
        # Slicing
        arr = arr[
            slice(*configs.ref_z_trim),
            slice(*configs.ref_y_trim),
            slice(*configs.ref_x_trim),
        ]
        # Saving
        tifffile.imwrite(fp_o, arr)
    # Copying region mapping json to project folder
    shutil.copyfile(ref_fp_dict.map, pfm.map)
    # Copying transformation files
    shutil.copyfile(ref_fp_dict.affine, pfm.affine)
    shutil.copyfile(ref_fp_dict.bspline, pfm.bspline)


# @flow
def img_rough_pipeline(pfm: ProjFpModel, **kwargs):
    # Update registration params json
    configs = ConfigParamsModel.update_params_file(pfm.config_params, **kwargs)
    with cluster_proc_contxt(LocalCluster()):
        # Reading
        arr_raw = da.from_zarr(pfm.raw)
        # Rough downsample
        arr_downsmpl1 = downsmpl_rough_arr(
            arr_raw, configs.z_rough, configs.y_rough, configs.x_rough
        )
        arr_downsmpl1 = arr_downsmpl1.compute()
        # Saving
        tifffile.imwrite(pfm.downsmpl1, arr_downsmpl1)


# @flow
def img_fine_pipeline(pfm: dict, **kwargs):
    # Update registration params json
    rp = ConfigParamsModel.update_params_file(pfm.config_params, **kwargs)
    # Reading
    arr_downsmpl1 = tifffile.imread(pfm.downsmpl1)
    # Fine downsample
    arr_downsmpl2 = downsmpl_fine_arr(arr_downsmpl1, rp.z_fine, rp.y_fine, rp.x_fine)
    # Saving
    tifffile.imwrite(pfm.downsmpl2, arr_downsmpl2)


# @flow
def img_trim_pipeline(pfm: dict, **kwargs):
    # Update registration params json
    rp = ConfigParamsModel.update_params_file(pfm.config_params, **kwargs)
    # Reading
    arr_downsmpl2 = tifffile.imread(pfm.downsmpl2)
    # Trim
    arr_trimmed = arr_downsmpl2[slice(*rp.z_trim), slice(*rp.y_trim), slice(*rp.x_trim)]
    # Saving
    tifffile.imwrite(pfm.trimmed, arr_trimmed)


# @flow
def registration_pipeline(pfm: dict, **kwargs):
    # Running Elastix registration
    registration(
        fixed_img_fp=pfm.trimmed,
        moving_img_fp=pfm.ref,
        output_img_fp=pfm.regresult,
        affine_fp=pfm.affine,
        bspline_fp=pfm.bspline,
    )


if __name__ == "__main__":
    # Filenames
    # in_fp = "/home/linux1/Desktop/A-1-1/large_cellcount/raw.zarr"
    atlas_rsc_dir = "/home/linux1/Desktop/iDISCO/resources/atlas_resources/"
    proj_dir = "/home/linux1/Desktop/A-1-1/large_cellcount"

    ref_fp_dict = get_ref_fp_model()
    pfm = get_proj_fp_model(proj_dir)
    make_proj_dirs(proj_dir)

    # Making params json
    init_configs(pfm)

    # # Preparing reference images
    # prepare_ref(
    #     ref_fp_dict=ref_fp_dict,
    #     pfm=pfm,
    #     ref_orient_ls=(-2, 3, 1),
    #     ref_z_trim=(None, None, None),
    #     ref_y_trim=(None, None, None),
    #     ref_x_trim=(None, None, None),
    # )

    # # Preparing image itself
    # prepare_img_rough(
    #     pfm,
    #     z_rough=3,
    #     y_rough=6,
    #     x_rough=6,
    # )
    # prepare_img_fine(
    #     pfm,
    #     z_fine=1,
    #     y_fine=0.6,
    #     x_fine=0.6,
    # )
    # prepare_img_trim(
    #     pfm,
    #     z_trim=(None, -5, None),
    #     y_trim=(80, -75, None),
    #     x_trim=(None, None, None),
    # )

    # Running Elastix registration
    registration_pipeline(pfm)

    # # Transformix
    # arr_masked_trfm = tifffile.imread(pfm.trimmed)
    # arr_regresult = transformation_img(
    #     pfm.ref,
    #     pfm.regresult,
    # )
    # tifffile.imwrite(pfm.regresult, arr_regresult)
