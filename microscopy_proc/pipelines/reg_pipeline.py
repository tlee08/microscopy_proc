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
from microscopy_proc.utils.dask_utils import cluster_proc_contxt
from microscopy_proc.utils.io_utils import read_json, write_json
from microscopy_proc.utils.proj_org_utils import (
    get_proj_fp_dict,
    get_ref_fp_dict,
    make_proj_dirs,
)
from microscopy_proc.utils.reg_params_model import RegParamsModel

# logging.basicConfig(level=logging.DEBUG)


# @flow
def prepare_ref(
    ref_fp_dict: dict,
    proj_fp_dict: dict,
    # Ref Params
    **kwargs,
):
    # Making registration params json
    try:  # If file exists
        rp = RegParamsModel.model_validate(read_json(proj_fp_dict["reg_params"]))
    except Exception:  # If file does not exist
        rp = RegParamsModel()
    rp = RegParamsModel.model_validate(rp.model_copy(update=kwargs))
    write_json(proj_fp_dict["reg_params"], rp.model_dump())
    # Making atlas images
    for fp_i, fp_o in [
        (ref_fp_dict["ref"], proj_fp_dict["ref"]),
        (ref_fp_dict["annot"], proj_fp_dict["annot"]),
    ]:
        # Reading
        arr = tifffile.imread(fp_i)
        # Reorienting
        arr = reorient_arr(arr, rp.ref_orient_ls)
        # Slicing
        arr = arr[slice(*rp.ref_z_trim), slice(*rp.ref_y_trim), slice(*rp.ref_x_trim)]
        # Saving
        tifffile.imwrite(fp_o, arr)
    # Copying region mapping json to project folder
    shutil.copyfile(ref_fp_dict["map"], proj_fp_dict["map"])
    # Copying transformation files
    shutil.copyfile(ref_fp_dict["affine"], proj_fp_dict["affine"])
    shutil.copyfile(ref_fp_dict["bspline"], proj_fp_dict["bspline"])


# @flow
def prepare_img_rough(proj_fp_dict: dict, **kwargs):
    with cluster_proc_contxt(LocalCluster()):
        # Update registration params json
        rp = RegParamsModel.model_validate(read_json(proj_fp_dict["reg_params"]))
        rp = RegParamsModel.model_validate(rp.model_copy(update=kwargs))
        write_json(proj_fp_dict["reg_params"], rp.model_dump())
        # Reading
        arr_raw = da.from_zarr(proj_fp_dict["raw"])
        # Rough downsample
        arr_downsmpl1 = downsmpl_rough_arr(arr_raw, rp.z_rough, rp.y_rough, rp.x_rough)
        arr_downsmpl1 = arr_downsmpl1.compute()
        # Saving
        tifffile.imwrite(proj_fp_dict["downsmpl1"], arr_downsmpl1)


# @flow
def prepare_img_fine(proj_fp_dict: dict, **kwargs):
    # Update registration params json
    rp = RegParamsModel.model_validate(read_json(proj_fp_dict["reg_params"]))
    rp = RegParamsModel.model_validate(rp.model_copy(update=kwargs))
    write_json(proj_fp_dict["reg_params"], rp.model_dump())
    # Reading
    arr_downsmpl1 = tifffile.imread(proj_fp_dict["downsmpl1"])
    # Fine downsample
    arr_downsmpl2 = downsmpl_fine_arr(arr_downsmpl1, rp.z_fine, rp.y_fine, rp.x_fine)
    # Saving
    tifffile.imwrite(proj_fp_dict["downsmpl2"], arr_downsmpl2)


# @flow
def prepare_img_trim(proj_fp_dict: dict, **kwargs):
    # Update registration params json
    rp = RegParamsModel.model_validate(read_json(proj_fp_dict["reg_params"]))
    rp = RegParamsModel.model_validate(rp.model_copy(update=kwargs))
    write_json(proj_fp_dict["reg_params"], rp.model_dump())
    # Reading
    arr_downsmpl2 = tifffile.imread(proj_fp_dict["downsmpl2"])
    # Trim
    arr_trimmed = arr_downsmpl2[slice(*rp.z_trim), slice(*rp.y_trim), slice(*rp.x_trim)]
    # Saving
    tifffile.imwrite(proj_fp_dict["trimmed"], arr_trimmed)


if __name__ == "__main__":
    # Filenames
    # in_fp = "/home/linux1/Desktop/A-1-1/large_cellcount/raw.zarr"
    atlas_rsc_dir = "/home/linux1/Desktop/iDISCO/resources/atlas_resources/"
    proj_dir = "/home/linux1/Desktop/A-1-1/large_cellcount"

    ref_fp_dict = get_ref_fp_dict(atlas_rsc_dir)
    proj_fp_dict = get_proj_fp_dict(proj_dir)
    make_proj_dirs(proj_dir)

    # Preparing reference images
    prepare_ref(
        ref_fp_dict=ref_fp_dict,
        proj_fp_dict=proj_fp_dict,
        ref_orient_ls=(2, 3, 1),
        ref_z_trim=(None, None, None),
        ref_y_trim=(None, None, None),
        ref_x_trim=(None, None, None),
    )

    # Preparing image itself
    prepare_img_rough(
        proj_fp_dict,
        z_rough=3,
        y_rough=6,
        x_rough=6,
    )
    prepare_img_fine(
        proj_fp_dict,
        z_fine=1,
        y_fine=0.6,
        x_fine=0.6,
    )
    prepare_img_trim(
        proj_fp_dict,
        z_trim=(None, -5, None),
        y_trim=(80, -75, None),
        x_trim=(None, None, None),
    )

    # Running Elastix registration
    registration(
        fixed_img_fp=proj_fp_dict["trimmed"],
        moving_img_fp=proj_fp_dict["ref"],
        output_img_fp=proj_fp_dict["regresult"],
        affine_fp=proj_fp_dict["affine"],
        bspline_fp=proj_fp_dict["bspline"],
    )
