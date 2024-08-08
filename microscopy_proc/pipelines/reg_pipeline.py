import shutil
from typing import Optional

import dask.array as da
import tifffile
from dask.distributed import LocalCluster

from microscopy_proc.funcs.elastix_funcs import registration

# from prefect import flow0
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

# logging.basicConfig(level=logging.DEBUG)


# @flow
def prepare_ref(
    ref_fp_dict: dict,
    proj_fp_dict: dict,
    # Ref Params
    ref_orient_ls: Optional[list] = (1, 2, 3),
    ref_z_trim: Optional[slice] = slice(None, None),
    ref_y_trim: Optional[slice] = slice(None, None),
    ref_x_trim: Optional[slice] = slice(None, None),
    z_rough: Optional[int] = 3,
    y_rough: Optional[int] = 6,
    x_rough: Optional[int] = 6,
    z_fine: Optional[float] = 1.0,
    y_fine: Optional[float] = 0.6,
    x_fine: Optional[float] = 0.6,
    z_trim: Optional[slice] = slice(None, None),
    y_trim: Optional[slice] = slice(None, None),
    x_trim: Optional[slice] = slice(None, None),
):
    # Making atlas images
    for fp_i, fp_o in [
        (ref_fp_dict["ref"], proj_fp_dict["ref"]),
        (ref_fp_dict["annot"], proj_fp_dict["annot"]),
    ]:
        # Reading
        arr = tifffile.imread(fp_i)
        # Reorienting
        arr = reorient_arr(arr, ref_orient_ls)
        # Slicing
        arr = arr[ref_z_trim, ref_y_trim, ref_x_trim]
        # Saving
        tifffile.imwrite(fp_o, arr)
    # Copying region mapping json to project folder
    shutil.copyfile(ref_fp_dict["map"], proj_fp_dict["map"])
    # Copying transformation files
    shutil.copyfile(ref_fp_dict["affine"], proj_fp_dict["affine"])
    shutil.copyfile(ref_fp_dict["bspline"], proj_fp_dict["bspline"])
    # Making registration params json
    write_json(
        proj_fp_dict["reg_params"],
        {
            # Ref params
            "ref_orient_ls": proj_fp_dict["ref"],
            "ref_z_trim": [ref_z_trim.start, ref_z_trim.stop],
            "ref_y_trim": [ref_y_trim.start, ref_y_trim.stop],
            "ref_x_trim": [ref_x_trim.start, ref_x_trim.stop],
            # Img params
            "z_rough": z_rough,
            "y_rough": y_rough,
            "x_rough": x_rough,
            "z_fine": z_fine,
            "y_fine": y_fine,
            "x_fine": x_fine,
            "z_trim": [z_trim.start, z_trim.stop],
            "y_trim": [y_trim.start, y_trim.stop],
            "x_trim": [x_trim.start, x_trim.stop],
        },
    )


# @flow
def prepare_img_rough(
    proj_fp_dict: dict,
    z_rough: Optional[int] = None,
    y_rough: Optional[int] = None,
    x_rough: Optional[int] = None,
):
    with cluster_proc_contxt(LocalCluster()):
        # Update registration params json
        rp = read_json(proj_fp_dict["reg_params"])
        rp["z_rough"] = z_rough if z_rough else rp["z_rough"]
        rp["y_rough"] = y_rough if y_rough else rp["y_rough"]
        rp["x_rough"] = x_rough if x_rough else rp["x_rough"]
        write_json(proj_fp_dict["reg_params"], rp)
        # Reading
        arr_raw = da.from_zarr(proj_fp_dict["raw"])
        # Rough downsample
        arr_downsmpl1 = downsmpl_rough_arr(
            arr_raw, rp["z_rough"], rp["y_rough"], rp["x_rough"]
        ).compute()
        # Saving
        tifffile.imwrite(proj_fp_dict["downsmpl1"], arr_downsmpl1)


# @flow
def prepare_img_fine(
    proj_fp_dict: dict,
    z_fine: Optional[float] = None,
    y_fine: Optional[float] = None,
    x_fine: Optional[float] = None,
):
    # Update registration params json
    rp = read_json(proj_fp_dict["reg_params"])
    rp["z_fine"] = z_fine if z_fine else rp["z_fine"]
    rp["y_fine"] = y_fine if y_fine else rp["y_fine"]
    rp["x_fine"] = x_fine if x_fine else rp["x_fine"]
    write_json(proj_fp_dict["reg_params"], rp)
    # Reading
    arr_downsmpl1 = tifffile.imread(proj_fp_dict["downsmpl1"])
    # Fine downsample
    arr_downsmpl2 = downsmpl_fine_arr(
        arr_downsmpl1, rp["z_fine"], rp["y_fine"], rp["x_fine"]
    )
    # Saving
    tifffile.imwrite(proj_fp_dict["downsmpl2"], arr_downsmpl2)


# @flow
def prepare_img_trim(
    proj_fp_dict: dict,
    z_trim: Optional[slice] = None,
    y_trim: Optional[slice] = None,
    x_trim: Optional[slice] = None,
):
    # Update registration params json
    rp = read_json(proj_fp_dict["reg_params"])
    rp["z_trim"] = [z_trim.start, z_trim.stop] if z_trim else rp["z_trim"]
    rp["y_trim"] = [y_trim.start, y_trim.stop] if y_trim else rp["y_trim"]
    rp["x_trim"] = [x_trim.start, x_trim.stop] if x_trim else rp["x_trim"]
    write_json(proj_fp_dict["reg_params"], rp)
    z_trim = slice(rp["z_trim"][0], rp["z_trim"][1])
    y_trim = slice(rp["y_trim"][0], rp["y_trim"][1])
    x_trim = slice(rp["x_trim"][0], rp["x_trim"][1])
    # Reading
    arr_downsmpl2 = tifffile.imread(proj_fp_dict["downsmpl2"])
    # Trim
    arr_trimmed = arr_downsmpl2[z_trim, y_trim, x_trim]
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
        ref_z_trim=slice(None, None),
        ref_y_trim=slice(None, None),
        ref_x_trim=slice(None, None),
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
        z_trim=slice(None, -5),
        y_trim=slice(80, -75),
        x_trim=slice(None, None),
    )

    # Running Elastix registration
    registration(
        fixed_img_fp=proj_fp_dict["trimmed"],
        moving_img_fp=proj_fp_dict["ref"],
        output_img_fp=proj_fp_dict["regresult"],
        affine_fp=proj_fp_dict["affine"],
        bspline_fp=proj_fp_dict["bspline"],
    )
