import shutil

import dask.array as da
import tifffile

from microscopy_proc.funcs.reg_funcs import (
    downsmpl_fine_arr,
    downsmpl_rough_arr,
    reorient_arr,
)
from microscopy_proc.utils.elastix_utils import registration
from microscopy_proc.utils.proj_org_utils import (
    get_proj_fp_dict,
    get_ref_fp_dict,
    make_proj_dirs,
)

# logging.basicConfig(level=logging.DEBUG)


def prepare_ref(ref_fp_dict, proj_fp_dict, orient_ls, z_trim, y_trim, x_trim):
    # Making atlas images
    for fp_i, fp_o in [
        (ref_fp_dict["ref"], proj_fp_dict["ref"]),
        (ref_fp_dict["annot"], proj_fp_dict["annot"]),
    ]:
        arr = tifffile.imread(fp_i)
        # Reorienting
        arr = reorient_arr(arr, orient_ls)
        # Slicing
        arr = arr[z_trim, y_trim, x_trim]
        tifffile.imwrite(fp_o, arr)
    # Copying region mapping json to project folder
    shutil.copyfile(ref_fp_dict["map"], proj_fp_dict["map"])
    # Copying transformation files
    shutil.copyfile(ref_fp_dict["affine"], proj_fp_dict["affine"])
    shutil.copyfile(ref_fp_dict["bspline"], proj_fp_dict["bspline"])


def prepare_img_rough(proj_fp_dict, z_rough, y_rough, x_rough):
    arr_raw = da.from_zarr(proj_fp_dict["raw"])
    # Rough downsample
    arr_downsmpl1 = downsmpl_rough_arr(arr_raw, z_rough, y_rough, x_rough).compute()
    tifffile.imwrite(proj_fp_dict["downsmpl1"], arr_downsmpl1)


def prepare_img_fine(proj_fp_dict, z_fine, y_fine, x_fine):
    arr_downsmpl1 = tifffile.imread(proj_fp_dict["downsmpl1"])
    # Fine downsample
    arr_downsmpl2 = downsmpl_fine_arr(arr_downsmpl1, z_fine, y_fine, x_fine)
    tifffile.imwrite(proj_fp_dict["downsmpl2"], arr_downsmpl2)


def prepare_img_trim(proj_fp_dict, z_trim, y_trim, x_trim):
    arr_downsmpl2 = tifffile.imread(proj_fp_dict["downsmpl2"])
    # Trim
    arr_trimmed = arr_downsmpl2[z_trim, y_trim, x_trim]
    tifffile.imwrite(proj_fp_dict["trimmed"], arr_trimmed)


def prepare_img(
    proj_fp_dict,
    z_rough,
    y_rough,
    x_rough,
    z_fine,
    y_fine,
    x_fine,
    z_trim,
    y_trim,
    x_trim,
):
    arr_raw = da.from_zarr(proj_fp_dict["raw"])
    # Rough downsample
    arr_downsmpl1 = downsmpl_rough_arr(arr_raw, z_rough, y_rough, x_rough).compute()
    # Fine downsample
    arr_downsmpl2 = downsmpl_fine_arr(arr_downsmpl1, z_fine, y_fine, x_fine)
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
        orient_ls=(2, 3, 1),
        z_trim=slice(None, None),
        y_trim=slice(None, None),
        x_trim=slice(None, None),
    )

    # Preparing image itself
    prepare_img(
        proj_fp_dict=proj_fp_dict,
        z_rough=0.3,
        y_rough=0.1,
        x_rough=0.1,
        z_fine=0.8,
        y_fine=0.8,
        x_fine=0.8,
        z_trim=slice(None, -5),
        y_trim=slice(60, -50),
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
