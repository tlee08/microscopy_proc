import numpy as np
import pandas as pd
import tifffile

from microscopy_proc.funcs.elastix_funcs import transformation_coords
from microscopy_proc.funcs.gpu_arr_funcs import GpuArrFuncs as Gf
from microscopy_proc.funcs.visual_check_funcs import coords_to_points

# logging.basicConfig(level=logging.DEBUG)
# from prefect import flow
from microscopy_proc.utils.config_params_model import ConfigParamsModel
from microscopy_proc.utils.proj_org_utils import (
    get_proj_fp_dict,
    make_proj_dirs,
)
from microscopy_proc.viewer.image_viewer import view_imgs


def make_outline(arr: np.ndarray) -> pd.DataFrame:
    # Shifting along last axis
    l_shift = np.concatenate([arr[..., 1:], np.zeros((*arr.shape[:-1], 1))], axis=-1)
    r_shift = np.concatenate([np.zeros((*arr.shape[:-1], 1)), arr[..., :-1]], axis=-1)
    # Finding outline (ins and outs)
    coords_df = pd.concat(
        [
            pd.DataFrame(
                np.asarray(np.where((arr == 1) & (r_shift == 0))).T,
                columns=["z", "y", "x"],
            ).assign(is_in=1),
            pd.DataFrame(
                np.asarray(np.where((arr == 1) & (l_shift == 0))).T,
                columns=["z", "y", "x"],
            ).assign(is_in=0),
        ]
    )
    # Ordering by z, y, x, so fill outline works
    coords_df = coords_df.sort_values(by=["z", "y", "x"]).reset_index(drop=True)
    # Returning
    return coords_df


def fill_outline(arr: np.ndarray, coords_df: pd.DataFrame) -> np.ndarray:
    # Initialize mask
    res = np.zeros(arr.shape, dtype=np.uint8)
    # Checking that type is 0 or 1
    assert coords_df["is_in"].isin([0, 1]).all()
    # For each outline coord
    for i, x in coords_df.iterrows():
        # If type is 1, fill in
        if x["is_in"] == 1:
            res[x["z"], x["y"], x["x"] :] = 1
        # If type is 0, stop filling in
        elif x["is_in"] == 0:
            res[x["z"], x["y"], x["x"] + 1 :] = 0
    # Returning
    return res


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
    arr_trimmed = tifffile.imread(proj_fp_dict["trimmed"])

    arr_smoothed = Gf.gauss_blur_filt(arr_trimmed, 1)
    arr_mask = Gf.manual_thresh(arr_smoothed, 400)
    tifffile.imwrite(proj_fp_dict["mask"], arr_mask)

    # Make outline
    outline_df = make_outline(arr_mask)

    # Transformix on coords
    outline_trfm_df = transformation_coords(
        outline_df,
        proj_fp_dict["ref"],
        proj_fp_dict["regresult"],
    )
    outline_df[["z", "y", "x"]] = (
        outline_trfm_df[["z", "y", "x"]].round(0).astype(np.int32)
    )
    outline_df.to_parquet(proj_fp_dict["outline_df"])

    # Filtering out of bounds coords
    arr_ref = tifffile.imread(proj_fp_dict["ref"])
    shape = arr_ref.shape
    outline_df = pd.read_parquet(proj_fp_dict["outline_df"])
    outline_df = outline_df.query(
        f"z >= 0 and z < {shape[0]} and y >= 0 and y < {shape[1]} and x >= 0 and x < {shape[2]}"
    )
    # Make outline img
    coords_to_points(
        outline_trfm_df,
        arr_ref.shape,
        proj_fp_dict["outline"],
    )

    # Fill in outline
    arr_mask_reg = fill_outline(arr_ref, outline_df)
    # Make warped mask img
    tifffile.imwrite(proj_fp_dict["mask_reg"], arr_mask_reg)

    # Convert 3d tensor to (z, y, x) coords
    # outline_df = Gf.get_coords(arr_outline)

    # # Transformix
    # outline_trfm_df = transformation_coords(
    #     Gf.get_coords(arr_outline),
    #     proj_fp_dict["ref"],
    #     proj_fp_dict["regresult"],
    # )
    # outline_trfm_df.to_parquet(proj_fp_dict["outline_df"])

    # # Coords to spatial
    # coords_to_points(
    #     outline_trfm_df,
    #     tifffile.imread(proj_fp_dict["ref"]).shape,
    #     proj_fp_dict["outline_reg"],
    # )

    # # Fill in outline
    # arr_outline_reg = tifffile.imread(proj_fp_dict["outline_reg"])

    view_imgs(
        [
            proj_fp_dict["ref"],
            # proj_fp_dict["trimmed"],
            # proj_fp_dict["smoothed"],
            proj_fp_dict["mask"],
            proj_fp_dict["outline"],
            # proj_fp_dict["outline_reg"],
            proj_fp_dict["mask_reg"],
        ],
        [5, 5, 5, 5, 5, 5],
        [slice(None, None), slice(None, None), slice(None, None)],
    )
