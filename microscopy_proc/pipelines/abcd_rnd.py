import numpy as np
import pandas as pd
import tifffile

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

# TODO: JUST TAKE THE BRAIN MASK OF THE DOWNSAMPLED, TRIMMED IMAGE!!


def make_outline(arr: np.ndarray) -> pd.DataFrame:
    # arr_outline = np.zeros(arr.shape, np.uint8)
    # idx = [i.ravel() for i in np.meshgrid(*[np.arange(i) for i in arr.shape[:-1]])]

    l_shift = np.concatenate([arr[1:], np.zeros((1, *arr.shape[1:]))])
    r_shift = np.concatenate([np.zeros((1, *arr.shape[1:])), arr[:-1]])

    coords_df = pd.concat(
        [
            pd.DataFrame(
                np.asarray(np.where((arr == 1) & (r_shift == 0))).T,
                columns=["z", "y", "x"],
            ).assign(type="in"),
            pd.DataFrame(
                np.asarray(np.where((arr == 1) & (l_shift == 0))).T,
                columns=["z", "y", "x"],
            ).assign(type="out"),
        ]
    )

    # df_ls = np.full(idx[0].shape, None)
    # coords_df = pd.DataFrame(columns=["z", "y", "x"])
    # for i in zip(*idx):
    #     l_shift = np.concatenate([arr[*i, 1:], [0]])
    #     r_shift = np.concatenate([[0], arr[*i, :-1]])
    #     in_df = pd.DataFrame(np.where((arr[*i] == 1) & (r_shift == 0)), columns=["z", "y", "x"])
    #     out_df = pd.DataFrame(np.where((arr[*i] == 1) & (l_shift == 0)), columns=["z", "y", "x"])
    #     # arr_outline[*i] = arr[*i] - ndimage.binary_erosion(arr[*i])
    #     df_ls.append
    return coords_df


def fill_outline(arr):
    mask = np.zeros(arr.shape, dtype=np.uint8)
    idx = [i.ravel() for i in np.meshgrid(*[np.arange(i) for i in arr.shape[:-1]])]
    for i in zip(*idx):
        # cumsum of every seen cross (border) indicates in/out ROI
        mask[*i] = (np.cumsum(arr[*i]) % 2 == 1).astype(np.uint8)
        # Adding in borders
        mask[*i] += arr[*i] > 0
        mask[*i] = np.minimum(mask[*i], 1)
    return mask


# def fill_outline(coords_df: pd.DataFrame):
#     mask = np.zeros(arr.shape, dtype=np.uint8)
#     for i, x in coords_df.iterrows():
#         if x["type"] == "in":
#             mask[x["z"]:, x["y"], x["x"]] = 1
#         elif x["type"] == "out":
#             mask[x["z"]+1:, x["y"], x["x"]] = 0
#         else:
#             print("WRONG TYPE:", x["type"])


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
    tifffile.imwrite(proj_fp_dict["smoothed"], arr_smoothed)

    arr_mask = Gf.manual_thresh(arr_smoothed, 400)
    tifffile.imwrite(proj_fp_dict["mask"], arr_mask)

    arr_mask_l = np.concatenate([arr_mask[1:], np.zeros((1, *arr_mask.shape[1:]))])
    tifffile.imwrite(proj_fp_dict["mask_l"], arr_mask_l)
    arr_mask_r = np.concatenate([np.zeros((1, *arr_mask.shape[1:])), arr_mask[:-1]])
    tifffile.imwrite(proj_fp_dict["mask_r"], arr_mask_r)

    outline_df = make_outline(arr_mask)
    outline_df.to_parquet(proj_fp_dict["outline_df"])

    arr_outline = coords_to_points(
        outline_df,
        arr_mask.shape,
        proj_fp_dict["outline"],
    )

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
            # proj_fp_dict["ref"],
            # proj_fp_dict["trimmed"],
            # proj_fp_dict["smoothed"],
            proj_fp_dict["mask"],
            proj_fp_dict["outline"],
            # proj_fp_dict["outline_reg"],
        ],
        [5, 5, 5, 5, 5, 5],
        [slice(None, None), slice(None, None), slice(None, None)],
    )
