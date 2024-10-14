import os

from natsort import natsorted

from microscopy_proc.funcs.viewer_funcs import CMAP_D, VIEWER_IMGS, VRANGE_D, view_arrs
from microscopy_proc.utils.proj_org_utils import (
    get_proj_fp_model,
)

if __name__ == "__main__":
    # Filenames
    root_dir = "/run/user/1000/gvfs/smb-share:server=shared.sydney.edu.au,share=research-data/PRJ-BowenLab/Experiments/2024/Other/2024_whole_brain_clearing_TS/KNX_Aggression_cohort_1_analysed_images"

    overwrite = False

    # Get all experiments
    exp_ls = natsorted(os.listdir(root_dir))
    exp_ls = [i for i in exp_ls if os.path.isdir(os.path.join(root_dir, i))]

    for i in exp_ls:
        # Only given files
        if i not in [
            "B3_2.5x_1x_zoom_08082024",
            # "B9_2.5x_1x_zoom_06082024",
            # "G5_agg_2.5x_1xzoom_05072024",
            # "G8_2.5x_1x_zoom_08082024",
            # "G13_2.5x_1x_zoom_07082024",
        ]:
            continue
        proj_dir = os.path.join(root_dir, i)
        pfm = get_proj_fp_model(proj_dir)

        trimmer = (
            slice(750, 760, None),
            slice(None, None, None),
            slice(None, None, None),
            # slice(None, None, None),
            # slice(None, None, None),
            # slice(None, None, None),
        )

        imgs_to_run_dict = {
            "Atlas": [
                # "ref",
                # "annot",
            ],
            "Raw": [
                "raw",
            ],
            "Registration": [
                # "downsmpl1",
                # "downsmpl2",
                # "trimmed",
                # "regresult",
            ],
            "Mask": [
                # "premask_blur",
                # "mask",
                # "outline",
                # "mask_reg",
            ],
            "Cell Counting (overlapped)": [
                # "overlap",
                # "bgrm",
                # "dog",
                # "adaptv",
                # "threshd",
                # "threshd_volumes",
                # "threshd_filt",
                # "maxima",
                # "wshed_volumes",
                # "wshed_filt",
            ],
            "Cell Counting (trimmed)": [
                # "threshd_final",
                "maxima_final",
                "wshed_final",
            ],
            "Post Processing Checks": [
                # "points_check",
                # "heatmap_check",
                # "points_trfm_check",
                # "heatmap_trfm_check",
            ],
        }

        fp_ls = []
        name = []
        contrast_limits = []
        colormap = []
        for group_k, group_v in imgs_to_run_dict.items():
            for img_i in group_v:
                fp_ls.append(getattr(pfm, img_i))
                name.append(img_i)
                contrast_limits.append(VIEWER_IMGS[group_k][img_i][VRANGE_D])
                colormap.append(VIEWER_IMGS[group_k][img_i][CMAP_D])

        view_arrs(
            fp_ls=tuple(fp_ls),
            trimmer=trimmer,
            name=tuple(name),
            contrast_limits=tuple(contrast_limits),
            colormap=tuple(colormap),
        )
