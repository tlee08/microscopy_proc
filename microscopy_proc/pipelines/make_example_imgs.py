import os

from natsort import natsorted

from microscopy_proc.funcs.viewer_funcs import combine_arrs
from microscopy_proc.pipelines.pipeline_funcs import (
    coords2heatmap_trfm_pipeline,
    coords2points_raw_pipeline,
    coords2points_trfm_pipeline,
)
from microscopy_proc.utils.proj_org_utils import (
    get_proj_fp_model,
)

# TODO implement propper export function

if __name__ == "__main__":
    # Filenames
    root_dir = "/run/user/1000/gvfs/smb-share:server=shared.sydney.edu.au,share=research-data/PRJ-BowenLab/Experiments/2024/Other/2024_whole_brain_clearing_TS/KNX_Aggression_cohort_1_analysed_images"
    # out_dir = "/home/linux1/Desktop/imgs_for_nick"

    overwrite = False

    trimmer = (
        slice(750, 760, None),
        slice(None, None, None),
        slice(None, None, None),
    )

    # Get all experiments
    exp_ls = natsorted(os.listdir(root_dir))
    exp_ls = [i for i in exp_ls if os.path.isdir(os.path.join(root_dir, i))]

    for i in exp_ls:
        print(i)

        proj_dir = os.path.join(root_dir, i)
        pfm = get_proj_fp_model(proj_dir)

        out_dir = os.path.join(proj_dir, "example_imgs")
        os.makedirs(out_dir, exist_ok=True)

        # # Only given files
        # if i not in [
        #     "B3_2.5x_1x_zoom_08082024",
        #     "G1_reimage_agg_2.5x_1xzoom_05072024",
        #     "P8_2.5x_1x_zoom_07082024",
        #     "R14_agg_2.5x_1xzoom_02072024",
        # ]:
        #     continue

        # Making points and heatmap images
        coords2points_raw_pipeline(pfm)
        coords2points_trfm_pipeline(pfm)
        coords2heatmap_trfm_pipeline(pfm)

        # COMBINING ARRAYS (ZYXC)
        # Combining reg
        combine_arrs(
            fp_in_ls=(pfm.trimmed, pfm.regresult, pfm.regresult),
            # 2nd regresult means the combining works in ImageJ
            fp_out=os.path.join(out_dir, "combined_reg.tif"),
        )
        # Combining cellc
        combine_arrs(
            fp_in_ls=(pfm.raw, pfm.maxima_final, pfm.wshed_final),
            fp_out=os.path.join(out_dir, "combined_cellc.tif"),
            trimmer=trimmer,
        )
        # Combining transformed points
        combine_arrs(
            fp_in_ls=(pfm.ref, pfm.annot, pfm.heatmap_trfm),
            fp_out=os.path.join(out_dir, "combined_points.tif"),
        )
