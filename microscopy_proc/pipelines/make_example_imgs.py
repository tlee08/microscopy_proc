import os

from natsort import natsorted

from microscopy_proc.funcs.viewer_funcs import combine_arrs
from microscopy_proc.utils.proj_org_utils import (
    get_proj_fp_model,
)

# TODO implement propper export function

if __name__ == "__main__":
    # Filenames
    root_dir = "/run/user/1000/gvfs/smb-share:server=shared.sydney.edu.au,share=research-data/PRJ-BowenLab/Experiments/2024/Other/2024_whole_brain_clearing_TS/KNX_Aggression_cohort_1_analysed_images"
    out_dir = "/home/linux1/Desktop/imgs_for_nick"

    overwrite = False

    # Get all experiments
    exp_ls = natsorted(os.listdir(root_dir))
    exp_ls = [i for i in exp_ls if os.path.isdir(os.path.join(root_dir, i))]

    for i in exp_ls:
        print(i)

        proj_dir = os.path.join(root_dir, i)
        pfm = get_proj_fp_model(proj_dir)

        # Only given files
        if i not in [
            "B3_2.5x_1x_zoom_08082024",
            "G1_reimage_agg_2.5x_1xzoom_05072024",
            "P8_2.5x_1x_zoom_07082024",
            "R14_agg_2.5x_1xzoom_02072024",
        ]:
            continue

        trimmer = (
            slice(750, 760, None),
            slice(None, None, None),
            slice(None, None, None),
        )

        # Exporting
        os.makedirs(os.path.join(out_dir, i), exist_ok=True)
        # # trimmed
        # save_arr(pfm.trimmed, os.path.join(out_dir, i, "trimmed.tif"))
        # # regresult
        # save_arr(pfm.regresult, os.path.join(out_dir, i, "regresult.tif"))
        # # raw
        # save_arr(pfm.raw, os.path.join(out_dir, i, "raw.tif"), trimmer)
        # # maxima_final
        # save_arr(
        #     pfm.maxima_final, os.path.join(out_dir, i, "maxima_final.tif"), trimmer
        # )
        # # wshed_final
        # save_arr(pfm.wshed_final, os.path.join(out_dir, i, "wshed_final.tif"), trimmer)
        # # ref
        # save_arr(pfm.ref, os.path.join(out_dir, i, "ref.tif"))
        # # annot
        # save_arr(pfm.annot, os.path.join(out_dir, i, "annot.tif"))
        # heatmap_trfm
        # save_arr(pfm.heatmap_trfm, os.path.join(out_dir, i, "heatmap_trfm.tif"))

        # # COMBINING ARRAYS (ZYXC)
        # Combining reg
        combine_arrs(
            (
                os.path.join(out_dir, i, "trimmed.tif"),
                # 2nd means the combining works in ImageJ
                os.path.join(out_dir, i, "trimmed.tif"),
                os.path.join(out_dir, i, "regresult.tif"),
            ),
            os.path.join(out_dir, i, "combined_reg.tif"),
        )
        # # Combining cellc
        # combine_arrs(
        #     (
        #         os.path.join(out_dir, i, "raw.tif"),
        #         os.path.join(out_dir, i, "maxima_final.tif"),
        #         os.path.join(out_dir, i, "wshed_final.tif"),
        #     ),
        #     os.path.join(out_dir, i, "combined_cellc.tif"),
        # )
        # Combining transformed points
        # combine_arrs(
        #     (
        #         os.path.join(out_dir, i, "ref.tif"),
        #         os.path.join(out_dir, i, "annot.tif"),
        #         os.path.join(out_dir, i, "heatmap_trfm.tif"),
        #     ),
        #     os.path.join(out_dir, i, "combined_points.tif"),
        # )
