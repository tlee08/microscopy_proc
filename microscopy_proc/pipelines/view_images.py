import os

from microscopy_proc.utils.proj_org_utils import get_proj_fp_model

if __name__ == "__main__":
    # Filenames
    # proj_dir = "/home/linux1/Desktop/A-1-1/large_cellcount"
    # proj_dir = "/home/linux1/Desktop/A-1-1/cellcount"
    proj_dir = "/run/user/1000/gvfs/smb-share:server=shared.sydney.edu.au,share=research-data/PRJ-BowenLab/Experiments/2024/Other/2024_whole_brain_clearing_TS/KNX_Aggression_cohort_1_analysed_images"
    proj_dir = os.path.join(proj_dir, "P15_2.5x_1x_zoom_07082024")

    proj_dir = "/home/linux1/Desktop/example_proj"

    pfm = get_proj_fp_model(proj_dir)

    trimmer = (
        slice(600, 650, None),
        slice(1400, 3100, None),
        slice(500, 3100, None),
        # slice(None, None, None),
        # slice(None, None, None),
        # slice(None, None, None),
    )

    imgs_to_run_dict = {
        "Atlas": [
            "ref",
            "annot",
        ],
        "Raw": [
            "raw",
        ],
        "Registration": [
            "downsmpl1",
            "downsmpl2",
            "trimmed",
            "regresult",
        ],
        "Mask": [
            "premask_blur",
            "mask",
            "outline",
            "mask_reg",
        ],
        "Cell Counting (overlapped)": [
            "overlap",
            "bgrm",
            "dog",
            "adaptv",
            "threshd",
            "threshd_volumes",
            "threshd_filt",
            "maxima",
            "wshed_volumes",
            "wshed_filt",
        ],
        "Cell Counting (trimmed)": [
            "threshd_final",
            "maxima_final",
            "wshed_final",
        ],
        "Post Processing Checks": [
            "points_check",
            "heatmap_check",
            "points_trfm_check",
            "heatmap_trfm_check",
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
            contrast_limits.append(IMGS[group_k][img_i][VRANGE])
            colormap.append(IMGS[group_k][img_i][CMAP])

    view_arrs(
        fp_ls=tuple(fp_ls),
        trimmer=trimmer,
        name=tuple(name),
        contrast_limits=tuple(contrast_limits),
        colormap=tuple(colormap),
    )
