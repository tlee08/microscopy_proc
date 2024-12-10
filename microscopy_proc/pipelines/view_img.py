from microscopy_proc.funcs.viewer_funcs import CMAP, IMGS, VRANGE, view_arrs
from microscopy_proc.utils.proj_org_utils import get_proj_fp_model

if __name__ == "__main__":
    # Filenames
    in_fp = "/path/to/tiff_img_folder"
    proj_dir = "/path/to/analysis_output_folder"
    # Trimmer
    trimmer = (
        # slice(600, 650, None),
        # slice(1400, 3100, None),
        # slice(500, 3100, None),
        slice(None, None, None),
        slice(None, None, None),
        slice(None, None, None),
    )
    # Imgs to run
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

    pfm = get_proj_fp_model(proj_dir)
    # Making parameter lists. Index i refers to the same image
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
    # Running the Napari viewer
    view_arrs(
        fp_ls=tuple(fp_ls),
        trimmer=trimmer,
        name=tuple(name),
        contrast_limits=tuple(contrast_limits),
        colormap=tuple(colormap),
    )
