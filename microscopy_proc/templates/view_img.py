from microscopy_proc.funcs.viewer_funcs import CMAP, VRANGE, ViewerFuncs, imgs_view_params
from microscopy_proc.pipeline.pipeline import Pipeline
import asyncio

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
    imgs_to_run_ls = [
        "ref",
        "annot",
        "raw",
        "downsmpl1",
        "downsmpl2",
        "trimmed",
        "bounded",
        "regresult",
        "premask_blur",
        "mask",
        "outline",
        "mask_reg",
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
        "threshd_final",
        "maxima_final",
        "wshed_final",
        "points_check",
        "heatmap_check",
        "points_trfm_check",
        "heatmap_trfm_check",
    ]

    pfm = Pipeline.get_pfm(proj_dir)
    # Making parameter lists. Index i refers to the same image
    fp_ls = []
    name = []
    contrast_limits = []
    colormap = []
    for img_to_run in imgs_to_run_ls:
        fp_ls.append(getattr(pfm, img_to_run).val)
        name.append(img_to_run)
        contrast_limits.append(imgs_view_params[img_to_run][VRANGE])
        colormap.append(imgs_view_params[img_to_run][CMAP])
    # Running the Napari viewer
    await ViewerFuncs.view_arrs(
        fp_ls=tuple(fp_ls),
        trimmer=trimmer,
        name=tuple(name),
        contrast_limits=tuple(contrast_limits),
        colormap=tuple(colormap),
    )
