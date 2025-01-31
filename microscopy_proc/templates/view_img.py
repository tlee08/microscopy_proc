from microscopy_proc.funcs.viewer_funcs import ViewerFuncs
from microscopy_proc.pipeline.pipeline import Pipeline

if __name__ == "__main__":
    # Filenames
    proj_dir = "/path/to/analysis_output_folder"
    # Trimmer
    trimmer = [
        slice(None, None, None),
        slice(None, None, None),
        slice(None, None, None),
    ]
    # Images to run
    # COMMENT OUT THE IMAGES THAT YOU DON'T WANT TO VIEW
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
    # Making the project file model
    pfm = Pipeline.get_pfm(proj_dir)
    # Viewing the images
    ViewerFuncs.view_arrs_from_pfm(pfm, imgs_to_run_ls, trimmer)
