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
        # "raw",
        # "ref",
        # "annot",
        # "downsmpl1",
        # "downsmpl2",
        # "trimmed",
        # "bounded",
        # "regresult",
        # "premask_blur",
        # "mask_fill",
        # "mask_outline",
        # "mask_reg",
        "overlap",
        "bgrm",
        "dog",
        "adaptv",
        "threshd",
        "threshd_volumes",
        "threshd_filt",
        # "maxima",
        "wshed_volumes",
        "wshed_filt",
        # "threshd_final",
        # "maxima_final",
        # "wshed_final",
        # "points_raw",
        # "heatmap_raw",
        # "points_trfm",
        # "heatmap_trfm",
    ]
    # Making the project file model
    # pfm = Pipeline.get_pfm(proj_dir)
    pfm = Pipeline.get_pfm_tuning(proj_dir)
    # Viewing the images
    ViewerFuncs.view_arrs_from_pfm(pfm, imgs_to_run_ls, trimmer)
