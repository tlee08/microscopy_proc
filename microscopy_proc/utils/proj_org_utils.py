import os


def get_ref_fp_dict(atlas_dir, ref_v=None, annot_v=None, map_v=None):
    # atlas_rsc_dir = "/home/linux1/Desktop/iDISCO/resources/atlas_resources/"
    # Atlas from https://download.alleninstitute.org/informatics-archive/current-release/mouse_ccf/

    # NAMES OF ATLAS FILES USED
    # "average_template_25", "ara_nissl_25"
    ref_v = "average_template_25" if ref_v is None else ref_v
    # "ccf_2017_25", "ccf_2016_25", "ccf_2015_25"
    annot_v = "ccf_2016_25" if annot_v is None else ref_v
    # "ABA_annotations", "CM_annotations"
    map_v = "ABA_annotations" if map_v is None else ref_v

    return {
        # ATLAS FILES ORIGIN
        "ref": os.path.join(atlas_dir, "reference", f"{ref_v}.tif"),
        "annot": os.path.join(atlas_dir, "annotation", f"{annot_v}.tif"),
        "map": os.path.join(atlas_dir, "region_mapping", f"{map_v}.json"),
        # ELASTIX PARAM FILES ORIGIN
        "affine": os.path.join(atlas_dir, "elastix_params", "align_affine.txt"),
        "bspline": os.path.join(atlas_dir, "elastix_params", "align_bspline.txt"),
    }


def get_proj_fp_dict(proj_dir):
    return {
        # MY ATLAS AND ELASTIX PARAMS FILES
        "ref": os.path.join(proj_dir, "registration", "0a_reference.tif"),
        "annot": os.path.join(proj_dir, "registration", "0b_annotation.tif"),
        "map": os.path.join(proj_dir, "registration", "0c_mapping.json"),
        "affine": os.path.join(proj_dir, "registration", "0d_align_affine.txt"),
        "bspline": os.path.join(proj_dir, "registration", "0e_align_bspline.txt"),
        # RAW IMG FILE
        "raw": os.path.join(proj_dir, "raw.zarr"),
        # REGISTRATION PROCESSING FILES
        "downsmpl1": os.path.join(proj_dir, "registration", "1_downsmpl1.tif"),
        "downsmpl2": os.path.join(proj_dir, "registration", "2_downsmpl2.tif"),
        "trimmed": os.path.join(proj_dir, "registration", "3_trimmed.tif"),
        "regresult": os.path.join(proj_dir, "registration", "4_regresult.tif"),
        # CELL COUNTING FILES
        "overlap": os.path.join(proj_dir, "cellcount", "0_overlap.zarr"),
        "bgrm": os.path.join(proj_dir, "cellcount", "1_bgrm.zarr"),
        "dog": os.path.join(proj_dir, "cellcount", "2_dog.zarr"),
        "adaptv": os.path.join(proj_dir, "cellcount", "3_adaptv.zarr"),
        "threshd": os.path.join(proj_dir, "cellcount", "4_threshd.zarr"),
        "sizes": os.path.join(proj_dir, "cellcount", "5_sizes.zarr"),
        "filt": os.path.join(proj_dir, "cellcount", "6_filt.zarr"),
        "maxima": os.path.join(proj_dir, "cellcount", "7_maxima.zarr"),
        "watershed": os.path.join(proj_dir, "cellcount", "8_watershed.zarr"),
        "filt_final": os.path.join(proj_dir, "cellcount", "9_filt_f.zarr"),
        "maxima_final": os.path.join(proj_dir, "cellcount", "9_maxima_f.zarr"),
        # POST PROC FILES
        "region_df": os.path.join(proj_dir, "analysis", "10_region.parquet"),
        "maxima_df": os.path.join(proj_dir, "analysis", "10_maxima.parquet"),
    }


def make_proj_dirs(proj_dir):
    os.makedirs(os.path.join(proj_dir, "registration"), exist_ok=True)
    os.makedirs(os.path.join(proj_dir, "cellcount"), exist_ok=True)
    os.makedirs(os.path.join(proj_dir, "analysis"), exist_ok=True)