from microscopy_proc.constants import DEPTH, PROC_CHUNKS
from microscopy_proc.funcs.elastix_funcs import registration
from microscopy_proc.pipelines.cellc_pipeline import (
    img_overlap_pipeline,
    img_proc_pipeline,
)
from microscopy_proc.pipelines.make_zarr import tiff_to_zarr
from microscopy_proc.pipelines.map_pipeline import (
    cells2csv,
    get_cell_mappings,
    grouping_cells,
    transform_coords,
)
from microscopy_proc.pipelines.reg_pipeline import (
    prepare_img_fine,
    prepare_img_rough,
    prepare_img_trim,
    prepare_ref,
)
from microscopy_proc.utils.proj_org_utils import (
    get_proj_fp_dict,
    get_ref_fp_dict,
    init_params,
    make_proj_dirs,
)

if __name__ == "__main__":
    # Filenames
    # proj_dir = "/home/linux1/Desktop/A-1-1/cellcount"
    # in_fp = "/home/linux1/Desktop/A-1-1/cropped abcd_larger.tif"
    # in_fp = "/home/linux1/Desktop/A-1-1/example"
    # proj_dir = "/home/linux1/Desktop/A-1-1/large_cellcount"
    in_fp = "/run/user/1000/gvfs/smb-share:server=shared.sydney.edu.au,share=research-data/PRJ-BowenLab/Experiments/2024/Other/2024_whole_brain_clearing_TS/KNX Aggression cohort 1 stitched TIF images for analysis/B15_agg_2.5x_1xzoom_03072024"
    proj_dir = "/run/user/1000/gvfs/smb-share:server=shared.sydney.edu.au,share=research-data/PRJ-BowenLab/Experiments/2024/Other/2024_whole_brain_clearing_TS/KNX_Aggression_cohort_1_analysed_images/B15_agg_2.5x_1xzoom_03072024"

    # atlas_rsc_dir = "/home/linux1/Desktop/iDISCO/resources/atlas_resources/"
    ref_fp_dict = get_ref_fp_dict()
    proj_fp_dict = get_proj_fp_dict(proj_dir)
    # Making project folders
    make_proj_dirs(proj_dir)

    # Making params json
    init_params(proj_fp_dict)

    # Making zarr from tiff file(s)
    tiff_to_zarr(in_fp, proj_fp_dict["raw"], chunks=PROC_CHUNKS)

    # Preparing reference images
    prepare_ref(
        ref_fp_dict=ref_fp_dict,
        proj_fp_dict=proj_fp_dict,
        ref_orient_ls=(2, 3, 1),
        ref_z_trim=(None, None, None),
        ref_y_trim=(None, None, None),
        ref_x_trim=(None, None, None),
    )

    # Preparing image itself
    prepare_img_rough(
        proj_fp_dict,
        z_rough=3,
        y_rough=6,
        x_rough=6,
    )
    prepare_img_fine(
        proj_fp_dict,
        z_fine=1,
        y_fine=0.6,
        x_fine=0.6,
    )
    prepare_img_trim(
        proj_fp_dict,
        z_trim=(None, -5, None),
        y_trim=(80, -75, None),
        x_trim=(None, None, None),
    )

    # Running Elastix registration
    registration(
        fixed_img_fp=proj_fp_dict["trimmed"],
        moving_img_fp=proj_fp_dict["ref"],
        output_img_fp=proj_fp_dict["regresult"],
        affine_fp=proj_fp_dict["affine"],
        bspline_fp=proj_fp_dict["bspline"],
    )

    img_overlap_pipeline(proj_fp_dict, chunks=PROC_CHUNKS, d=DEPTH)

    img_proc_pipeline(
        proj_fp_dict=proj_fp_dict,
        d=DEPTH,
        tophat_sigma=10,
        dog_sigma1=1,
        dog_sigma2=4,
        gauss_sigma=101,
        thresh_p=60,
        min_threshd=100,
        max_threshd=9000,
        maxima_sigma=10,
        min_wshed=1,
        max_wshed=700,
    )

    # Converting maxima from raw space to refernce atlas space
    transform_coords(proj_fp_dict)

    get_cell_mappings(proj_fp_dict)

    grouping_cells(proj_fp_dict)

    cells2csv(proj_fp_dict)
