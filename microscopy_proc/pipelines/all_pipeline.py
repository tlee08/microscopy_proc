import os

from microscopy_proc.constants import DEPTH, PROC_CHUNKS
from microscopy_proc.pipelines.cellc_pipeline import (
    img_overlap_pipeline,
    img_proc_pipeline,
)
from microscopy_proc.pipelines.make_zarr import tiff2zarr
from microscopy_proc.pipelines.map_pipeline import (
    cells2csv,
    get_cell_mappings,
    grouping_cells,
    transform_coords,
)
from microscopy_proc.pipelines.reg_pipeline import (
    img_fine_pipeline,
    img_rough_pipeline,
    img_trim_pipeline,
    ref_prepare_pipeline,
    registration_pipeline,
)
from microscopy_proc.utils.proj_org_utils import (
    get_proj_fp_model,
    get_ref_fp_model,
    init_configs,
    make_proj_dirs,
)

if __name__ == "__main__":
    # Filenames
    # proj_dir = "/home/linux1/Desktop/A-1-1/cellcount"
    # in_fp = "/home/linux1/Desktop/A-1-1/cropped abcd_larger.tif"
    # in_fp = "/home/linux1/Desktop/A-1-1/example"
    # proj_dir = "/home/linux1/Desktop/A-1-1/large_cellcount"
    in_fp = "/run/user/1000/gvfs/smb-share:server=shared.sydney.edu.au,share=research-data/PRJ-BowenLab/Experiments/2024/Other/2024_whole_brain_clearing_TS/KNX Aggression cohort 1 stitched TIF images for analysis"
    proj_dir = "/run/user/1000/gvfs/smb-share:server=shared.sydney.edu.au,share=research-data/PRJ-BowenLab/Experiments/2024/Other/2024_whole_brain_clearing_TS/KNX_Aggression_cohort_1_analysed_images"
    exp_name = "G17_2.5x_1x_zoom_07082024"

    in_fp = os.path.join(in_fp, exp_name)
    proj_dir = os.path.join(proj_dir, exp_name)

    # atlas_rsc_dir = "/home/linux1/Desktop/iDISCO/resources/atlas_resources/"
    rfm = get_ref_fp_model()
    pfm = get_proj_fp_model(proj_dir)
    # Making project folders
    make_proj_dirs(proj_dir)

    # Making params json
    init_configs(pfm)

    # Making zarr from tiff file(s)
    tiff2zarr(in_fp, pfm.raw, chunks=PROC_CHUNKS)

    # Preparing reference images
    ref_prepare_pipeline(
        rfm=rfm,
        pfm=pfm,
        ref_orient_ls=(-2, 3, 1),
        ref_z_trim=(None, None, None),
        ref_y_trim=(None, -110, None),
        ref_x_trim=(None, None, None),
    )

    # Preparing image itself
    img_rough_pipeline(
        pfm,
        z_rough=3,
        y_rough=6,
        x_rough=6,
    )
    img_fine_pipeline(
        pfm,
        z_fine=1,
        y_fine=0.6,
        x_fine=0.6,
    )
    img_trim_pipeline(
        pfm,
        # z_trim=(None, -5, None),
        # y_trim=(80, -75, None),
        # x_trim=(None, None, None),
    )

    # Running Elastix registration
    registration_pipeline(pfm)

    img_overlap_pipeline(pfm, chunks=PROC_CHUNKS, d=DEPTH)

    img_proc_pipeline(
        pfm=pfm,
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
    transform_coords(pfm)

    get_cell_mappings(pfm)

    grouping_cells(pfm)

    cells2csv(pfm)
