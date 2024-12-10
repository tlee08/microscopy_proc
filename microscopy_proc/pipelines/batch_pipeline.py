import logging
import os

from natsort import natsorted

from microscopy_proc.pipelines.pipeline_funcs import PipelineFuncs
from microscopy_proc.utils.proj_org_utils import (
    get_proj_fp_model,
    update_configs,
)

if __name__ == "__main__":
    # Filenames
    # atlas_rsc_dir = "/home/linux1/Desktop/iDISCO/resources/atlas_resources/"
    in_root_dir = "/run/user/1000/gvfs/smb-share:server=shared.sydney.edu.au,share=research-data/PRJ-BowenLab/Experiments/2024/Other/2024_whole_brain_clearing_TS/KNX Aggression cohort 1 stitched TIF images for analysis"
    root_dir = "/run/user/1000/gvfs/smb-share:server=shared.sydney.edu.au,share=research-data/PRJ-BowenLab/Experiments/2024/Other/2024_whole_brain_clearing_TS/KNX_Aggression_cohort_1_analysed_images"
    # in_fp_dir and batch_proj_dir cannot be the same
    assert in_root_dir != root_dir

    overwrite = True

    # Get all experiments
    exp_ls = natsorted(os.listdir(in_root_dir))
    exp_ls = [i for i in exp_ls if os.path.isdir(os.path.join(in_root_dir, i))]

    for i in exp_ls:
        # Only runs given files
        if i not in [
            "example_img",
        ]:
            continue
        # Logging which file is being processed
        logging.info(f"Running: {i}")
        try:
            # Filenames
            in_fp = os.path.join(in_root_dir, i)
            proj_dir = os.path.join(root_dir, i)
            # Getting file paths
            pfm = get_proj_fp_model(proj_dir)

            # Making params json
            update_configs(
                pfm,
                # # REFERENCE
                # # RAW
                # # REGISTRATION
                ref_orient_ls=(-2, 3, 1),
                # ref_z_trim=(None, None, None),
                # ref_y_trim=(None, None, None),
                # ref_x_trim=(None, None, None),
                # z_rough=3,
                # y_rough=6,
                # x_rough=6,
                # z_fine=1,
                # y_fine=0.6,
                # x_fine=0.6,
                # z_trim=(None, None, None),
                # y_trim=(None, None, None),
                # x_trim=(None, None, None),
                # # MASK
                # # OVERLAP
                # # CELL COUNTING
                # tophat_sigma=10,
                # dog_sigma1=1,
                # dog_sigma2=4,
                # gauss_sigma=101,
                # thresh_p=60,
                # min_threshd=100,
                # max_threshd=9000,
                # maxima_sigma=10,
                # min_wshed=1,
                # max_wshed=700,
            )

            # Making zarr from tiff file(s)
            PipelineFuncs.tiff2zarr(pfm, in_fp, overwrite=overwrite)
            # Preparing reference images
            PipelineFuncs.ref_prepare(pfm, overwrite=overwrite)
            # Preparing image itself
            PipelineFuncs.img_rough(pfm, overwrite=overwrite)
            PipelineFuncs.img_fine(pfm, overwrite=overwrite)
            PipelineFuncs.img_trim(pfm, overwrite=overwrite)
            # Running Elastix registration
            PipelineFuncs.elastix_registration(pfm, overwrite=overwrite)
            # Running mask pipeline
            PipelineFuncs.make_mask(pfm, overwrite=overwrite)
            # Making overlap chunks in preparation for cell counting
            PipelineFuncs.img_overlap(pfm, overwrite=overwrite)
            # Counting cells
            PipelineFuncs.cellc1(pfm, overwrite=overwrite)
            PipelineFuncs.cellc2(pfm, overwrite=overwrite)
            PipelineFuncs.cellc3(pfm, overwrite=overwrite)
            PipelineFuncs.cellc4(pfm, overwrite=overwrite)
            PipelineFuncs.cellc5(pfm, overwrite=overwrite)
            PipelineFuncs.cellc6(pfm, overwrite=overwrite)
            PipelineFuncs.cellc7(pfm, overwrite=overwrite)
            PipelineFuncs.cellc8(pfm, overwrite=overwrite)
            PipelineFuncs.cellc9(pfm, overwrite=overwrite)
            PipelineFuncs.cellc10(pfm, overwrite=overwrite)
            PipelineFuncs.cellc11(pfm, overwrite=overwrite)
            # Converting maxima from raw space to refernce atlas space
            PipelineFuncs.transform_coords(pfm, overwrite=overwrite)
            # Getting Region ID mappings for each cell
            PipelineFuncs.cell_mapping(pfm, overwrite=overwrite)
            # Grouping cells
            PipelineFuncs.group_cells(pfm, overwrite=overwrite)
            # Exporting cells_agg parquet as csv
            PipelineFuncs.cells2csv(pfm, overwrite=overwrite)
            # Making points and heatmap images
            PipelineFuncs.coords2points_raw(pfm, overwrite=overwrite)
            PipelineFuncs.coords2points_trfm(pfm, overwrite=overwrite)
            PipelineFuncs.coords2heatmap_trfm(pfm, overwrite=overwrite)
            # Combining arrays
            PipelineFuncs.combine_reg(pfm, overwrite=overwrite)
            PipelineFuncs.combine_cellc(pfm, overwrite=overwrite)
            PipelineFuncs.combine_points(pfm, overwrite=overwrite)
        except Exception as e:
            logging.info(f"Error in {i}: {e}")
            continue
