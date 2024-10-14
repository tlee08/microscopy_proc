import logging
import os

from natsort import natsorted

from microscopy_proc.pipelines.pipeline_funcs import (
    cell_mapping_pipeline,
    cellc1_pipeline,
    cellc2_pipeline,
    cellc3_pipeline,
    cellc4_pipeline,
    cellc5_pipeline,
    cellc6_pipeline,
    cellc7_pipeline,
    cellc8_pipeline,
    cellc9_pipeline,
    cellc10_pipeline,
    cellc11_pipeline,
    cells2csv_pipeline,
    group_cells_pipeline,
    img_fine_pipeline,
    img_overlap_pipeline,
    img_rough_pipeline,
    img_trim_pipeline,
    make_mask_pipeline,
    ref_prepare_pipeline,
    registration_pipeline,
    tiff2zarr_pipeline,
    transform_coords_pipeline,
)
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

    overwrite = False

    # Get all experiments
    exp_ls = natsorted(os.listdir(in_root_dir))
    exp_ls = [i for i in exp_ls if os.path.isdir(os.path.join(in_root_dir, i))]

    for i in exp_ls:
        # Only given files
        if i not in [
            "B3_2.5x_1x_zoom_08082024",
            "B9_2.5x_1x_zoom_06082024",
            "G5_agg_2.5x_1xzoom_05072024",
            "G8_2.5x_1x_zoom_08082024",
            "G13_2.5x_1x_zoom_07082024",
        ]:
            continue
        # Logging which file is being processed
        print(f"Running: {i}")
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
                # REFERENCE
                # RAW
                # REGISTRATION
                ref_orient_ls=(-2, 3, 1),
                ref_z_trim=(None, None, None),
                ref_y_trim=(None, None, None),
                ref_x_trim=(None, None, None),
                z_rough=3,
                y_rough=6,
                x_rough=6,
                z_fine=1,
                y_fine=0.6,
                x_fine=0.6,
                z_trim=(None, None, None),
                y_trim=(None, None, None),
                x_trim=(None, None, None),
                # MASK
                # OVERLAP
                # CELL COUNTING
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

            # Making zarr from tiff file(s)
            tiff2zarr_pipeline(pfm, in_fp, overwrite=overwrite)
            # Preparing reference images
            ref_prepare_pipeline(pfm, overwrite=overwrite)
            # Preparing image itself
            img_rough_pipeline(pfm, overwrite=overwrite)
            img_fine_pipeline(pfm, overwrite=overwrite)
            img_trim_pipeline(pfm, overwrite=overwrite)
            # Running Elastix registration
            registration_pipeline(pfm)
            # Running mask pipeline
            make_mask_pipeline(pfm)
            # Making overlap chunks in preparation for cell counting
            img_overlap_pipeline(pfm)
            # Counting cells
            cellc1_pipeline(pfm, overwrite=overwrite)
            cellc2_pipeline(pfm, overwrite=overwrite)
            cellc3_pipeline(pfm, overwrite=overwrite)
            cellc4_pipeline(pfm, overwrite=overwrite)
            cellc5_pipeline(pfm, overwrite=overwrite)
            cellc6_pipeline(pfm, overwrite=overwrite)
            cellc7_pipeline(pfm, overwrite=overwrite)
            cellc8_pipeline(pfm, overwrite=overwrite)
            cellc9_pipeline(pfm, overwrite=overwrite)
            cellc10_pipeline(pfm, overwrite=overwrite)
            cellc11_pipeline(pfm, overwrite=overwrite)
            # Converting maxima from raw space to refernce atlas space
            transform_coords_pipeline(pfm, overwrite=overwrite)
            # Getting Region ID mappings for each cell
            cell_mapping_pipeline(pfm, overwrite=overwrite)
            # Grouping cells
            group_cells_pipeline(pfm, overwrite=overwrite)
            # Exporting cells_agg parquet as csv
            cells2csv_pipeline(pfm, overwrite=overwrite)

            print()
        except Exception as e:
            logging.info(f"Error in {i}: {e}")
            continue
