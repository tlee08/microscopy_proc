import logging
import os

from natsort import natsorted

from microscopy_proc.pipelines.pipeline_funcs import (
    cell_mapping_pipeline,
    cells2csv_pipeline,
    group_cells_pipeline,
    img_fine_pipeline,
    img_trim_pipeline,
    ref_prepare_pipeline,
    registration_pipeline,
    transform_coords_pipeline,
)
from microscopy_proc.utils.proj_org_utils import (
    get_proj_fp_model,
    make_proj_dirs,
)

if __name__ == "__main__":
    # Filenames
    # atlas_rsc_dir = "/home/linux1/Desktop/iDISCO/resources/atlas_resources/"
    in_fp_dir = "/run/user/1000/gvfs/smb-share:server=shared.sydney.edu.au,share=research-data/PRJ-BowenLab/Experiments/2024/Other/2024_whole_brain_clearing_TS/KNX Aggression cohort 1 stitched TIF images for analysis"
    batch_proj_dir = "/run/user/1000/gvfs/smb-share:server=shared.sydney.edu.au,share=research-data/PRJ-BowenLab/Experiments/2024/Other/2024_whole_brain_clearing_TS/KNX_Aggression_cohort_1_analysed_images"
    # in_fp_dir and batch_proj_dir cannot be the same
    assert in_fp_dir != batch_proj_dir

    # Get all experiments
    exp_ls = natsorted(os.listdir(batch_proj_dir))
    exp_ls = [i for i in exp_ls if os.path.isdir(os.path.join(batch_proj_dir, i))]

    for i in exp_ls:
        # Only given files
        if i not in [
            "P12_2.5x_1x_zoom_07082024",
            "P15_2.5x_1x_zoom_07082024",
        ]:
            continue
        # Checking if it is a directory
        if not os.path.isdir(os.path.join(in_fp_dir, i)):
            continue
        # Logging which file is being processed
        print(f"Running: {i}")
        logging.info(f"Running: {i}")
        try:
            # Filenames
            in_fp = os.path.join(in_fp_dir, i)
            proj_dir = os.path.join(batch_proj_dir, i)

            # Getting file paths
            pfm = get_proj_fp_model(proj_dir)
            # Making project folders
            make_proj_dirs(proj_dir)

            # # Making params json
            # init_configs(pfm)

            # if not os.path.exists(pfm.raw):
            #     print("Making zarr")
            #     # Making zarr from tiff file(s)
            #     tiff2zarr(in_fp, pfm.raw, chunks=PROC_CHUNKS)

            # if not os.path.exists(pfm.regresult):
            # Preparing reference images
            ref_prepare_pipeline(
                pfm=pfm,
                # ref_orient_ls=(-2, 3, 1),
                # ref_z_trim=(None, None, None),
                # ref_y_trim=(None, None, None),
                # ref_x_trim=(None, None, None),
            )
            # Preparing image itself
            # img_rough_pipeline(
            #     pfm,
            #     # z_rough=3,
            #     # y_rough=6,
            #     # x_rough=6,
            # )
            img_fine_pipeline(
                pfm,
                # z_fine=1,
                # y_fine=0.6,
                # x_fine=0.6,
            )
            img_trim_pipeline(
                pfm,
                # z_trim=(None, None, None),
                # y_trim=(None, None, None),
                # x_trim=(None, None, None),
            )
            # Running Elastix registration
            registration_pipeline(pfm)

            # if not os.path.exists(pfm.cells_raw_df):
            #     # Making overlapped chunks images for processing
            #     img_overlap_pipeline(
            #         pfm,
            #         chunksize=PROC_CHUNKS,
            #         depth=DEPTH,
            #     )
            #     # Cell counting
            #     img_proc_pipeline(
            #         pfm=pfm,
            #         depth=DEPTH,
            #         tophat_sigma=10,
            #         dog_sigma1=1,
            #         dog_sigma2=4,
            #         gauss_sigma=101,
            #         thresh_p=60,
            #         min_threshd=50,
            #         max_threshd=9000,
            #         maxima_sigma=10,
            #         min_wshed=1,
            #         max_wshed=700,
            #     )
            #     # Patch to fix extra smb column error
            #     cells_df_smb_field_patch(pfm.cells_raw_df)

            # if not os.path.exists(pfm.cells_trfm_df):
            # Converting maxima from raw space to refernce atlas space
            transform_coords_pipeline(pfm)
            # Getting ID mappings
            cell_mapping_pipeline(pfm)
            # Grouping cells
            group_cells_pipeline(pfm)
            # Saving cells to csv
            cells2csv_pipeline(pfm)
            print()
        except Exception as e:
            logging.info(f"Error in {i}: {e}")
            print(f"Error in {i}: {e}")
