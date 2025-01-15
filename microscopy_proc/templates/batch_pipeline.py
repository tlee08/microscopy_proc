import os

from natsort import natsorted

from microscopy_proc.funcs.batch_combine_funcs import BatchCombineFuncs
from microscopy_proc.pipeline.pipeline import Pipeline
from microscopy_proc.utils.logging_utils import init_logger

if __name__ == "__main__":
    # Filenames
    in_root_dir = "/path/to/tiff_imgs_folder"
    root_dir = "/path/to/analysis_outputs_folder"
    # Whether to overwrite existing files
    overwrite = True

    # in_fp_dir and batch_proj_dir cannot be the same
    assert in_root_dir != root_dir

    logger = init_logger(__name__)

    # Get all experiments
    exp_ls = [fp for fp in natsorted(os.listdir(in_root_dir)) if os.path.isdir(os.path.join(in_root_dir, fp))]
    # exp_ls = ["example_img"]
    for exp in exp_ls:
        logger.info(f"Running: {exp}")
        try:
            in_fp = os.path.join(in_root_dir, exp)
            proj_dir = os.path.join(root_dir, exp)
            pfm = Pipeline.get_pfm(proj_dir)
            # Can change cell counting to tuning mode here
            pfm_tuning = Pipeline.get_pfm_tuning(proj_dir)
            # Updating project pipeline configs
            Pipeline.update_configs(
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
                # lower_bound=(500, 0),
                # upper_bound=(5000, 5000),
                # # MASK
                # # CELL COUNT TUNING CROP
                tuning_z_trim=(700, 800, None),
                tuning_y_trim=(1000, 3000, None),
                tuning_x_trim=(1000, 3000, None),
                # # OVERLAP
                # # CELL COUNTING
                # tophat_sigma=10,
                # dog_sigma1=1,
                # dog_sigma2=4,
                # large_gauss_sigma=101,
                # threshd_value=60,
                # min_threshd_size=100,
                # max_threshd_size=9000,
                # maxima_sigma=10,
                # min_wshed_size=1,
                # max_wshed_size=700,
                # # VISUAL CHECK
                # heatmap_raw_radius=5,
                # heatmap_trfm_radius=3,
                # # COMBINE ARRAYS
                combine_cellc_z_trim=(750, 760, None),
                # combine_cellc_y_trim=(None, None, None),
                # combine_cellc_x_trim=(None, None, None),
            )

            # Making zarr from tiff file(s)
            Pipeline.tiff2zarr(pfm, in_fp, overwrite=overwrite)
            # Preparing reference images
            Pipeline.reg_ref_prepare(pfm, overwrite=overwrite)
            # Preparing image itself
            Pipeline.reg_img_rough(pfm, overwrite=overwrite)
            Pipeline.reg_img_fine(pfm, overwrite=overwrite)
            Pipeline.reg_img_trim(pfm, overwrite=overwrite)
            Pipeline.reg_img_bound(pfm, overwrite=overwrite)
            # Running Elastix registration
            Pipeline.reg_elastix(pfm, overwrite=overwrite)
            # Running mask pipeline
            Pipeline.make_mask(pfm, overwrite=overwrite)
            # Making trimmed image for cell count tuning
            Pipeline.make_tuning_arr(pfm, overwrite=overwrite)

            # Cell Counting in both tuning and production mode
            for pfm_i in [
                pfm_tuning,
                pfm,
            ]:
                # Making overlap chunks in preparation for cell counting
                Pipeline.img_overlap(pfm_i, overwrite=overwrite)
                # Counting cells
                Pipeline.cellc1(pfm_i, overwrite=overwrite)
                Pipeline.cellc2(pfm_i, overwrite=overwrite)
                Pipeline.cellc3(pfm_i, overwrite=overwrite)
                Pipeline.cellc4(pfm_i, overwrite=overwrite)
                Pipeline.cellc5(pfm_i, overwrite=overwrite)
                Pipeline.cellc6(pfm_i, overwrite=overwrite)
                Pipeline.cellc7(pfm_i, overwrite=overwrite)
                Pipeline.cellc8(pfm_i, overwrite=overwrite)
                Pipeline.cellc9(pfm_i, overwrite=overwrite)
                Pipeline.cellc10(pfm_i, overwrite=overwrite)
                Pipeline.cellc11(pfm_i, overwrite=overwrite)

            # Converting maxima from raw space to refernce atlas space
            Pipeline.transform_coords(pfm, overwrite=overwrite)
            # Getting Region ID mappings for each cell
            Pipeline.cell_mapping(pfm, overwrite=overwrite)
            # Grouping cells
            Pipeline.group_cells(pfm, overwrite=overwrite)
            # Exporting cells_agg parquet as csv
            Pipeline.cells2csv(pfm, overwrite=overwrite)
            # Making points and heatmap images
            Pipeline.coords2points_raw(pfm, overwrite=overwrite)
            Pipeline.coords2points_trfm(pfm, overwrite=overwrite)
            Pipeline.coords2heatmap_trfm(pfm, overwrite=overwrite)
            # Combining arrays
            Pipeline.combine_reg(pfm, overwrite=overwrite)
            Pipeline.combine_cellc(pfm, overwrite=overwrite)
            Pipeline.combine_points(pfm, overwrite=overwrite)
        except Exception as e:
            logger.info(f"Error in {exp}: {e}")
    # Combining all experiment dataframes
    BatchCombineFuncs.combine_root_pipeline(root_dir, os.path.dirname(root_dir), overwrite=True)
