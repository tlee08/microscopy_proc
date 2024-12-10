from microscopy_proc.pipeline_funcs.pipeline_funcs import PipelineFuncs
from microscopy_proc.utils.proj_org_utils import (
    get_proj_fp_model,
    update_configs,
)

# logging.basicConfig(level=logging.INFO)
# logging.disable(logging.CRITICAL)


if __name__ == "__main__":
    # Filenames
    in_fp = "/path/to/tiff_img_folder"
    proj_dir = "/path/to/analysis_output_folder"
    # Whether to overwrite existing files
    overwrite = True

    # in_fp_dir and batch_proj_dir cannot be the same
    assert in_fp != proj_dir

    pfm = get_proj_fp_model(proj_dir)
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
