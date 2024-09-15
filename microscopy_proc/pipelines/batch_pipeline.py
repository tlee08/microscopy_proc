import logging
import os

from microscopy_proc.constants import DEPTH, PROC_CHUNKS
from microscopy_proc.funcs.elastix_funcs import registration
from microscopy_proc.pipelines.cellc_pipeline import (
    cells_df_smb_field_patch,
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

# logging.basicConfig(level=logging.INFO)
logging.disable(logging.CRITICAL)

if __name__ == "__main__":
    # Filenames
    # atlas_rsc_dir = "/home/linux1/Desktop/iDISCO/resources/atlas_resources/"
    in_fp_dir = "/run/user/1000/gvfs/smb-share:server=shared.sydney.edu.au,share=research-data/PRJ-BowenLab/Experiments/2024/Other/2024_whole_brain_clearing_TS/KNX Aggression cohort 1 stitched TIF images for analysis"
    batch_proj_dir = "/run/user/1000/gvfs/smb-share:server=shared.sydney.edu.au,share=research-data/PRJ-BowenLab/Experiments/2024/Other/2024_whole_brain_clearing_TS/KNX_Aggression_cohort_1_analysed_images"
    # in_fp_dir and batch_proj_dir cannot be the same
    assert in_fp_dir != batch_proj_dir

    for i in os.listdir(in_fp_dir):
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
            ref_fp_dict = get_ref_fp_dict()
            proj_fp_dict = get_proj_fp_dict(proj_dir)
            # Making project folders
            make_proj_dirs(proj_dir)

            # Making params json
            init_params(proj_fp_dict)

            if not os.path.exists(proj_fp_dict["raw"]):
                print("Making zarr")
                # Making zarr from tiff file(s)
                tiff_to_zarr(in_fp, proj_fp_dict["raw"], chunks=PROC_CHUNKS)

            if not os.path.exists(proj_fp_dict["regresult"]):
                # Preparing reference images
                prepare_ref(
                    ref_fp_dict=ref_fp_dict,
                    proj_fp_dict=proj_fp_dict,
                    ref_orient_ls=(-2, 3, 1),
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
                    z_trim=(None, None, None),
                    y_trim=(None, None, None),
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

            if not os.path.exists(proj_fp_dict["cells_raw_df"]):
                # Making overlapped chunks images for processing
                img_overlap_pipeline(proj_fp_dict, chunks=PROC_CHUNKS, d=DEPTH)
                # Cell counting
                img_proc_pipeline(
                    proj_fp_dict=proj_fp_dict,
                    d=DEPTH,
                    tophat_sigma=10,
                    dog_sigma1=1,
                    dog_sigma2=4,
                    gauss_sigma=101,
                    thresh_p=60,
                    min_threshd=50,
                    max_threshd=9000,
                    maxima_sigma=10,
                    min_wshed=1,
                    max_wshed=700,
                )
                # Patch to fix extra smb column error
                cells_df_smb_field_patch(proj_fp_dict["cells_raw_df"])

            if not os.path.exists(proj_fp_dict["cells_trfm_df"]):
                # Converting maxima from raw space to refernce atlas space
                transform_coords(proj_fp_dict)
                # Getting ID mappings
                get_cell_mappings(proj_fp_dict)
                # Grouping cells
                grouping_cells(proj_fp_dict)
                # Saving cells to csv
                cells2csv(proj_fp_dict)
            print()
        except Exception as e:
            logging.info(f"Error in {i}: {e}")
            continue
