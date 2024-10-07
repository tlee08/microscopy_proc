import logging
import os

from natsort import natsorted

from microscopy_proc.pipelines.map_pipeline import (
    cells2csv,
    grouping_cells,
)
from microscopy_proc.utils.proj_org_utils import (
    get_proj_fp_model,
    get_ref_fp_model,
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

    for i in natsorted(os.listdir(in_fp_dir)):
        # # Only given files
        # if i not in [
        #     "B3_2.5x_1x_zoom_08082024",
        #     #     "B9_2.5x_1x_zoom_06082024",
        #     #     "G5_agg_2.5x_1xzoom_05072024",
        #     #     "G8_2.5x_1x_zoom_08082024",
        #     #     "G13_2.5x_1x_zoom_07082024",
        # ]:
        #     continue
        # Checking if it is a directory
        if not os.path.isdir(os.path.join(in_fp_dir, i)):
            continue
        # Logging which file is being processed
        print(f"Running: {i}")
        logging.info(f"Running: {i}")
        # try:
        # Filenames
        in_fp = os.path.join(in_fp_dir, i)
        proj_dir = os.path.join(batch_proj_dir, i)

        # Getting file paths
        rfm = get_ref_fp_model()
        pfm = get_proj_fp_model(proj_dir)
        # Making project folders
        make_proj_dirs(proj_dir)
        # print(dd.read_parquet(pfm.cells_raw_df))
        # Converting maxima from raw space to refernce atlas space
        # transform_coords(pfm)
        # Getting ID mappings
        # get_cell_mappings(pfm)
        # Grouping cells
        grouping_cells(pfm)
        # Saving cells to csv
        cells2csv(pfm)
        print()
        # except Exception as e:
        #     logging.info(f"Error in {i}: {e}")
        #     print(f"Error in {i}: {e}")
        #     # continue
        break
