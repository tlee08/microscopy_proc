import os

from microscopy_proc.pipelines.mask_pipeline import make_mask_for_ref
from microscopy_proc.utils.proj_org_utils import (
    get_proj_fp_dict,
    make_proj_dirs,
)

if __name__ == "__main__":
    # # Filenames
    # proj_dir = "/home/linux1/Desktop/A-1-1/large_cellcount"
    # # Getting file paths
    # proj_fp_dict = get_proj_fp_dict(proj_dir)
    # Making project folders
    # make_proj_dirs(proj_dir)
    # # Running mask pipeline
    # make_mask_for_ref(proj_fp_dict)

    # Filenames
    # atlas_rsc_dir = "/home/linux1/Desktop/iDISCO/resources/atlas_resources/"
    batch_proj_dir = "/run/user/1000/gvfs/smb-share:server=shared.sydney.edu.au,share=research-data/PRJ-BowenLab/Experiments/2024/Other/2024_whole_brain_clearing_TS/KNX_Aggression_cohort_1_analysed_images"
    # in_fp_dir and batch_proj_dir cannot be the same

    # for i in os.listdir(batch_fp_dir):
    proj_dir = os.path.join(batch_proj_dir, "B3_2.5x_1x_zoom_08082024")
    # Getting file paths
    proj_fp_dict = get_proj_fp_dict(proj_dir)

    # Checking if it is a directory
    # if not os.path.isdir(proj_dir):
    #     continue
    # Logging which file is being processed
    # logging.info(f"Running: {i}")
    # Making project folders
    make_proj_dirs(proj_dir)

    # Running mask pipeline
    make_mask_for_ref(proj_fp_dict)
    # except Exception as e:
    #     logging.error(f"Error: {e}")
    #     # continue
