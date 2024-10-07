import json
import logging
import os

import pandas as pd
from natsort import natsorted

from microscopy_proc.constants import ANNOT_COLUMNS_FINAL, CELL_AGG_MAPPINGS
from microscopy_proc.funcs.map_funcs import annot_df_get_parents, annot_dict2df
from microscopy_proc.utils.io_utils import sanitise_smb_df
from microscopy_proc.utils.proj_org_utils import get_proj_fp_model, get_ref_fp_model

# logging.basicConfig(level=logging.INFO)
logging.disable(logging.CRITICAL)


if __name__ == "__main__":
    # Filenames
    batch_proj_dir = "/run/user/1000/gvfs/smb-share:server=shared.sydney.edu.au,share=research-data/PRJ-BowenLab/Experiments/2024/Other/2024_whole_brain_clearing_TS/KNX_Aggression_cohort_1_analysed_images"

    out_fp = os.path.join(batch_proj_dir, "combined_agg_df.parquet")

    # Get all experiments
    exp_ls = natsorted(os.listdir(batch_proj_dir))
    exp_ls = [i for i in exp_ls if os.path.isdir(os.path.join(batch_proj_dir, i))]

    # Check if all experiments have cells_agg_df file
    for i in exp_ls:
        proj_dir = os.path.join(batch_proj_dir, i)
        pfm = get_proj_fp_model(proj_dir)
        assert os.path.exists(pfm.cells_agg_df), f"Missing cells_agg_df for {i}"

    # Making combined_agg_df with (annot_df)
    rfm = get_ref_fp_model()
    with open(pfm.map, "r") as f:
        total_df = annot_dict2df(json.load(f))
    total_df = annot_df_get_parents(total_df)
    # Keeping only the required columns
    total_df = total_df[ANNOT_COLUMNS_FINAL]
    # Making columns a multindex with levels as
    # ("annotations", annot columns)
    total_df = pd.concat(
        [total_df],
        keys=["annotations"],
        names=["specimen"],
        axis=1,
    )
    # Get all experiments
    for i in exp_ls:
        # Logging which file is being processed
        print(f"Running: {i}")
        logging.info(f"Running: {i}")
        # Filenames
        proj_dir = os.path.join(batch_proj_dir, i)
        pfm = get_proj_fp_model(proj_dir)
        try:
            # Reading experiment's cells_agg dataframe
            cells_agg_df = pd.read_parquet(pfm.cells_agg_df)
            # Sanitising (removing smb columns)
            cells_agg_df = sanitise_smb_df(cells_agg_df)
            # Keeping only the required columns (not annot columns)
            cells_agg_df = cells_agg_df[list(CELL_AGG_MAPPINGS.keys())]
            # Making columns a multindex with levels as
            # (specimen name, cell agg columns)
            cells_agg_df = pd.concat(
                [cells_agg_df],
                keys=[i],
                names=["specimen"],
                axis=1,
            )
            # Merging with comb_agg_df (ID is index for both dfs)
            total_df = pd.merge(
                left=total_df,
                right=cells_agg_df,
                left_index=True,
                right_index=True,
                how="outer",
            )
            print()
        except Exception as e:
            logging.info(f"Error in {i}: {e}")
            print(f"Error in {i}: {e}")
    # Saving to disk
    total_df.to_parquet(out_fp)
    # break
