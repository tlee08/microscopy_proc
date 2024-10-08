import logging
import os

import pandas as pd
from natsort import natsorted

from microscopy_proc.constants import ANNOT_COLUMNS_FINAL, CellColumns, MaskColumns
from microscopy_proc.funcs.map_funcs import annot_df_get_parents, annot_dict2df
from microscopy_proc.utils.io_utils import read_json, sanitise_smb_df
from microscopy_proc.utils.misc_utils import enum2list
from microscopy_proc.utils.proj_org_utils import get_proj_fp_model

# logging.basicConfig(level=logging.INFO)
logging.disable(logging.CRITICAL)


if __name__ == "__main__":
    # Filenames
    batch_proj_dir = "/run/user/1000/gvfs/smb-share:server=shared.sydney.edu.au,share=research-data/PRJ-BowenLab/Experiments/2024/Other/2024_whole_brain_clearing_TS/KNX_Aggression_cohort_1_analysed_images"

    out_fp = os.path.join(batch_proj_dir, "combined_agg_df.parquet")
    out_csv_fp = os.path.join(batch_proj_dir, "combined_agg_df.csv")

    # Get all experiments
    exp_ls = natsorted(os.listdir(batch_proj_dir))
    exp_ls = [i for i in exp_ls if os.path.isdir(os.path.join(batch_proj_dir, i))]

    # Check if all experiments have cells_agg_df and mask_df files
    for i in exp_ls:
        proj_dir = os.path.join(batch_proj_dir, i)
        pfm = get_proj_fp_model(proj_dir)
        assert os.path.exists(pfm.cells_agg_df), f"Missing cells_agg_df for {i}"
        assert os.path.exists(pfm.mask_df), f"Missing mask_df for {i}"

    # Making combined_agg_df with (annot_df)
    total_df = annot_dict2df(read_json(pfm.map))
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
        # try:
        # CELL_AGG_DF
        # Reading experiment's cells_agg dataframe
        cells_agg_df = pd.read_parquet(pfm.cells_agg_df)
        # Sanitising (removing smb columns)
        cells_agg_df = sanitise_smb_df(cells_agg_df)
        # Keeping only the required columns (not annot columns)
        cells_agg_df = cells_agg_df[enum2list(CellColumns)]
        # MASK_DF
        # Reading experiment's mask_counts dataframe
        mask_df = pd.read_parquet(pfm.mask_df)
        # Keeping only the required columns
        mask_df = mask_df[enum2list(MaskColumns)]
        # Merging cells_agg_df with mask_df to combine columns
        exp_df = pd.merge(
            left=cells_agg_df,
            right=mask_df,
            left_index=True,
            right_index=True,
            how="outer",
        )
        # Making columns a multindex with levels as
        # (specimen name, cell agg columns)
        exp_df = pd.concat(
            [exp_df],
            keys=[i],
            names=["specimen"],
            axis=1,
        )
        # Merging with comb_agg_df (ID is index for both dfs)
        total_df = pd.merge(
            left=total_df,
            right=exp_df,
            left_index=True,
            right_index=True,
            how="outer",
        )
        print()
        # except Exception as e:
        #     logging.info(f"Error in {i}: {e}")
        #     print(f"Error in {i}: {e}")
    # Setting column MultiIndex's level names
    total_df.columns = total_df.columns.set_names(["specimen", "measure"])
    # Saving to disk
    total_df.to_parquet(out_fp)
    # Also saving as csv to disk
    total_df.to_csv(out_csv_fp)
    # break
