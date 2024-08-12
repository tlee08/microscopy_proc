import json

import dask.dataframe as dd
import numpy as np
import pandas as pd
import tifffile
from dask.distributed import LocalCluster

# from prefect import flow
from microscopy_proc.constants import CELL_MEASURES
from microscopy_proc.funcs.elastix_funcs import transformation_coords
from microscopy_proc.funcs.map_funcs import (
    combine_nested_regions,
    nested_tree_dict_to_df,
)
from microscopy_proc.utils.dask_utils import cluster_proc_contxt
from microscopy_proc.utils.io_utils import read_json
from microscopy_proc.utils.proj_org_utils import get_proj_fp_dict, make_proj_dirs
from microscopy_proc.utils.reg_params_model import RegParamsModel


# @flow
def transform_coords(proj_fp_dict: dict):
    """
    `in_id` and `out_id` are either maxima or region
    """
    with cluster_proc_contxt(LocalCluster(n_workers=4, threads_per_worker=1)):
        # Getting registration parameters
        rp = RegParamsModel.model_validate(read_json(proj_fp_dict["reg_params"]))
        # Setting output key (in the form "<maxima/region>_trfm_df")
        # Getting cell coords
        cells_df = dd.read_parquet(proj_fp_dict["cells_raw_df"]).compute()
        cells_df = cells_df[["z", "y", "x"]]
        # Scaling to resampled rough space
        # NOTE: this downsampling uses slicing so must be computed differently
        cells_df = cells_df / np.array((rp.z_rough, rp.y_rough, rp.x_rough))
        # Scaling to resampled space
        cells_df = cells_df * np.array((rp.z_fine, rp.y_fine, rp.x_fine))
        # Trimming/offsetting to sliced space
        cells_df = cells_df - np.array(
            [s[0] if s[0] else 0 for s in (rp.z_trim, rp.y_trim, rp.x_trim)]
        )

        cells_df = transformation_coords(
            cells_df, proj_fp_dict["ref"], proj_fp_dict["regresult"]
        )
        cells_df = dd.from_pandas(cells_df, npartitions=1)
        # Fitting resampled space to atlas image with Transformix (from Elastix registration step)
        # NOTE: does not work with dask yet
        # cells_df = cells_df.repartition(
        #     npartitions=int(np.ceil(cells_df.shape[0].compute() / ROWSPPART))
        # )
        # cells_df = cells_df.map_partitions(
        #     transformation_coords, proj_fp_dict["ref"], proj_fp_dict["regresult"]
        # )
        cells_df.to_parquet(proj_fp_dict["cells_trfm_df"], overwrite=True)


# @flow
def get_cell_mappings(proj_fp_dict: dict):
    """
    Using the transformed cell coordinates, get the region ID and name for each cell
    corresponding to the reference atlas.
    """
    with cluster_proc_contxt(LocalCluster()):
        # Reading cells dataframe
        cells_df = dd.read_parquet(proj_fp_dict["cells_raw_df"]).compute()
        coords_trfm = dd.read_parquet(proj_fp_dict["cells_trfm_df"]).compute()
        # Making unique index
        cells_df = cells_df.reset_index(drop=True)
        # Setting the transformed coords
        cells_df["z_trfm"] = coords_trfm["z"].values
        cells_df["y_trfm"] = coords_trfm["y"].values
        cells_df["x_trfm"] = coords_trfm["x"].values

        # Reading annotation image
        annot_arr = tifffile.imread(proj_fp_dict["annot"])
        # Getting the annotation ID for every cell (zyx coord)
        # Getting transformed coords (that are within tbe arr bounds, and their corresponding idx)
        s = annot_arr.shape
        trfm_loc = (
            cells_df[["z_trfm", "y_trfm", "x_trfm"]]
            .round(0)
            .astype(np.int32)
            .query(
                f"z_trfm >= 0 & z_trfm < {s[0]} & y_trfm >= 0 & y_trfm < {s[1]} & x_trfm >= 0 & x_trfm < {s[2]}"
            )
        )
        # Getting the pixel values of each valid transformed coord (hence the specified index).
        # Invalids are set to -1
        cells_df["id"] = pd.Series(
            annot_arr[*trfm_loc.values.T].astype(np.uint32),
            index=trfm_loc.index,
        ).fillna(-1)

        # Reading annotation mappings dataframe
        with open(proj_fp_dict["map"], "r") as f:
            annot_df = nested_tree_dict_to_df(json.load(f)["msg"][0])
        # Getting the annotation name for every cell (zyx coord)
        # Left-joining the cells dataframe with the annotation mappings dataframe
        cells_df = pd.merge(
            left=cells_df,
            right=annot_df,
            how="left",
            on="id",
        )
        # Setting points with ID == -1 as "invalid" label
        cells_df.loc[cells_df["id"] == -1, "name"] = "invalid"
        # Setting points with ID == 0 as "universe" label
        cells_df.loc[cells_df["id"] == 0, "name"] = "universe"
        # Setting points with no region map name (but have a positive ID value) as "no label" label
        cells_df.loc[cells_df["name"].isna(), "name"] = "no label"
        # Saving to disk
        cells_df = dd.from_pandas(cells_df)
        cells_df.to_parquet(proj_fp_dict["cells_df"], overwrite=True)


def grouping_cells(proj_fp_dict: dict):
    """
    Grouping cells by region name and aggregating total cell volume
    and cell count for each region.
    """
    with cluster_proc_contxt(LocalCluster()):
        # Reading cells dataframe
        cells_df = dd.read_parquet(proj_fp_dict["cells_df"])
        # Grouping cells by region name
        cells_grouped = (
            cells_df.groupby("id")
            .agg(
                {
                    "z": "count",
                    "size": "sum",
                    "sum_itns": "sum",
                    # "max_itns": "sum",
                }
            )
            .rename(columns=CELL_MEASURES)
            .compute()
        )
        # Reading annotation mappings dataframe
        # Making df of region names and their parent region names
        with open(proj_fp_dict["map"], "r") as f:
            annot_df = nested_tree_dict_to_df(json.load(f)["msg"][0])
        cells_grouped = combine_nested_regions(cells_grouped, annot_df)
        # Saving to disk
        cells_grouped = dd.from_pandas(cells_grouped)
        cells_grouped.to_parquet(proj_fp_dict["cells_agg_df"], overwrite=True)


if __name__ == "__main__":
    # Filenames
    proj_dir = "/home/linux1/Desktop/A-1-1/large_cellcount"

    proj_fp_dict = get_proj_fp_dict(proj_dir)
    make_proj_dirs(proj_dir)

    # Converting maxima from raw space to refernce atlas space
    transform_coords(proj_fp_dict)

    get_cell_mappings(proj_fp_dict)

    grouping_cells(proj_fp_dict)
