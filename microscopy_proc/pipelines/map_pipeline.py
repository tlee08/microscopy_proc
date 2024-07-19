import json

import dask.dataframe as dd
import numpy as np
import pandas as pd
import tifffile
from dask.distributed import LocalCluster

from microscopy_proc.utils.dask_utils import (
    cluster_proc_dec,
)
from microscopy_proc.utils.elastix_utils import transformation_coords
from microscopy_proc.utils.map_utils import nested_tree_dict_to_df
from microscopy_proc.utils.proj_org_utils import get_proj_fp_dict, make_proj_dirs


@cluster_proc_dec(lambda: LocalCluster())
def transform_coords(
    proj_fp_dict: dict,
    z_rough: int,
    y_rough: int,
    x_rough: int,
    z_fine: float,
    y_fine: float,
    x_fine: float,
    z_trim: slice,
    y_trim: slice,
    x_trim: slice,
):
    # Getting cell coords
    coords = dd.read_parquet(proj_fp_dict["maxima_df"])
    coords = coords[["z", "y", "x"]]
    # Scaling to resampled rough space
    # NOTE: this downsampling uses slicing so must be computed differently
    coords = coords / np.array((z_rough, y_rough, x_rough))
    # Scaling to resampled space
    coords = coords * np.array((z_fine, y_fine, x_fine))
    # Trimming/offsetting to sliced space
    coords = coords - np.array(
        [s.start if s.start else 0 for s in (z_trim, y_trim, x_trim)]
    )
    # Fitting resampled space to atlas image with Transformix (from Elastix registration step)
    # NOTE: does not work with dask yet
    coords = coords.compute()
    coords = transformation_coords(
        coords, proj_fp_dict["ref"], proj_fp_dict["regresult"]
    )
    # Saving to disk
    dd.from_pandas(coords).to_parquet(proj_fp_dict["maxima_trfm_df"])


def get_cell_mappings(proj_fp_dict: dict):
    # Reading cells dataframe
    cells_df = dd.read_parquet(proj_fp_dict["maxima_df"]).compute()
    coords_trfm = dd.read_parquet(proj_fp_dict["maxima_trfm_df"]).compute()
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
    )
    cells_df["id"] = cells_df["id"].fillna(-1)

    # Reading annotation mappings dataframe
    with open(proj_fp_dict["map"], "r") as f:
        annot_df = nested_tree_dict_to_df(json.load(f)["msg"][0])
    # Getting the annotation name for every cell (zyx coord)
    cells_df = get_cell_mappings_names(cells_df, annot_df)
    # Saving to disk
    dd.from_pandas(cells_df).to_parquet(proj_fp_dict["cells_df"])


def get_cell_mappings_names(cells_df, annot_df):
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
    return cells_df


if __name__ == "__main__":
    # Filenames
    proj_dir = "/home/linux1/Desktop/A-1-1/large_cellcount"

    proj_fp_dict = get_proj_fp_dict(proj_dir)
    make_proj_dirs(proj_dir)

    # Converting from raw space to refernce atlas space
    transform_coords(
        proj_fp_dict=proj_fp_dict,
        z_rough=8,
        y_rough=10,
        x_rough=10,
        z_fine=0.8,
        y_fine=0.8,
        x_fine=0.8,
        z_trim=slice(None, -5),
        y_trim=slice(60, -50),
        x_trim=slice(None, None),
    )

    get_cell_mappings(proj_fp_dict)
