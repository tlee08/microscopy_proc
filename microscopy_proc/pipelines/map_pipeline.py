import json

import dask.dataframe as dd
import numpy as np
import pandas as pd
import tifffile
from dask.distributed import LocalCluster

from microscopy_proc.funcs.elastix_funcs import transformation_coords

# from prefect import flow
from microscopy_proc.utils.dask_utils import (
    cluster_proc_dec,
)
from microscopy_proc.utils.map_utils import nested_tree_dict_to_df
from microscopy_proc.utils.proj_org_utils import get_proj_fp_dict, make_proj_dirs


@cluster_proc_dec(lambda: LocalCluster(n_workers=4, threads_per_worker=1))
# @flow
def transform_coords(
    proj_fp_dict: dict,
    in_id: str,
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
    """
    `in_id` and `out_id` are either maxima or region
    """
    # Setting output key (in the form "<maxima/region>_trfm_df")
    out_id = f"{in_id.split('_')[0]}_trfm_df"
    # Getting cell coords
    coords = dd.read_parquet(proj_fp_dict[in_id])
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
    # coords = coords.compute()
    # coords_d_ls = coords.to_delayed()
    # coords_d_ls = [
    #     transformation_coords(i, proj_fp_dict["ref"], proj_fp_dict["regresult"])
    #     for i in coords_d_ls
    # ]
    # coords_c_ls = dask.compute(i for i in coords_d_ls)
    # coords = dd.from_delayed(coords_c_ls)
    # coords = coords.repartition(
    #     npartitions=int(np.ceil(coords.shape[0].compute() / ROWSPERPART))
    # )
    coords = coords.map_partitions(
        transformation_coords, proj_fp_dict["ref"], proj_fp_dict["regresult"]
    )
    coords.to_parquet(proj_fp_dict[out_id], overwrite=True)
    # coords = transformation_coords(
    #     coords, proj_fp_dict["ref"], proj_fp_dict["regresult"]
    # )
    # # Saving to disk
    # dd.from_pandas(coords).to_parquet(proj_fp_dict[out_id], overwrite=True)


# @flow
def get_cell_mappings(proj_fp_dict: dict):
    # Reading cells dataframe
    cells_df = dd.read_parquet(proj_fp_dict["cells_raw_df"])
    coords_trfm = dd.read_parquet(proj_fp_dict["cells_trfm_df"])
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
    # Left-joining the cells dataframe with the annotation mappings dataframe
    cells_df = dd.merge(
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
    dd.from_pandas(cells_df).to_parquet(proj_fp_dict["cells_df"], overwrite=True)


if __name__ == "__main__":
    # Filenames
    proj_dir = "/home/linux1/Desktop/A-1-1/large_cellcount"

    proj_fp_dict = get_proj_fp_dict(proj_dir)
    make_proj_dirs(proj_dir)

    # Converting maxima from raw space to refernce atlas space
    transform_coords(
        proj_fp_dict=proj_fp_dict,
        in_id="cells_raw_df",
        z_rough=3,
        y_rough=6,
        x_rough=6,
        z_fine=1,
        y_fine=0.6,
        x_fine=0.6,
        z_trim=slice(None, -5),
        y_trim=slice(80, -75),
        x_trim=slice(None, None),
    )

    # get_cell_mappings(proj_fp_dict)
    # get_cell_mappings(proj_fp_dict)
