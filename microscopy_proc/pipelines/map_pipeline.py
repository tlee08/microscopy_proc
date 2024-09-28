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
    df_map_ids,
    nested_tree_dict2df,
)
from microscopy_proc.utils.config_params_model import ConfigParamsModel
from microscopy_proc.utils.dask_utils import cluster_proc_contxt
from microscopy_proc.utils.io_utils import read_json
from microscopy_proc.utils.proj_org_utils import (
    get_proj_fp_dict,
    init_params,
    make_proj_dirs,
)


# @flow
def transform_coords(proj_fp_dict: dict):
    """
    `in_id` and `out_id` are either maxima or region
    """
    with cluster_proc_contxt(LocalCluster(n_workers=4, threads_per_worker=1)):
        # Getting registration parameters
        rp = ConfigParamsModel.model_validate(read_json(proj_fp_dict["config_params"]))
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
        # NOTE: Using pandas parquet. does not work with dask yet
        # cells_df = dd.from_pandas(cells_df, npartitions=1)
        # Fitting resampled space to atlas image with Transformix (from Elastix registration step)
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
        # Making unique incrementing index
        cells_df = cells_df.reset_index(drop=True)
        # Setting the transformed coords
        cells_df["z_trfm"] = coords_trfm["z"].values
        cells_df["y_trfm"] = coords_trfm["y"].values
        cells_df["x_trfm"] = coords_trfm["x"].values

        # Reading annotation image
        arr_annot = tifffile.imread(proj_fp_dict["annot"])
        # Getting the annotation ID for every cell (zyx coord)
        # Getting transformed coords (that are within tbe arr bounds, and their corresponding idx)
        s = arr_annot.shape
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
            arr_annot[*trfm_loc.values.T].astype(np.uint32),
            index=trfm_loc.index,
        ).fillna(-1)

        # Reading annotation mappings dataframe
        with open(proj_fp_dict["map"], "r") as f:
            annot_df = nested_tree_dict2df(json.load(f)["msg"][0])
        # Getting the annotation name for every cell (zyx coord)
        cells_df = df_map_ids(cells_df, annot_df)
        # Saving to disk
        # NOTE: Using pandas parquet. does not work with dask yet
        # cells_df = dd.from_pandas(cells_df)
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
        cells_grouped_df = (
            cells_df.groupby("id")
            .agg(
                {
                    "z": "count",
                    "size": "sum",
                    "sum_intensity": "sum",
                    # "max_intensity": "sum",
                }
            )
            .rename(columns=CELL_MEASURES)
            .compute()
        )
        # Reading annotation mappings dataframe
        # Making df of region names and their parent region names
        with open(proj_fp_dict["map"], "r") as f:
            annot_df = nested_tree_dict2df(json.load(f)["msg"][0])
        # Combining (summing) the cells_grouped_df values for parent regions using the annot_df
        cells_grouped_df = combine_nested_regions(cells_grouped_df, annot_df)
        # Calculating integrated average intensity (sum_intensity / size)
        cells_grouped_df["iov"] = cells_grouped_df["sum"] / cells_grouped_df["volume"]
        # Saving to disk
        # NOTE: Using pandas parquet. does not work with dask yet
        # cells_grouped = dd.from_pandas(cells_grouped)
        cells_grouped_df.to_parquet(proj_fp_dict["cells_agg_df"], overwrite=True)


def cells2csv(proj_fp_dict: dict):
    (
        dd.read_parquet(proj_fp_dict["cells_agg_df"])
        .compute()
        .to_csv(proj_fp_dict["cells_agg_csv"])
    )


if __name__ == "__main__":
    # Filenames
    # proj_dir = "/home/linux1/Desktop/A-1-1/large_cellcount"
    proj_dir = r"/run/user/1000/gvfs/smb-share:server=shared.sydney.edu.au,share=research-data/PRJ-BowenLab/Experiments/2024/Other/2024_whole_brain_clearing_TS/KNX_Aggression_cohort_1_analysed_images/R18_agg_2.5x_1xzoom_03072024"

    proj_fp_dict = get_proj_fp_dict(proj_dir)
    make_proj_dirs(proj_dir)

    # Making params json
    init_params(proj_fp_dict)

    # Converting maxima from raw space to refernce atlas space
    transform_coords(proj_fp_dict)

    # get_cell_mappings(proj_fp_dict)

    # grouping_cells(proj_fp_dict)

    # cells2csv(proj_fp_dict)
