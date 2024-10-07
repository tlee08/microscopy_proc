import json

import dask.dataframe as dd
import numpy as np
import pandas as pd
import tifffile
from dask.distributed import LocalCluster

# from prefect import flow
from microscopy_proc.constants import (
    ANNOT_COLUMNS_FINAL,
    CELL_AGG_MAPPINGS,
    TRFM,
    AnnotColumns,
    CellColumns,
    Coords,
)
from microscopy_proc.funcs.elastix_funcs import transformation_coords
from microscopy_proc.funcs.map_funcs import (
    annot_dict2df,
    combine_nested_regions,
    df_map_ids,
)
from microscopy_proc.utils.config_params_model import ConfigParamsModel
from microscopy_proc.utils.dask_utils import cluster_proc_contxt
from microscopy_proc.utils.io_utils import read_json, sanitise_smb_df
from microscopy_proc.utils.misc_utils import enum2list
from microscopy_proc.utils.proj_org_utils import (
    ProjFpModel,
    get_proj_fp_model,
    init_configs,
    make_proj_dirs,
)


# @flow
def transform_coords(pfm: ProjFpModel):
    """
    `in_id` and `out_id` are either maxima or region

    NOTE: saves the cells_trfm dataframe as pandas parquet.
    """
    with cluster_proc_contxt(LocalCluster(n_workers=4, threads_per_worker=1)):
        # Getting registration parameters
        rp = ConfigParamsModel.model_validate(read_json(pfm.config_params))
        # Setting output key (in the form "<maxima/region>_trfm_df")
        # Getting cell coords
        cells_df = dd.read_parquet(pfm.cells_raw_df).compute()
        # Sanitising (removing smb columns)
        cells_df = sanitise_smb_df(cells_df)
        # Taking only Coords.Z.value, Coords.Y.value, Coords.X.value coord columns
        cells_df = cells_df[[Coords.Z.value, Coords.Y.value, Coords.X.value]]
        # Scaling to resampled rough space
        # NOTE: this downsampling uses slicing so must be computed differently
        cells_df = cells_df / np.array((rp.z_rough, rp.y_rough, rp.x_rough))
        # Scaling to resampled space
        cells_df = cells_df * np.array((rp.z_fine, rp.y_fine, rp.x_fine))
        # Trimming/offsetting to sliced space
        cells_df = cells_df - np.array(
            [s[0] if s[0] else 0 for s in (rp.z_trim, rp.y_trim, rp.x_trim)]
        )

        cells_df = transformation_coords(cells_df, pfm.ref, pfm.regresult)
        # NOTE: Using pandas parquet. does not work with dask yet
        # cells_df = dd.from_pandas(cells_df, npartitions=1)
        # Fitting resampled space to atlas image with Transformix (from Elastix registration step)
        # cells_df = cells_df.repartition(
        #     npartitions=int(np.ceil(cells_df.shape[0].compute() / ROWSPPART))
        # )
        # cells_df = cells_df.map_partitions(
        #     transformation_coords, pfm.ref, pfm.regresult
        # )
        cells_df.to_parquet(pfm.cells_trfm_df)


# @flow
def get_cell_mappings(pfm: ProjFpModel):
    """
    Using the transformed cell coordinates, get the region ID and name for each cell
    corresponding to the reference atlas.

    NOTE: saves the cells dataframe as pandas parquet.
    """
    with cluster_proc_contxt(LocalCluster()):
        # Reading cells_raw and cells_trfm dataframes
        cells_df = dd.read_parquet(pfm.cells_raw_df).compute()
        coords_trfm = pd.read_parquet(pfm.cells_trfm_df)
        # Sanitising (removing smb columns)
        cells_df = sanitise_smb_df(cells_df)
        coords_trfm = sanitise_smb_df(coords_trfm)
        # Making unique incrementing index
        cells_df = cells_df.reset_index(drop=True)
        # Setting the transformed coords
        cells_df[f"{Coords.Z.value}_{TRFM}"] = coords_trfm[Coords.Z.value].values
        cells_df[f"{Coords.Y.value}_{TRFM}"] = coords_trfm[Coords.Y.value].values
        cells_df[f"{Coords.X.value}_{TRFM}"] = coords_trfm[Coords.X.value].values

        # Reading annotation image
        arr_annot = tifffile.imread(pfm.annot)
        # Getting the annotation ID for every cell (zyx coord)
        # Getting transformed coords (that are within tbe arr bounds, and their corresponding idx)
        s = arr_annot.shape
        trfm_loc = (
            cells_df[
                [
                    f"{Coords.Z.value}_{TRFM}",
                    f"{Coords.Y.value}_{TRFM}",
                    f"{Coords.X.value}_{TRFM}",
                ]
            ]
            .round(0)
            .astype(np.int32)
            .query(
                f"{Coords.Z.value}_{TRFM} >= 0 & {Coords.Z.value}_{TRFM} < {s[0]} & "
                + f"{Coords.Y.value}_{TRFM} >= 0 & {Coords.Y.value}_{TRFM} < {s[1]} & "
                + f"{Coords.X.value}_{TRFM} >= 0 & {Coords.X.value}_{TRFM} < {s[2]}"
            )
        )
        # Getting the pixel values of each valid transformed coord (hence the specified index)
        # By complex array indexing on arr_annot's (z, y, x) dimensions.
        # nulls are imputed with -1
        cells_df[AnnotColumns.ID.value] = pd.Series(
            arr_annot[*trfm_loc.values.T].astype(np.uint32),
            index=trfm_loc.index,
        ).fillna(-1)

        # Reading annotation mappings dataframe
        with open(pfm.map, "r") as f:
            annot_df = annot_dict2df(json.load(f))
        # Getting the annotation name for every cell (zyx coord)
        cells_df = df_map_ids(cells_df, annot_df)
        # Saving to disk
        # NOTE: Using pandas parquet. does not work with dask yet
        # cells_df = dd.from_pandas(cells_df)
        cells_df.to_parquet(pfm.cells_df)


def grouping_cells(pfm: ProjFpModel):
    """
    Grouping cells by region name and aggregating total cell volume
    and cell count for each region.

    NOTE: saves the cells_agg dataframe as pandas parquet.
    """
    with cluster_proc_contxt(LocalCluster()):
        # Reading cells dataframe
        cells_df = pd.read_parquet(pfm.cells_df)
        # Sanitising (removing smb columns)
        cells_df = sanitise_smb_df(cells_df)
        # Grouping cells by region name
        cells_agg_df = cells_df.groupby(AnnotColumns.ID.value).agg(CELL_AGG_MAPPINGS)
        cells_agg_df.columns = list(CELL_AGG_MAPPINGS.keys())
        # Reading annotation mappings dataframe
        # Making df of region names and their parent region names
        with open(pfm.map, "r") as f:
            annot_df = annot_dict2df(json.load(f))
        # Combining (summing) the cells_groagg values for parent regions using the annot_df
        cells_agg_df = combine_nested_regions(cells_agg_df, annot_df)
        # Calculating integrated average intensity (sum_intensity / size)
        cells_agg_df[CellColumns.IOV.value] = (
            cells_agg_df[CellColumns.SUM_INTENSITY.value]
            / cells_agg_df[CellColumns.VOLUME.value]
        )
        # Selecting and ordering relevant columns
        cells_agg_df = cells_agg_df[[*ANNOT_COLUMNS_FINAL, *enum2list(CellColumns)]]
        # Saving to disk
        # NOTE: Using pandas parquet. does not work with dask yet
        # cells_agg = dd.from_pandas(cells_agg)
        cells_agg_df.to_parquet(pfm.cells_agg_df)


def cells2csv(pfm: ProjFpModel):
    # Reading cells dataframe
    cells_agg_df = pd.read_parquet(pfm.cells_agg_df)
    # Sanitising (removing smb columns)
    cells_agg_df = sanitise_smb_df(cells_agg_df)
    # Saving to csv
    cells_agg_df.to_csv(pfm.cells_agg_csv)


if __name__ == "__main__":
    # Filenames
    # proj_dir = "/home/linux1/Desktop/A-1-1/large_cellcount"
    proj_dir = r"/run/user/1000/gvfs/smb-share:server=shared.sydney.edu.au,share=research-data/PRJ-BowenLab/Experiments/2024/Other/2024_whole_brain_clearing_TS/KNX_Aggression_cohort_1_analysed_images/R18_agg_2.5x_1xzoom_03072024"

    pfm = get_proj_fp_model(proj_dir)
    make_proj_dirs(proj_dir)

    # Making params json
    init_configs(pfm)

    # Converting maxima from raw space to refernce atlas space
    transform_coords(pfm)

    # get_cell_mappings(pfm)

    # grouping_cells(pfm)

    # cells2csv(pfm)
