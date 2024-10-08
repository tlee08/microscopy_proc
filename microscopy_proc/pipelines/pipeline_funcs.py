import json
import os
import re
import shutil

import dask.array as da
import dask.dataframe as dd
import numpy as np
import pandas as pd
import tifffile
from dask.distributed import LocalCluster
from dask_cuda import LocalCUDACluster
from natsort import natsorted
from scipy import ndimage

# from prefect import flow
from microscopy_proc.constants import (
    ANNOT_COLUMNS_FINAL,
    CELL_AGG_MAPPINGS,
    TRFM,
    AnnotColumns,
    CellColumns,
    Coords,
    MaskColumns,
)

# from prefect import flow
from microscopy_proc.funcs.cpu_arr_funcs import CpuArrFuncs as Cf

# from prefect import flow
from microscopy_proc.funcs.elastix_funcs import registration, transformation_coords
from microscopy_proc.funcs.gpu_arr_funcs import GpuArrFuncs as Gf

# from prefect import flow
from microscopy_proc.funcs.io_funcs import btiff2zarr, tiffs2zarr
from microscopy_proc.funcs.map_funcs import (
    annot_dict2df,
    combine_nested_regions,
    df_map_ids,
)
from microscopy_proc.funcs.mask_funcs import (
    fill_outline,
    make_outline,
    mask2region_counts,
)

# from prefect import flow
from microscopy_proc.funcs.reg_funcs import (
    downsmpl_fine_arr,
    downsmpl_rough_arr,
    reorient_arr,
)
from microscopy_proc.funcs.visual_check_funcs import coords2points
from microscopy_proc.utils.config_params_model import ConfigParamsModel
from microscopy_proc.utils.dask_utils import (
    block2coords,
    cluster_proc_contxt,
    da_overlap,
    da_trim,
    disk_cache,
)
from microscopy_proc.utils.io_utils import read_json, sanitise_smb_df
from microscopy_proc.utils.misc_utils import enum2list
from microscopy_proc.utils.proj_org_utils import (
    ProjFpModel,
    RefFpModel,
)


# @flow
def tiff2zarr_pipeline(
    in_fp: str,
    pfm: ProjFpModel,
    **kwargs,
):
    # Update registration params json
    configs = ConfigParamsModel.update_params_file(pfm.config_params, **kwargs)
    # Making zarr from tiff file(s)
    with cluster_proc_contxt(LocalCluster(n_workers=1, threads_per_worker=6)):
        if os.path.isdir(in_fp):
            tiffs2zarr(
                natsorted(
                    [
                        os.path.join(in_fp, f)
                        for f in os.listdir(in_fp)
                        if re.search(r".tif$", f)
                    ]
                ),
                pfm.raw,
                chunks=configs.chunksize,
            )
        elif os.path.isfile(in_fp):
            btiff2zarr(
                in_fp,
                pfm.raw,
                chunks=configs.chunksize,
            )
        else:
            raise ValueError("Input file path does not exist.")


# @flow
def ref_prepare_pipeline(
    pfm: ProjFpModel,
    **kwargs,
):
    # Update registration params json
    configs = ConfigParamsModel.update_params_file(pfm.config_params, **kwargs)
    # Making ref_fp_model of original atlas images filepaths
    rfm = RefFpModel.get_ref_fp_model(
        configs.atlas_dir,
        configs.ref_v,
        configs.annot_v,
        configs.map_v,
    )
    # Making atlas images
    for fp_i, fp_o in [
        (rfm.ref, pfm.ref),
        (rfm.annot, pfm.annot),
    ]:
        # Reading
        arr = tifffile.imread(fp_i)
        # Reorienting
        arr = reorient_arr(arr, configs.ref_orient_ls)
        # Slicing
        arr = arr[
            slice(*configs.ref_z_trim),
            slice(*configs.ref_y_trim),
            slice(*configs.ref_x_trim),
        ]
        # Saving
        tifffile.imwrite(fp_o, arr)
    # Copying region mapping json to project folder
    shutil.copyfile(rfm.map, pfm.map)
    # Copying transformation files
    shutil.copyfile(rfm.affine, pfm.affine)
    shutil.copyfile(rfm.bspline, pfm.bspline)


# @flow
def img_rough_pipeline(
    pfm: ProjFpModel,
    **kwargs,
):
    # Update registration params json
    configs = ConfigParamsModel.update_params_file(pfm.config_params, **kwargs)
    with cluster_proc_contxt(LocalCluster()):
        # Reading
        arr_raw = da.from_zarr(pfm.raw)
        # Rough downsample
        arr_downsmpl1 = downsmpl_rough_arr(
            arr_raw, configs.z_rough, configs.y_rough, configs.x_rough
        )
        arr_downsmpl1 = arr_downsmpl1.compute()
        # Saving
        tifffile.imwrite(pfm.downsmpl1, arr_downsmpl1)


# @flow
def img_fine_pipeline(
    pfm: ProjFpModel,
    **kwargs,
):
    # Update registration params json
    configs = ConfigParamsModel.update_params_file(pfm.config_params, **kwargs)
    # Reading
    arr_downsmpl1 = tifffile.imread(pfm.downsmpl1)
    # Fine downsample
    arr_downsmpl2 = downsmpl_fine_arr(
        arr_downsmpl1, configs.z_fine, configs.y_fine, configs.x_fine
    )
    # Saving
    tifffile.imwrite(pfm.downsmpl2, arr_downsmpl2)


# @flow
def img_trim_pipeline(
    pfm: ProjFpModel,
    **kwargs,
):
    # Update registration params json
    configs = ConfigParamsModel.update_params_file(pfm.config_params, **kwargs)
    # Reading
    arr_downsmpl2 = tifffile.imread(pfm.downsmpl2)
    # Trim
    arr_trimmed = arr_downsmpl2[
        slice(*configs.z_trim), slice(*configs.y_trim), slice(*configs.x_trim)
    ]
    # Saving
    tifffile.imwrite(pfm.trimmed, arr_trimmed)


# @flow
def registration_pipeline(
    pfm: ProjFpModel,
    **kwargs,
):
    # Update registration params json
    configs = ConfigParamsModel.update_params_file(pfm.config_params, **kwargs)
    # Running Elastix registration
    registration(
        fixed_img_fp=pfm.trimmed,
        moving_img_fp=pfm.ref,
        output_img_fp=pfm.regresult,
        affine_fp=pfm.affine,
        bspline_fp=pfm.bspline,
    )


# @flow
def make_mask_pipeline(
    pfm: ProjFpModel,
    **kwargs,
):
    """
    Makes mask of actual image in reference space.
    Also stores # and proportion of existent voxels
    for each region.
    """
    # Update registration params json
    configs = ConfigParamsModel.update_params_file(pfm.config_params, **kwargs)
    # Reading ref and trimmed imgs
    arr_ref = tifffile.imread(pfm.ref)
    arr_trimmed = tifffile.imread(pfm.trimmed)
    # Making mask
    arr_blur = Gf.gauss_blur_filt(arr_trimmed, configs.mask_gaus_blur)
    tifffile.imwrite(pfm.premask_blur, arr_blur)
    arr_mask = Gf.manual_thresh(arr_blur, configs.mask_thresh)
    tifffile.imwrite(pfm.mask, arr_mask)

    # Make outline
    outline_df = make_outline(arr_mask)
    # Transformix on coords
    outline_df[[Coords.Z.value, Coords.Y.value, Coords.X.value]] = (
        transformation_coords(
            outline_df,
            pfm.ref,
            pfm.regresult,
        )[[Coords.Z.value, Coords.Y.value, Coords.X.value]]
        .round(0)
        .astype(np.int32)
    )
    # Filtering out of bounds coords
    outline_df = outline_df.query(
        f"z >= 0 and z < {arr_ref.shape[0]} and y >= 0 and y < {arr_ref.shape[1]} and x >= 0 and x < {arr_ref.shape[2]}"
    )

    # Make outline img (1 for in, 2 for out)
    coords2points(
        outline_df[outline_df.is_in == 1],
        arr_ref.shape,
        pfm.outline,
    )
    outline_in = tifffile.imread(pfm.outline)
    coords2points(
        outline_df[outline_df.is_in == 0],
        arr_ref.shape,
        pfm.outline,
    )
    outline_out = tifffile.imread(pfm.outline)
    tifffile.imwrite(pfm.outline, outline_in + outline_out * 2)

    # Fill in outline to recreate mask (not perfect)
    arr_mask_reg = fill_outline(arr_ref, outline_df)
    # Opening (removes FP) and closing (fills FN)
    arr_mask_reg = ndimage.binary_closing(arr_mask_reg, iterations=2).astype(np.uint8)
    arr_mask_reg = ndimage.binary_opening(arr_mask_reg, iterations=2).astype(np.uint8)
    # Saving
    tifffile.imwrite(pfm.mask_reg, arr_mask_reg)

    # Counting mask voxels in each region
    arr_annot = tifffile.imread(pfm.annot)
    with open(pfm.map, "r") as f:
        annot_df = annot_dict2df(json.load(f))
    # Getting the annotation name for every cell (zyx coord)
    mask_df = pd.merge(
        left=mask2region_counts(np.full(arr_annot.shape, 1), arr_annot),
        right=mask2region_counts(arr_mask_reg, arr_annot),
        how="left",
        left_index=True,
        right_index=True,
        suffixes=("_annot", "_mask"),
    ).fillna(0)
    # Combining (summing) the cells_agg_df values for parent regions using the annot_df
    mask_df = combine_nested_regions(mask_df, annot_df)
    # Calculating proportion of mask volume in each region
    mask_df[MaskColumns.VOLUME_PROP.value] = (
        mask_df[MaskColumns.VOLUME_MASK.value] / mask_df[MaskColumns.VOLUME_ANNOT.value]
    )
    # Selecting and ordering relevant columns
    mask_df = mask_df[[*ANNOT_COLUMNS_FINAL, *enum2list(MaskColumns)]]
    # Saving
    mask_df.to_parquet(pfm.mask_df)


# @flow
def img_overlap_pipeline(
    pfm: ProjFpModel,
    **kwargs,
):
    # Update registration params json
    configs = ConfigParamsModel.update_params_file(pfm.config_params, **kwargs)
    # Making overlap image
    with cluster_proc_contxt(LocalCluster(n_workers=1, threads_per_worker=4)):
        arr_raw = da.from_zarr(pfm.raw, chunks=configs.chunksize)
        arr_overlap = da_overlap(arr_raw, d=configs.de)
        arr_overlap = disk_cache(arr_overlap, pfm.overlap)


# @flow
def cell_count_pipeline(
    pfm: ProjFpModel,
    **kwargs,
):
    # Update registration params json
    configs = ConfigParamsModel.update_params_file(pfm.config_params, **kwargs)

    # Reading raw image
    arr_raw = da.from_zarr(pfm.raw)
    # Reading overlapped image
    arr_overlap = da.from_zarr(pfm.overlap)

    with cluster_proc_contxt(LocalCUDACluster()):
        # Step 1: Top-hat filter (background subtraction)
        arr_bgrm = da.map_blocks(Gf.tophat_filt, arr_overlap, configs.tophat_sigma)
        arr_bgrm = disk_cache(arr_bgrm, pfm.bgrm)

        # Step 2: Difference of Gaussians (edge detection)
        arr_dog = da.map_blocks(
            Gf.dog_filt, arr_bgrm, configs.dog_sigma1, configs.dog_sigma2
        )
        arr_dog = disk_cache(arr_dog, pfm.dog)

        # Step 3: Gaussian subtraction with large sigma for adaptive thresholding
        arr_adaptv = da.map_blocks(Gf.gauss_subt_filt, arr_dog, configs.gauss_sigma)
        arr_adaptv = disk_cache(arr_adaptv, pfm.adaptv)

    with cluster_proc_contxt(LocalCluster()):
        # Step 4: Mean thresholding with standard deviation offset
        # # Visually inspect sd offset
        # t_p = arr_adaptv.sum() / (np.prod(arr_adaptv.shape) - (arr_adaptv == 0).sum())
        # t_p = t_p.compute()
        # logging.debug(t_p)
        arr_threshd = da.map_blocks(Cf.manual_thresh, arr_adaptv, configs.thresh_p)
        arr_threshd = disk_cache(arr_threshd, pfm.threshd)

    with cluster_proc_contxt(LocalCluster(n_workers=6, threads_per_worker=1)):
        # Step 5: Object sizes
        arr_sizes = da.map_blocks(Cf.label_with_sizes, arr_threshd)
        arr_sizes = disk_cache(arr_sizes, pfm.threshd_sizes)

    with cluster_proc_contxt(LocalCluster()):
        # Step 6: Filter out large objects (likely outlines, not cells)
        arr_threshd_filt = da.map_blocks(
            Cf.filt_by_size, arr_sizes, configs.min_threshd, configs.max_threshd
        )
        arr_threshd_filt = disk_cache(arr_threshd_filt, pfm.threshd_filt)

    with cluster_proc_contxt(LocalCUDACluster()):
        # Step 7: Get maxima of image masked by labels
        arr_maxima = da.map_blocks(
            Gf.get_local_maxima, arr_overlap, configs.maxima_sigma, arr_threshd_filt
        )
        arr_maxima = disk_cache(arr_maxima, pfm.maxima)

    with cluster_proc_contxt(LocalCluster(n_workers=3, threads_per_worker=1)):
        # n_workers=2
        # Step 8: Watershed segmentation sizes
        arr_wshed_sizes = da.map_blocks(
            Cf.wshed_segm_sizes, arr_overlap, arr_maxima, arr_threshd_filt
        )
        arr_wshed_sizes = disk_cache(arr_wshed_sizes, pfm.wshed_sizes)

    with cluster_proc_contxt(LocalCluster()):
        # Step 9: Filter out large watershed objects (again likely outlines, not cells)
        arr_wshed_filt = da.map_blocks(
            Cf.filt_by_size, arr_wshed_sizes, configs.min_wshed, configs.max_wshed
        )
        arr_wshed_filt = disk_cache(arr_wshed_filt, pfm.wshed_filt)

        # Step 10: Trimming filtered regions overlaps
        # Trimming maxima points overlaps
        arr_maxima_f = da_trim(arr_maxima, d=configs.d)
        disk_cache(arr_maxima_f, pfm.maxima_final)
        # Trimming filtered regions overlaps
        arr_threshd_final = da_trim(arr_threshd_filt, d=configs.d)
        disk_cache(arr_threshd_final, pfm.threshd_final)
        # Trimming watershed sizes overlaps
        arr_wshed_final = da_trim(arr_wshed_sizes, d=configs.d)
        disk_cache(arr_wshed_final, pfm.wshed_final)

    with cluster_proc_contxt(LocalCluster(n_workers=2, threads_per_worker=1)):
        # n_workers=2
        # Step 11: From maxima and watershed, save the cells
        # (coords, size, and intensity) to a df
        # Getting maxima coords and corresponding watershed sizes in table
        cells_df = block2coords(
            Cf.get_cells, arr_raw, arr_overlap, arr_maxima, arr_wshed_filt, configs.d
        )
        # Filtering out by size
        cells_df = cells_df.query(
            f"size >= {configs.min_wshed} and size <= {configs.max_wshed}"
        )
        # Saving to parquet
        cells_df.to_parquet(pfm.cells_raw_df, overwrite=True)


def cell_count_coords_only_pipeline(
    pfm: ProjFpModel,
    **kwargs,
):
    # Update registration params json
    configs = ConfigParamsModel.update_params_file(pfm.config_params, **kwargs)
    # Reading filtered and maxima images (trimmed - orig space)
    with cluster_proc_contxt(LocalCluster(n_workers=6, threads_per_worker=1)):
        # Read filtered and maxima images (trimmed - orig space)
        arr_maxima_f = da.from_zarr(pfm.maxima_final)
        # Step 10a: Get coords of maxima and get corresponding sizes from watershed
        coords_df = block2coords(Gf.get_coords, arr_maxima_f)
        coords_df.to_parquet(pfm.maxima_df, overwrite=True)


# @flow
def transform_coords(
    pfm: ProjFpModel,
):
    """
    `in_id` and `out_id` are either maxima or region

    NOTE: saves the cells_trfm dataframe as pandas parquet.
    """
    with cluster_proc_contxt(LocalCluster(n_workers=4, threads_per_worker=1)):
        # Getting registration parameters
        configs = ConfigParamsModel.model_validate(read_json(pfm.config_params))
        # Setting output key (in the form "<maxima/region>_trfm_df")
        # Getting cell coords
        cells_df = dd.read_parquet(pfm.cells_raw_df).compute()
        # Sanitising (removing smb columns)
        cells_df = sanitise_smb_df(cells_df)
        # Taking only Coords.Z.value, Coords.Y.value, Coords.X.value coord columns
        cells_df = cells_df[[Coords.Z.value, Coords.Y.value, Coords.X.value]]
        # Scaling to resampled rough space
        # NOTE: this downsampling uses slicing so must be computed differently
        cells_df = cells_df / np.array(
            (configs.z_rough, configs.y_rough, configs.x_rough)
        )
        # Scaling to resampled space
        cells_df = cells_df * np.array((configs.z_fine, configs.y_fine, configs.x_fine))
        # Trimming/offsetting to sliced space
        cells_df = cells_df - np.array(
            [
                s[0] if s[0] else 0
                for s in (configs.z_trim, configs.y_trim, configs.x_trim)
            ]
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
def get_cell_mappings(
    pfm: ProjFpModel,
    **kwargs,
):
    """
    Using the transformed cell coordinates, get the region ID and name for each cell
    corresponding to the reference atlas.

    NOTE: saves the cells dataframe as pandas parquet.
    """
    # Update registration params json
    configs = ConfigParamsModel.update_params_file(pfm.config_params, **kwargs)
    # Getting region for each detected cell (i.e. row) in cells_df
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


def grouping_cells(
    pfm: ProjFpModel,
    **kwargs,
):
    """
    Grouping cells by region name and aggregating total cell volume
    and cell count for each region.

    NOTE: saves the cells_agg dataframe as pandas parquet.
    """
    # Update registration params json
    configs = ConfigParamsModel.update_params_file(pfm.config_params, **kwargs)
    # Making cells_agg_df
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


def cells2csv(
    pfm: ProjFpModel,
    **kwargs,
):
    # Update registration params json
    configs = ConfigParamsModel.update_params_file(pfm.config_params, **kwargs)
    # Reading cells dataframe
    cells_agg_df = pd.read_parquet(pfm.cells_agg_df)
    # Sanitising (removing smb columns)
    cells_agg_df = sanitise_smb_df(cells_agg_df)
    # Saving to csv
    cells_agg_df.to_csv(pfm.cells_agg_csv)
